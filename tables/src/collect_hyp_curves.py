"""Hypothesis figure fig_h_curves: training dynamics of MSCA and MSCA + XRE (validation Recall@20 per epoch).

Hypothesis. XRE does not win by training longer: its selected (best-validation) epoch is not later than that of MSCA, and
its validation curve lies above the MSCA curve at matched epochs, reaching the MSCA peak no later than MSCA does.
Runs = the Table I runs, one run per seed, seeds 1-5 (HOST_A)
  MSCA        tables/table1_runs.tsv rows MSCA Native with chosen = 1 (objx_s <ds>_native; collect_table1.py)
  MSCA + XRE  tables/xre_runs.tsv rows MSCA XRE, eligible (xres <ds>_xres_xre_l010_k40, replicate 1; collect_xre.py)
  Datasets Baby, Pantry, Scientific, Sports.
Per run, the trainer writes <run>/<train_time>/<train_time>_train_val_loss_log.csv (one line per epoch, 1-based: column
'val_criterion recall@20' = validation Recall@20 of the epoch, 4 dp; 'is_best_epoch' > 0 when it improves the best so far).
This script reads it over ssh and checks that the last improving epoch equals best_epoch of result.json and that its
validation Recall@20 equals best_valid_score (stops otherwise).
Writes
  tables/hyp_curves_epochs.csv   dataset, arm, seed, epoch, val_r20 (every logged epoch of every run)
  tables/hyp_curves_runs.csv     per run: best_epoch, best val R@20, last epoch, first epoch reaching the MSCA best of the
                                 same seed (XRE rows), MSCA val at the XRE best epoch of the same seed (XRE rows)
  tables/hyp_curves_summary.csv  per dataset: mean best epoch of both arms, seeds with XRE best epoch <= MSCA best epoch,
                                 mean best val R@20, matched-epoch gap (XRE best minus MSCA at the XRE selected epoch),
                                 epochs XRE needs to reach the MSCA best
  tables/macros_hyp_curves.tex   hyCurves{Msca,Xre}Best<DS> (mean selected epoch, 1 dp), hyCurvesEarlier<DS> (seeds with
                                 XRE best epoch <= MSCA best epoch), hyCurvesReach<DS> (mean first epoch at which XRE
                                 reaches the MSCA best of the same seed, 1 dp), hyCurvesReachSeeds<DS>, hyCurvesMatched<DS>
                                 (XRE best val R@20 over the MSCA val R@20 at the XRE selected epoch of the same seed, %, 1 dp),
                                 hyCurvesMatchedSeeds<DS> (seeds with a positive gap), hyCurvesValGain<DS> (best val gain, %)
Usage: python -s tables/src/collect_hyp_curves.py      (then python -s figures/src/fig_h_curves.py)
"""
import csv
import statistics as st
import subprocess
import sys
from pathlib import Path

TABLES = Path(__file__).resolve().parents[1]
DATASETS = [('baby', 'Baby'), ('pantry', 'Pantry'), ('scientific', 'Scientific'), ('sports', 'Sports')]
SEEDS = [1, 2, 3, 4, 5]
SSH = ['ssh', '-o', 'BatchMode=yes', 'HOST_A']
REMOTE = r'''
import csv, glob, json, os, sys
for f in sys.argv[1:]:
    d = os.path.dirname(f); r = json.load(open(f))
    logs = glob.glob(os.path.join(d, '*', '*_train_val_loss_log.csv'))
    if len(logs) != 1: print('ERR', f, len(logs), sep='\t'); continue
    print('RUN', f, r.get('best_epoch'), r.get('best_valid_score'), r.get('gpu'), sep='\t')
    for row in csv.DictReader(open(logs[0])):
        print('EP', f, row['epoch'], row['val_criterion recall@20'], row['is_best_epoch'], sep='\t')
'''


def runs():
    out = {}
    with open(TABLES / 'table1_runs.tsv', newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            if r['backbone'] == 'MSCA' and r['row'] == 'Native' and r['chosen'] == '1' and not r['ineligible_reason']:
                out[(r['dataset'], 'msca', int(r['seed']))] = r['path']
    with open(TABLES / 'xre_runs.tsv', newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            if r['backbone'] == 'MSCA' and r['row'] == 'XRE' and not r['ineligible_reason']:
                k = (r['dataset'], 'xre', int(r['seed']))
                if k in out:
                    raise SystemExit(f'two eligible XRE runs for {k}: one run per seed is required')
                out[k] = r['path']
    need = {(ds, a, s) for ds, _ in DATASETS for a in ('msca', 'xre') for s in SEEDS}
    if need - set(out):
        raise SystemExit(f'missing runs: {sorted(need - set(out))}')
    return {k: out[k] for k in need}


def main():
    paths = runs()
    res = subprocess.run(SSH + ['python3', '-'] + sorted(paths.values()), input=REMOTE.encode(), capture_output=True,
                         check=True)
    key_of = {p: k for k, p in paths.items()}
    meta, curve = {}, {k: {} for k in paths}
    for line in res.stdout.decode().splitlines():
        p = line.split('\t')
        if p[0] == 'ERR':
            raise SystemExit(f'no unique validation log: {p[1:]}')
        k = key_of[p[1]]
        if p[0] == 'RUN':
            meta[k] = (int(p[2]), float(p[3]), p[4].replace('NVIDIA ', ''))
        else:
            curve[k][int(p[2])] = (float(p[3]), int(p[4]))
    ep_rows, run_rows = [], []
    for k in sorted(paths):
        ds, arm, s = k
        c = curve[k]
        be, bv, gpu = meta[k]
        last_imp = max(e for e, (_, b) in c.items() if b > 0)
        if last_imp != be or abs(c[be][0] - bv) > 1e-9:
            raise SystemExit(f'{k}: log best epoch {last_imp} ({c[last_imp][0]}) != result.json {be} ({bv})')
        ep_rows += [dict(dataset=ds, arm=arm, seed=s, epoch=e, val_r20=f'{v:.4f}') for e, (v, _) in sorted(c.items())]
    for k in sorted(paths):
        ds, arm, s = k
        be, bv, gpu = meta[k]
        row = dict(dataset=ds, arm=arm, seed=s, gpu=gpu, best_epoch=be, best_val=f'{bv:.4f}', last_epoch=max(curve[k]),
                   reach_msca_best='', msca_val_at_xre_best='', path=paths[k])
        if arm == 'xre':
            mbe, mbv, _ = meta[(ds, 'msca', s)]
            c = curve[k]
            reach = [e for e in sorted(c) if c[e][0] >= mbv]
            row['reach_msca_best'] = reach[0] if reach else ''
            mc = curve[(ds, 'msca', s)]
            row['msca_val_at_xre_best'] = f'{mc[be][0]:.4f}' if be in mc else ''
        run_rows.append(row)
    write(TABLES / 'hyp_curves_epochs.csv', ep_rows)
    write(TABLES / 'hyp_curves_runs.csv', run_rows)
    R = {(r['dataset'], r['arm'], r['seed']): r for r in run_rows}
    summ, mac = [], []
    for ds, D in DATASETS:
        mb = [R[(ds, 'msca', s)]['best_epoch'] for s in SEEDS]
        xb = [R[(ds, 'xre', s)]['best_epoch'] for s in SEEDS]
        mv = [float(R[(ds, 'msca', s)]['best_val']) for s in SEEDS]
        xv = [float(R[(ds, 'xre', s)]['best_val']) for s in SEEDS]
        reach = [R[(ds, 'xre', s)]['reach_msca_best'] for s in SEEDS]
        gap = [100 * (xv[i] / float(R[(ds, 'xre', s)]['msca_val_at_xre_best']) - 1) for i, s in enumerate(SEEDS)
               if R[(ds, 'xre', s)]['msca_val_at_xre_best'] != '']
        rr = [e for e in reach if e != '']
        summ.append(dict(dataset=ds, msca_best_epoch=f'{st.mean(mb):.1f}', xre_best_epoch=f'{st.mean(xb):.1f}',
                         msca_best_epochs='/'.join(map(str, mb)), xre_best_epochs='/'.join(map(str, xb)),
                         seeds_xre_not_later=sum(x <= m for x, m in zip(xb, mb)),
                         msca_best_val=f'{st.mean(mv):.4f}', xre_best_val=f'{st.mean(xv):.4f}',
                         best_val_gain_pct=f'{100 * (st.mean(xv) / st.mean(mv) - 1):.2f}',
                         matched_gap_pct=f'{st.mean(gap):.2f}' if gap else '', matched_gap_pos=sum(g > 0 for g in gap),
                         matched_n=len(gap), reach_epoch_mean=f'{st.mean(rr):.1f}' if rr else '', reach_seeds=len(rr),
                         reach_epochs='/'.join(str(e) for e in reach)))
        x = summ[-1]
        mac += [(f'hyCurvesMscaBest{D}', x['msca_best_epoch']), (f'hyCurvesXreBest{D}', x['xre_best_epoch']),
                (f'hyCurvesEarlier{D}', str(x['seeds_xre_not_later'])), (f'hyCurvesReach{D}', x['reach_epoch_mean']),
                (f'hyCurvesReachSeeds{D}', str(x['reach_seeds'])), (f'hyCurvesMatched{D}', f"{float(x['matched_gap_pct']):.1f}"),
                (f'hyCurvesMatchedSeeds{D}', f"{x['matched_gap_pos']}/{x['matched_n']}"),
                (f'hyCurvesValGain{D}', f"{float(x['best_val_gain_pct']):.1f}")]
    write(TABLES / 'hyp_curves_summary.csv', summ)
    mac.append(('hyCurvesEarlierAll', f"{sum(int(x['seeds_xre_not_later']) for x in summ)}/{len(SEEDS) * len(DATASETS)}"))
    lines = ['% Generated by tables/src/collect_hyp_curves.py from the per-epoch validation logs of the Table I MSCA and',
             '% MSCA + XRE runs (HOST_A, seeds 1-5, one run per seed). Do not edit by hand. Epochs are 1-based means over seeds;',
             '% Matched = XRE best val Recall@20 minus the MSCA val Recall@20 at the XRE selected epoch, same seed (% of it);',
             '% Reach = first epoch at which XRE reaches the MSCA best of the same seed; ValGain = best val gain (%).']
    lines += [f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in mac]
    (TABLES / 'macros_hyp_curves.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
    for x in summ:
        print(x)


def write(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


if __name__ == '__main__':
    main()
