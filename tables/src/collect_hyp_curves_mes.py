"""MES version of tables/src/collect_hyp_curves.py (PREREG_MESCKPT unit U5, 2026-10-16): training dynamics of MSCA and
MSCA + MES, validation Recall@20 per epoch, five seeds, four datasets.

Runs: MSCA = tables/table1_runs.tsv rows MSCA Native chosen = 1 (as the original); MES = tables/mes_perseed.csv rows
backbone MSCA, row MES (xres0, the MAIN-table setting per dataset: Baby l030_k40, Pantry l010_k20, Scientific l020_k40, Sports
l010_k80), run_dir + /result.json. Same checks as the original: the last improving epoch of the validation log must equal best_epoch
of result.json and its validation Recall@20 must equal best_valid_score (a run that fails is reported and aborts the unit).

Workflow (no ssh from here):
  local   python tables/src/collect_hyp_curves_mes.py --paths <paths.txt>          (list of result.json paths to read)
  server  python collect_hyp_curves_mes.py remote $(cat paths.txt) > out/curves_raw.tsv       (CPU, reads logs only)
  local   python tables/src/collect_hyp_curves_mes.py --raw curves_raw.tsv         (writes the files below)
Writes tables/src/data/hyp_curves_{epochs,runs,summary}_mes.csv and tables/macros_hyp_curves_mes.tex (macro names of the
E + R + A version are kept, hyCurvesXre*, values are MES; arm value 'mes'; column names xre_* -> mes_*).
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
    with open(TABLES / 'mes_perseed.csv', newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['backbone'] == 'MSCA' and r['row'] == 'MES':
                k = (r['dataset'], 'mes', int(r['seed']))
                if k in out:
                    raise SystemExit(f'two MES runs for {k}: one run per seed is required')
                out[k] = r['run_dir'].rstrip('/') + '/result.json'
    need = {(ds, a, s) for ds, _ in DATASETS for a in ('msca', 'mes') for s in SEEDS}
    if need - set(out):
        raise SystemExit(f'missing runs: {sorted(need - set(out))}')
    return {k: out[k] for k in need}


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'remote':
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        exec(REMOTE)
        return
    paths = runs()
    if '--paths' in sys.argv:
        Path(sys.argv[sys.argv.index('--paths') + 1]).write_text('\n'.join(sorted(paths.values())) + '\n', encoding='utf-8')
        print('wrote', len(paths), 'paths')
        return
    raw = open(sys.argv[sys.argv.index('--raw') + 1], encoding='utf-8').read()
    key_of = {p: k for k, p in paths.items()}
    meta, curve = {}, {k: {} for k in paths}
    for line in raw.splitlines():
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
                   reach_msca_best='', msca_val_at_mes_best='', path=paths[k])
        if arm == 'mes':
            mbe, mbv, _ = meta[(ds, 'msca', s)]
            c = curve[k]
            reach = [e for e in sorted(c) if c[e][0] >= mbv]
            row['reach_msca_best'] = reach[0] if reach else ''
            mc = curve[(ds, 'msca', s)]
            row['msca_val_at_mes_best'] = f'{mc[be][0]:.4f}' if be in mc else ''
        run_rows.append(row)
    write(TABLES / 'src/data/hyp_curves_epochs_mes.csv', ep_rows)
    write(TABLES / 'src/data/hyp_curves_runs_mes.csv', run_rows)
    R = {(r['dataset'], r['arm'], r['seed']): r for r in run_rows}
    summ, mac = [], []
    for ds, D in DATASETS:
        mb = [R[(ds, 'msca', s)]['best_epoch'] for s in SEEDS]
        xb = [R[(ds, 'mes', s)]['best_epoch'] for s in SEEDS]
        mv = [float(R[(ds, 'msca', s)]['best_val']) for s in SEEDS]
        xv = [float(R[(ds, 'mes', s)]['best_val']) for s in SEEDS]
        reach = [R[(ds, 'mes', s)]['reach_msca_best'] for s in SEEDS]
        gap = [100 * (xv[i] / float(R[(ds, 'mes', s)]['msca_val_at_mes_best']) - 1) for i, s in enumerate(SEEDS)
               if R[(ds, 'mes', s)]['msca_val_at_mes_best'] != '']
        rr = [e for e in reach if e != '']
        summ.append(dict(dataset=ds, msca_best_epoch=f'{st.mean(mb):.1f}', mes_best_epoch=f'{st.mean(xb):.1f}',
                         msca_best_epochs='/'.join(map(str, mb)), mes_best_epochs='/'.join(map(str, xb)),
                         seeds_mes_not_later=sum(x <= m for x, m in zip(xb, mb)),
                         msca_best_val=f'{st.mean(mv):.4f}', mes_best_val=f'{st.mean(xv):.4f}',
                         best_val_gain_pct=f'{100 * (st.mean(xv) / st.mean(mv) - 1):.2f}',
                         matched_gap_pct=f'{st.mean(gap):.2f}' if gap else '', matched_gap_pos=sum(g > 0 for g in gap),
                         matched_n=len(gap), reach_epoch_mean=f'{st.mean(rr):.1f}' if rr else '', reach_seeds=len(rr),
                         reach_epochs='/'.join(str(e) for e in reach)))
        x = summ[-1]
        mac += [(f'hyCurvesMscaBest{D}', x['msca_best_epoch']), (f'hyCurvesXreBest{D}', x['mes_best_epoch']),
                (f'hyCurvesEarlier{D}', str(x['seeds_mes_not_later'])), (f'hyCurvesReach{D}', x['reach_epoch_mean']),
                (f'hyCurvesReachSeeds{D}', str(x['reach_seeds'])), (f'hyCurvesMatched{D}', f"{float(x['matched_gap_pct']):.1f}"),
                (f'hyCurvesMatchedSeeds{D}', f"{x['matched_gap_pos']}/{x['matched_n']}"),
                (f'hyCurvesValGain{D}', f"{float(x['best_val_gain_pct']):.1f}")]
    write(TABLES / 'src/data/hyp_curves_summary_mes.csv', summ)
    mac.append(('hyCurvesEarlierAll', f"{sum(int(x['seeds_mes_not_later']) for x in summ)}/{len(SEEDS) * len(DATASETS)}"))
    lines = ['% Generated by tables/src/collect_hyp_curves_mes.py (MES; macro names kept) from the per-epoch validation logs of the MSCA and',
             '% MSCA + MES runs (HOST_A, seeds 1-5, one run per seed). Do not edit by hand. Epochs are 1-based means over seeds;',
             '% Matched = MES best val Recall@20 minus the MSCA val Recall@20 at the XRE selected epoch, same seed (% of it);',
             '% Reach = first epoch at which MES reaches the MSCA best of the same seed; ValGain = best val gain (%).']
    lines += [f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in mac]
    (TABLES / 'macros_hyp_curves_mes.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
    for x in summ:
        print(x)


def write(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


if __name__ == '__main__':
    main()
