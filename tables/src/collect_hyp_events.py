"""Hypothesis figure fig_h_events: exchange events during training, masked view versus full view (Lemma lem:masked).

Hypothesis. The masked score carries no self-message (Lemma lem:masked). On the full view the self-message lifts the
held-out positives of a fold into the top-K list during training, so entry events (held-out positive outside the list)
fade and the exchange coefficient mass moves to retention pairs. The masked view keeps entry events alive.

Runs (MSCA, lambda 0.1, K 40, ONE run per seed, replicate 1 only, seeds 1-5; user rule: no replicates)
  pair 'exonly'  (the figure): exchange term alone, no anchor term = rows of Table tab:xmask
        masked view = row wo_A1 (xres <ds>_xres_xro_l010_k40), full view = row wo_A1_masked (xres <ds>_xres_xfo_l010_k40)
  pair 'anchor'  (robustness, CSV and macros only): XRE (masked view, xres <ds>_xres_xre_l010_k40) versus XRE on the full
        view (row wo_masked, xres <ds>_xres_xrf_l010_k40)
Sources
  Baby, Pantry, Scientific: tables/xre_events.tsv, written by tables/src/collect_xre.py (REPLICATES = 'rep1'): the
        [R12]/[XRE] monitor line that every exchange run prints every 500 training steps (HOST_A replicate-1 roots qxs2/A,
        qxs4/A). This script stops if a run of a replicate-2 root (qxs2r, qxs4r) or a second run of a seed appears.
        Selected (best-validation) epochs: column best_epoch of tables/xre_runs.tsv (eligible runs).
  Sports (pair exonly only): HOST_B <RUN_ROOT_C>/P04_sports_abl/runs_abl/xres (one run per seed, ablation
        package 6003cba), read over ssh from its HOST_A copy <RUN_ROOT_B>/P04_sports_abl_HOST_B/runs_abl/xres
        (synced with the commands of the README.txt of that work dir; all 45 jobs finished rc 0 on 2026-10-07 00:30). Sports enters only when both rows have a SUCCESS result.json for all of seeds 1-5;
        otherwise the figure keeps three datasets (state on 2026-10-06 22:07 KST: seeds 1-2 finished, 3-5 running).
Monitor fields (computed on the batch of the logged step, rows whose positive belongs to the held-out fold)
  p_in_list   share of those rows whose held-out positive is already inside the top-K list of the view (no entry event)
  mass_entry / mass_retention   summed detached sigmoid(gap) coefficients of the entry and retention terms
              (full view: before the per-batch mass rescaling); entry share = mass_entry / (mass_entry + mass_retention)
Epoch of a line = step // ceil(train interactions / 256) + 1 (1-based, as best_epoch and the validation log). A run value
per epoch = mean of its lines in that epoch; the curve = mean over the runs still training (drawn while >= MIN_RUNS runs);
the band = min-max over those runs.
Writes
  tables/hyp_events_epochs.csv   dataset, pair, view, epoch, n_runs, listed / entry-share mean, min, max
  tables/hyp_events_seed.csv     per run: listed share and entry share in the first epoch, at the selected epoch, and over
                                 the last five logged epochs
  tables/hyp_events_summary.csv  per dataset, pair, view: means over seeds of the per-run values; per dataset and pair the
                                 same-seed difference full minus masked (mean, seeds with full > masked, paired t-test p)
  tables/macros_hyp_events.tex   macros of the paragraph (percent, integers): hyEvents<View><Quantity><DS>
                                 View Mask/Full, Quantity Listed (positive in the list) / Entry (entry share of the mass),
                                 at the selected epoch; ...First<DS> in the first epoch; Anchor... = pair 'anchor'.
Usage: python -s tables/src/collect_hyp_events.py      (then python -s figures/src/fig_h_events.py)
"""
import csv
import math
import statistics as st
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from scipy import stats

TABLES = Path(__file__).resolve().parents[1]
SRC = TABLES / 'xre_events.tsv'
RUNS = TABLES / 'xre_runs.tsv'
DATASETS = ['baby', 'pantry', 'scientific']
SEEDS = [1, 2, 3, 4, 5]
PAIRS = {'exonly': {'wo_A1': 'mask', 'wo_A1_masked': 'full'}, 'anchor': {'XRE': 'mask', 'wo_masked': 'full'}}
REP2 = ('/qxs2r/', '/qxs4r/')
MIN_RUNS = 3
SPORTS_HOST = 'HOST_A'   # synced copy (2026-10-07) of HOST_B <RUN_ROOT_C>/P04_sports_abl/runs_abl/xres
SPORTS_DIR = '<RUN_ROOT_B>/P04_sports_abl_HOST_B/runs_abl/xres'
SPORTS_CFG = {'sports_xres_xro_l010_k40': 'wo_A1', 'sports_xres_xfo_l010_k40': 'wo_A1_masked'}
DSNAME = {'baby': 'Baby', 'pantry': 'Pantry', 'scientific': 'Scientific', 'sports': 'Sports'}

REMOTE = r'''
import glob, json, os, re, sys
D, CFG = sys.argv[1], sys.argv[2].split(',')
head = re.compile(r'history_keys=(\d+)'); step = re.compile(r'\[(?:R12|XRE)\] step=(\d+) (.*)')
for f in sorted(glob.glob(os.path.join(D, '*', 'result.json'))):
    name = os.path.basename(os.path.dirname(f)); parts = name.split('__')
    if len(parts) < 4 or parts[1] not in CFG: continue
    try: d = json.load(open(f))
    except Exception: continue
    print('RUN', parts[1], parts[3].lstrip('s'), d.get('status'), d.get('best_epoch'), d.get('gpu'), f, sep='\t')
    log = os.path.join(os.path.dirname(f), 'stdout.log'); n = ''
    if d.get('status') != 'SUCCESS' or not os.path.isfile(log): continue
    for chunk in open(log, encoding='utf-8', errors='replace'):
        for line in chunk.split('\r'):
            m = head.search(line)
            if m and not n: n = m.group(1)
            m = step.search(line)
            if m:
                kv = dict(x.split('=', 1) for x in m.group(2).split() if '=' in x)
                print('EV', parts[1], parts[3].lstrip('s'), n, m.group(1), kv.get('p_in_list', ''), kv.get('mass_entry', ''),
                      kv.get('mass_retention', ''), sep='\t')
'''


def epoch_of(step, n_train):
    return int(step) // math.ceil(int(n_train) / 256) + 1


def load_local():
    best = {}
    with open(RUNS, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            if r['backbone'] == 'MSCA' and not r['ineligible_reason']:
                best[r['path']] = int(r['best_epoch'])
    ev = defaultdict(lambda: defaultdict(list))   # (ds, pair, view, seed) -> epoch -> [(pin, share)]
    meta = {}
    with open(SRC, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            if r['backbone'] != 'MSCA' or r['dataset'] not in DATASETS:
                continue
            if r['setting'] != 'l010_k40/rep1':
                raise SystemExit(f"xre_events.tsv setting {r['setting']}: re-run collect_xre.py with REPLICATES = 'rep1'")
            if any(x in r['run'] for x in REP2):
                raise SystemExit(f"replicate-2 run in xre_events.tsv: {r['run']}")
            for pair, rows in PAIRS.items():
                if r['row'] not in rows:
                    continue
                key = (r['dataset'], pair, rows[r['row']], int(r['seed']))
                if key in meta and meta[key][0] != r['run']:
                    raise SystemExit(f'two runs for {key}: one run per seed is required')
                meta[key] = (r['run'], best[r['run']], 'GPU')
                me, mr = float(r['mass_entry']), float(r['mass_retention'])
                ev[key][epoch_of(r['step'], r['train_interactions'])].append(
                    (float(r['p_in_list']), me / (me + mr) if me + mr > 0 else 0.0))
    return ev, meta


def load_sports(ev, meta):
    try:
        out = subprocess.run(['ssh', '-o', 'BatchMode=yes', SPORTS_HOST, 'python3', '-', SPORTS_DIR, ','.join(SPORTS_CFG)],
                             input=REMOTE.encode(), capture_output=True, check=True, timeout=600)
    except Exception as e:   # host unreachable: keep three datasets
        print(f'Sports not read ({type(e).__name__})', file=sys.stderr)
        return False
    runs, lines = {}, defaultdict(list)
    for line in out.stdout.decode().splitlines():
        p = line.split('\t')
        if p[0] == 'RUN':
            k = (SPORTS_CFG[p[1]], int(p[2]))
            if k in runs and runs[k][0] == 'SUCCESS':
                raise SystemExit(f'two Sports runs for {k}: one run per seed is required')
            runs[k] = (p[3], p[4], p[5], p[6])
        elif p[0] == 'EV':
            lines[(SPORTS_CFG[p[1]], int(p[2]))].append(p[3:])
    done = {k for k, v in runs.items() if v[0] == 'SUCCESS'}
    need = {(row, s) for row in SPORTS_CFG.values() for s in SEEDS}
    print(f'Sports: {len(done & need)}/{len(need)} runs finished', file=sys.stderr)
    if need - done:
        return False
    for (row, s), ls in lines.items():
        view = PAIRS['exonly'][row]
        key = ('sports', 'exonly', view, s)
        meta[key] = (runs[(row, s)][3], int(runs[(row, s)][1]), runs[(row, s)][2].replace('NVIDIA ', ''))
        for n, step, pin, me, mr in ls:
            me, mr = float(me), float(mr)
            ev[key][epoch_of(step, n)].append((float(pin), me / (me + mr) if me + mr > 0 else 0.0))
    return True


def per_run(ev):
    """(ds, pair, view, seed) -> epoch -> (pin, share): mean of the lines of the run within the epoch"""
    return {k: {e: (st.mean(x[0] for x in v), st.mean(x[1] for x in v)) for e, v in d.items()} for k, d in ev.items()}


def main():
    ev, meta = load_local()
    sports = load_sports(ev, meta)
    datasets = DATASETS + (['sports'] if sports else [])
    pr = per_run(ev)
    # per-epoch curves
    rows_ep = []
    for ds in datasets:
        for pair in PAIRS:
            for view in ('mask', 'full'):
                runs = {s: pr[(ds, pair, view, s)] for s in SEEDS if (ds, pair, view, s) in pr}
                if not runs:
                    continue
                if len(runs) != len(SEEDS):
                    raise SystemExit(f'{ds} {pair} {view}: {len(runs)} runs, need seeds 1-5')
                for e in sorted({e for d in runs.values() for e in d}):
                    vals = [d[e] for d in runs.values() if e in d]
                    if len(vals) < MIN_RUNS:
                        continue
                    rows_ep.append(dict(dataset=ds, pair=pair, view=view, epoch=e, n_runs=len(vals),
                                        listed_mean=f'{st.mean(v[0] for v in vals):.4f}',
                                        listed_min=f'{min(v[0] for v in vals):.4f}', listed_max=f'{max(v[0] for v in vals):.4f}',
                                        entry_mean=f'{st.mean(v[1] for v in vals):.4f}',
                                        entry_min=f'{min(v[1] for v in vals):.4f}', entry_max=f'{max(v[1] for v in vals):.4f}'))
    write(TABLES / 'hyp_events_epochs.csv', rows_ep)
    # per-run values
    rows_seed = []
    for (ds, pair, view, s), d in sorted(pr.items()):
        if ds not in datasets:
            continue
        eps = sorted(d)
        run, be, gpu = meta[(ds, pair, view, s)]
        sel = d.get(be) or d[max(e for e in eps if e <= be)]   # selected epoch (a monitor line falls in every epoch)
        last = eps[-5:]
        rows_seed.append(dict(dataset=ds, pair=pair, view=view, seed=s, gpu=gpu, best_epoch=be, last_epoch=eps[-1],
                              listed_first=f'{d[eps[0]][0]:.4f}', entry_first=f'{d[eps[0]][1]:.4f}',
                              listed_sel=f'{sel[0]:.4f}', entry_sel=f'{sel[1]:.4f}',
                              listed_last5=f'{st.mean(d[e][0] for e in last):.4f}',
                              entry_last5=f'{st.mean(d[e][1] for e in last):.4f}', run=run))
    write(TABLES / 'hyp_events_seed.csv', rows_seed)
    # summary + paired tests (full minus masked, same seed)
    summ = []
    get = lambda ds, pair, view, col: [float(r[col]) for r in sorted(rows_seed, key=lambda r: r['seed'])
                                       if r['dataset'] == ds and r['pair'] == pair and r['view'] == view]
    for ds in datasets:
        for pair in PAIRS:
            if not get(ds, pair, 'mask', 'listed_sel'):
                continue
            for view in ('mask', 'full'):
                row = dict(dataset=ds, pair=pair, view=view, n=len(get(ds, pair, view, 'listed_sel')),
                           best_epoch_mean=f"{st.mean(get(ds, pair, view, 'best_epoch')):.1f}")
                for col in ('listed_first', 'entry_first', 'listed_sel', 'entry_sel', 'listed_last5', 'entry_last5'):
                    row[col] = f'{st.mean(get(ds, pair, view, col)):.4f}'
                summ.append(row)
            row = dict(dataset=ds, pair=pair, view='full-mask', n=len(get(ds, pair, 'mask', 'listed_sel')), best_epoch_mean='')
            for col in ('listed_first', 'entry_first', 'listed_sel', 'entry_sel', 'listed_last5', 'entry_last5'):
                a, b = get(ds, pair, 'full', col), get(ds, pair, 'mask', col)
                dd = [x - y for x, y in zip(a, b)]
                p = float(stats.ttest_rel(a, b).pvalue)
                row[col] = f'{st.mean(dd):+.4f} ({sum(x > 0 for x in dd)}/{len(dd)} up, p={p:.1e})'
            summ.append(row)
    write(TABLES / 'hyp_events_summary.csv', summ)
    # macros
    S = {(r['dataset'], r['pair'], r['view']): r for r in summ}
    pct = lambda x: f'{100 * float(x):.0f}'
    mac = []
    for ds in datasets:
        D = DSNAME[ds]
        for view, V in (('mask', 'Mask'), ('full', 'Full')):
            r = S[(ds, 'exonly', view)]
            mac += [(f'hyEvents{V}Listed{D}', pct(r['listed_sel'])), (f'hyEvents{V}Entry{D}', pct(r['entry_sel'])),
                    (f'hyEvents{V}ListedFirst{D}', pct(r['listed_first'])), (f'hyEvents{V}EntryFirst{D}', pct(r['entry_first'])),
                    (f'hyEvents{V}ListedLast{D}', pct(r['listed_last5'])), (f'hyEvents{V}EntryLast{D}', pct(r['entry_last5']))]
            if (ds, 'anchor', view) in S:
                r = S[(ds, 'anchor', view)]
                mac += [(f'hyEventsAnchor{V}Listed{D}', pct(r['listed_sel'])), (f'hyEventsAnchor{V}Entry{D}', pct(r['entry_sel']))]
    for view, V in (('mask', 'Mask'), ('full', 'Full')):
        for q, col in (('Listed', 'listed_sel'), ('Entry', 'entry_sel')):
            v = [100 * float(S[(ds, 'exonly', view)][col]) for ds in datasets]
            mac += [(f'hyEvents{V}{q}Min', f'{min(v):.0f}'), (f'hyEvents{V}{q}Max', f'{max(v):.0f}')]
    seeds_up = [int(S[(ds, 'exonly', 'full-mask')]['listed_sel'].split('(')[1].split('/')[0]) for ds in datasets]
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from numwords import k_of_n   # prose ratio as 'k of n' / 'all n' (user rule 2026-10-07)
    mac.append(('hyEventsListedSeedsUp', k_of_n(sum(seeds_up), len(SEEDS) * len(datasets))))
    mac.append(('hyEventsDatasets', str(len(datasets))))
    lines = ['% Generated by tables/src/collect_hyp_events.py from tables/xre_events.tsv and tables/xre_runs.tsv'
             + (' and HOST_B runs_abl (Sports)' if sports else '') + '. Do not edit by hand.',
             '% Percent (integers). Mask/Full = exchange term alone on the masked / full view (rows wo_A1 / wo_A1_masked),',
             '% Listed = held-out positives already in the top-K list, Entry = entry share of the coefficient mass,',
             '% at the selected (best-validation) epoch of each run, mean over seeds 1-5 (one run per seed);',
             '% First = first epoch, Last = last five logged epochs, Anchor = XRE versus XRE on the full view.']
    lines += [f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in mac]
    (TABLES / 'macros_hyp_events.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
    print(f'datasets {datasets}; {len(rows_ep)} epoch rows, {len(rows_seed)} runs, {len(mac)} macros', file=sys.stderr)


def write(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


if __name__ == '__main__':
    main()
