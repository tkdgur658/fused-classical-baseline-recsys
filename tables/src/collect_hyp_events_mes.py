"""MES version of the exchange-event analysis of tables/src/collect_hyp_events.py (PREREG_MESCKPT unit U7, 2026-10-16).

Question (same as the original): on the masked view the held-out positives of a fold are not lifted into the top-K list by a
self-message, so entry events stay alive; on the full view they are listed early and entry events fade (Lemma lem:masked).
Runs, one per seed, seeds 1-5, tables/mes_perseed.csv (backbone MSCA), both lambda 0.1 and K 40 so the pair differs only in the view:
  mask = row MES_k40   (xres0 <ds>_xres_xeoa0_l010_k40: MES, entry pairs on the masked view)
  full = row MES_fv    (xres0f <ds>_xres_xeoa0f_l010_k40: MES with the entry pairs on the full view, mass-matched)
Monitor lines '[R12] step=... p_in_list=... mass_entry=... mass_retention=...' of stdout.log (every 500 steps; held-out rows).
  listed  = p_in_list = share of held-out rows whose positive is already inside the top-K list (no entry event)
  mass    = mass_entry = summed detached sigmoid(gap) coefficients of the entry pairs (full view: before the mass rescaling)
MES has no retention term, so mass_retention is 0 in every line (asserted) and the entry SHARE of the original analysis is 1 by
construction in both views; it is therefore not reported. Epoch of a line = step // ceil(history_keys / 256) + 1 (1-based).
Workflow (no ssh from here):
  local   python tables/src/collect_hyp_events_mes.py --paths <runs.txt>
  server  python collect_hyp_events_mes.py remote $(cat runs.txt) > out/events_raw.tsv
  local   python tables/src/collect_hyp_events_mes.py --raw events_raw.tsv
Writes tables/src/data/hyp_events_{epochs,seed,summary}_mes.csv and tables/macros_hyp_events_mes.tex (names hyEventsMes*, new).
A dataset enters only when both views have all five seeds.
"""
import csv
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

TABLES = Path(__file__).resolve().parents[1]
DS = [('baby', 'Baby'), ('pantry', 'Pantry'), ('scientific', 'Scientific'), ('sports', 'Sports')]
SEEDS = [1, 2, 3, 4, 5]
VIEW = {'MES_k40': 'mask', 'MES_fv': 'full'}
MIN_RUNS = 3
REMOTE = r'''
import glob, json, os, re, sys
head = re.compile(r'history_keys=(\d+)'); step = re.compile(r'\[R12\] step=(\d+) (.*)')
for d in sys.argv[1:]:
    try: r = json.load(open(os.path.join(d, 'result.json')))
    except Exception: print('ERR', d, 'no result.json', sep='\t'); continue
    log = os.path.join(d, 'stdout.log'); n = ''
    if not os.path.isfile(log): print('ERR', d, 'no stdout.log', sep='\t'); continue
    print('RUN', d, r.get('status'), r.get('best_epoch'), sep='\t')
    for chunk in open(log, encoding='utf-8', errors='replace'):
        for line in chunk.split('\r'):
            m = head.search(line)
            if m and not n: n = m.group(1)
            m = step.search(line)
            if m:
                kv = dict(x.split('=', 1) for x in m.group(2).split() if '=' in x)
                print('EV', d, n, m.group(1), kv.get('p_in_list', ''), kv.get('mass_entry', ''), kv.get('mass_retention', ''), sep='\t')
'''


def runs():
    out = {}
    for r in csv.DictReader(open(TABLES / 'mes_perseed.csv', newline='', encoding='utf-8')):
        if r['backbone'] == 'MSCA' and r['row'] in VIEW:
            k = (r['dataset'], VIEW[r['row']], int(r['seed']))
            if k in out:
                raise SystemExit(f'two runs for {k}')
            out[k] = r['run_dir'].rstrip('/')
    return out


def write(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'remote':
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        exec(REMOTE)
        return
    R = runs()
    if '--paths' in sys.argv:
        Path(sys.argv[sys.argv.index('--paths') + 1]).write_text('\n'.join(sorted(R.values())) + '\n', encoding='utf-8')
        print('wrote', len(R), 'run dirs')
        return
    from scipy import stats
    key_of = {p: k for k, p in R.items()}
    meta, ev = {}, defaultdict(lambda: defaultdict(list))
    for line in open(sys.argv[sys.argv.index('--raw') + 1], encoding='utf-8').read().splitlines():
        p = line.split('\t')
        if p[0] == 'ERR':
            print('missing log:', p[1:], file=sys.stderr)
        elif p[0] == 'RUN':
            meta[key_of[p[1]]] = (p[2], int(p[3]), p[1])
        elif p[0] == 'EV':
            k = key_of[p[1]]
            assert float(p[6]) == 0.0, ('retention mass not zero', p)
            e = int(p[3]) // math.ceil(int(p[2]) / 256) + 1
            ev[k][e].append((float(p[4]), float(p[5])))
    datasets = [ds for ds, _ in DS if all(k in ev and meta[k][0] == 'SUCCESS' for v in ('mask', 'full') for s in SEEDS for k in [(ds, v, s)])]
    skipped = [ds for ds, _ in DS if ds not in datasets]
    pr = {k: {e: (st.mean(x[0] for x in v), st.mean(x[1] for x in v)) for e, v in d.items()} for k, d in ev.items() if k[0] in datasets}
    rows_ep = []
    for ds in datasets:
        for view in ('mask', 'full'):
            runs_ = {s: pr[(ds, view, s)] for s in SEEDS}
            for e in sorted({e for d in runs_.values() for e in d}):
                vals = [d[e] for d in runs_.values() if e in d]
                if len(vals) < MIN_RUNS:
                    continue
                rows_ep.append(dict(dataset=ds, view=view, epoch=e, n_runs=len(vals),
                                    listed_mean=f'{st.mean(v[0] for v in vals):.4f}', listed_min=f'{min(v[0] for v in vals):.4f}',
                                    listed_max=f'{max(v[0] for v in vals):.4f}', mass_mean=f'{st.mean(v[1] for v in vals):.3f}',
                                    mass_min=f'{min(v[1] for v in vals):.3f}', mass_max=f'{max(v[1] for v in vals):.3f}'))
    write(TABLES / 'src/data/hyp_events_epochs_mes.csv', rows_ep)
    rows_seed = []
    for (ds, view, s), d in sorted(pr.items()):
        eps = sorted(d)
        be = meta[(ds, view, s)][1]
        sel = d.get(be) or d[max(e for e in eps if e <= be)]
        last = eps[-5:]
        rows_seed.append(dict(dataset=ds, view=view, seed=s, best_epoch=be, last_epoch=eps[-1], listed_first=f'{d[eps[0]][0]:.4f}',
                              mass_first=f'{d[eps[0]][1]:.3f}', listed_sel=f'{sel[0]:.4f}', mass_sel=f'{sel[1]:.3f}',
                              listed_last5=f'{st.mean(d[e][0] for e in last):.4f}', mass_last5=f'{st.mean(d[e][1] for e in last):.3f}',
                              run=meta[(ds, view, s)][2]))
    write(TABLES / 'src/data/hyp_events_seed_mes.csv', rows_seed)
    get = lambda ds, view, col: [float(r[col]) for r in sorted(rows_seed, key=lambda r: r['seed']) if r['dataset'] == ds and r['view'] == view]
    cols = ('listed_first', 'mass_first', 'listed_sel', 'mass_sel', 'listed_last5', 'mass_last5')
    summ = []
    for ds in datasets:
        for view in ('mask', 'full'):
            row = dict(dataset=ds, view=view, n=5, best_epoch_mean=f"{st.mean(get(ds, view, 'best_epoch')):.1f}")
            row.update({c: f'{st.mean(get(ds, view, c)):.4f}' for c in cols})
            summ.append(row)
        row = dict(dataset=ds, view='full-mask', n=5, best_epoch_mean='')
        for c in cols:
            a, b = get(ds, 'full', c), get(ds, 'mask', c)
            dd = [x - y for x, y in zip(a, b)]
            row[c] = f'{st.mean(dd):+.4f} ({sum(x > 0 for x in dd)}/{len(dd)} up, p={float(stats.ttest_rel(a, b).pvalue):.1e})'
        summ.append(row)
    write(TABLES / 'src/data/hyp_events_summary_mes.csv', summ)
    S = {(r['dataset'], r['view']): r for r in summ}
    pct = lambda x: f'{100 * float(x):.0f}'
    L = ['% Generated by tables/src/collect_hyp_events_mes.py (MES_k40 masked view vs MES_fv full view, lambda 0.1, K 40, seeds 1-5).',
         '% Listed = held-out positives already in the top-K list (percent) at the selected epoch / first epoch / mean of the last five;',
         '% Mass = summed entry coefficient mass per logged step; Up = seeds with full > masked.']
    for ds in datasets:
        D = dict(DS)[ds]
        for view, V in (('mask', 'Mask'), ('full', 'Full')):
            r = S[(ds, view)]
            L += [f'\\newcommand{{\\hyEventsMes{V}Listed{D}}}{{{pct(r["listed_sel"])}}}', f'\\newcommand{{\\hyEventsMes{V}ListedFirst{D}}}{{{pct(r["listed_first"])}}}',
                  f'\\newcommand{{\\hyEventsMes{V}ListedLast{D}}}{{{pct(r["listed_last5"])}}}', f'\\newcommand{{\\hyEventsMes{V}Mass{D}}}{{{float(r["mass_sel"]):.1f}}}']
        L.append(f'\\newcommand{{\\hyEventsMesListedUp{D}}}{{{S[(ds, "full-mask")]["listed_sel"].split("(")[1].split("/")[0]}}}')
    for view, V in (('mask', 'Mask'), ('full', 'Full')):
        v = [100 * float(S[(ds, view)]['listed_sel']) for ds in datasets]
        L += [f'\\newcommand{{\\hyEventsMes{V}ListedMin}}{{{min(v):.0f}}}', f'\\newcommand{{\\hyEventsMes{V}ListedMax}}{{{max(v):.0f}}}']
    L.append(f'\\newcommand{{\\hyEventsMesDatasets}}{{{len(datasets)}}}')
    (TABLES / 'macros_hyp_events_mes.tex').write_text('\n'.join(L) + '\n', encoding='utf-8', newline='\n')
    print('datasets', datasets, 'skipped', skipped, len(rows_ep), 'epoch rows')


if __name__ == '__main__':
    main()
