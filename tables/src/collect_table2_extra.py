"""Collect the Table II baselines of the datasets added on HOST_A (user decision 2026-10-02) into
tables/table2_cells_extra.csv and tables/table2_runs_extra.tsv (same columns as collect_table2.py), read next to
table2_cells.csv / table2_runs.tsv by build_table2_main.py. Numbers are never typed by hand.

Sources: work dirs WORK[ds] on HOST_A (one per added dataset). Their job files jobs_t2*.txt list the configuration of every
(model, dataset) (TMM author-derived rule, commit messages of P04_t2b2048L_wt c038478 and earlier; README.txt of each work
dir); runs live in <work dir>/runs_t2/<package>/. Failed attempts are kept outside runs_t2 (failed_runs_t2/) and are not read.
Rules (fixed before this collection, the same as collect_table2.py)
  1. Eligible run: status SUCCESS, mode train, test_eval best_primed, seed 1-5, data copy canonical_v2,
     configuration = the one listed in the job files of that dataset.
  2. One run per (model, dataset, seed): the earliest completion (started + elapsed_sec). Value-independent.
  3. Cell = mean over the chosen seeds, SD = sample SD over seeds (ddof 1).
Usage (local machine, repository root): python tables/src/collect_table2_extra.py
"""
import csv
import datetime as dt
import glob
import json
import os
import statistics as st
import subprocess
import sys

WORK = {'clothing': '<RUN_ROOT_B>/P04_t2_clothing/', 'instruments': '<RUN_ROOT_B>/P04_t2_instruments/',
        'sports': '<RUN_ROOT_B>/P04_sports/'}
MODEL_OF = {'cohesion': 'COHESION', 'smore': 'SMORE', 'pgl': 'PGL', 'mentor': 'MENTOR', 'diffmm': 'DiffMM',
            'lgmrec': 'LGMRec', 'freedom': 'FREEDOM'}
SEEDS = [1, 2, 3, 4, 5]
FIELDS = ['package', 'config', 'dataset', 'seed', 'status', 'mode', 'test_eval', 'data', 'gpu', 'started', 'elapsed_sec',
          'best_epoch', 'test_r10', 'test_n10', 'test_r20', 'test_n20', 'path']


# ----------------------------------------------------------------------------------------------- remote side
def scan():
    w = csv.writer(sys.stdout, delimiter='\t', lineterminator='\n')
    w.writerow(['kind'] + FIELDS)
    for ds, base in WORK.items():
        for jf in sorted(glob.glob(base + 'jobs_t2*.txt')):
            for line in open(jf):
                t = line.split()
                if len(t) >= 5 and not t[0].startswith('#') and t[0] in MODEL_OF:
                    w.writerow(['job', t[0], t[1], ds] + [''] * (len(FIELDS) - 3))
        for f in glob.glob(base + 'runs_t2/*/*/result.json'):
            try:
                d = json.load(open(f, encoding='utf-8'))
            except Exception as e:
                print(f'UNREADABLE {f} {type(e).__name__}', file=sys.stderr)
                continue
            bt = d.get('best_test') or {}
            w.writerow(['run', d.get('package'), d.get('config'), str(d.get('dataset', '')).split('@')[0], d.get('seed'),
                        d.get('status'), d.get('mode'), d.get('test_eval'),
                        os.path.basename(str(d.get('data_root', '')).rstrip('/')),
                        (d.get('gpu') or '').replace('NVIDIA ', ''), d.get('started'), d.get('elapsed_sec'),
                        d.get('best_epoch'), bt.get('recall@10'), bt.get('ndcg@10'), bt.get('recall@20'), bt.get('ndcg@20'), f])


# ------------------------------------------------------------------------------------------------ local side
HERE = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.dirname(HERE)


def remote():
    with open(os.path.abspath(__file__), 'rb') as fh:
        out = subprocess.run(['ssh', '-o', 'BatchMode=yes', 'HOST_A', 'python3', '-', 'scan'], stdin=fh,
                             capture_output=True, check=True)
    if out.stderr.strip():
        print(out.stderr.decode(errors='replace'), file=sys.stderr)
    return list(csv.DictReader(out.stdout.decode().splitlines(), delimiter='\t'))


def completed(r):
    try:
        return dt.datetime.strptime(r['started'], '%Y-%m-%d %H:%M:%S') + dt.timedelta(seconds=float(r['elapsed_sec'] or 0))
    except (TypeError, ValueError):
        return dt.datetime.max


def ineligible(r, want):
    why = []
    if r['config'] != want:
        why.append(f'config {r["config"]} (job files: {want})')
    if r['status'] != 'SUCCESS':
        why.append(f"status {r['status']}")
    if r['mode'] != 'train':
        why.append(f"mode {r['mode']}")
    if r['test_eval'] != 'best_primed':
        why.append(f"test_eval {r['test_eval'] or 'none'}")
    if r['test_r20'] in ('', 'None') or r['test_n20'] in ('', 'None'):
        why.append('no test metrics')
    if r['seed'] not in [str(s) for s in SEEDS]:
        why.append('seed outside 1-5')
    # (device-name filter of the original collector removed in the public copy; other rules unchanged)
    if r['data'] != 'canonical_v2':
        why.append(f"data {r['data']}")
    return '; '.join(why)


def f6(x):
    return '' if x is None else f'{x:.6f}'


def main():
    rows = remote()
    want = {}
    for r in rows:
        if r['kind'] == 'job':
            want.setdefault((MODEL_OF[r['package']], r['dataset']), r['config'])
    runs = [r for r in rows if r['kind'] == 'run' and r['package'] in MODEL_OF]
    groups = {}
    for r in runs:
        r['model'] = MODEL_OF[r['package']]
        r['ineligible_reason'] = ineligible(r, want.get((r['model'], r['dataset']), '?'))
        r['chosen'] = '0'
        if not r['ineligible_reason']:
            groups.setdefault((r['model'], r['dataset'], int(r['seed'])), []).append(r)
    for g in groups.values():
        sorted(g, key=lambda x: (completed(x), x['path']))[0]['chosen'] = '1'
    cols = ['model', 'dataset', 'seed', 'chosen', 'ineligible_reason', 'rank', 'duplicate_note', 'host', 'queue', 'config',
            'status', 'mode', 'test_eval', 'data', 'superseded', 'completed', 'best_epoch', 'test_r10', 'test_n10',
            'test_r20', 'test_n20', 'gpu', 'origin']
    with open(os.path.join(TABLES, 'table2_runs_extra.tsv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f, delimiter='\t', lineterminator='\n')
        w.writerow(cols)
        for r in sorted(runs, key=lambda x: (x['model'], x['dataset'], x['seed'], x['path'])):
            w.writerow([r['model'], r['dataset'], r['seed'], r['chosen'], r['ineligible_reason'], '', '', 'HOST_A',
                        r['path'].split('/runs_t2/')[0].split('/')[-1], r['config'], r['status'], r['mode'],
                        r['test_eval'], r['data'], '', completed(r).strftime('%Y-%m-%d %H:%M:%S'), r['best_epoch'],
                        r['test_r10'], r['test_n10'], r['test_r20'], r['test_n20'], r['gpu'], r['path']])
    with open(os.path.join(TABLES, 'table2_cells_extra.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['model', 'dataset', 'config', 'n', 'seeds', 'missing_seeds', 'r20_mean', 'n20_mean', 'r20_std',
                    'n20_std', 'r10_mean', 'n10_mean', 'queues', 'duplicates'])
        for (m, d), cfg in sorted(want.items()):
            ch = sorted((r for r in runs if r['model'] == m and r['dataset'] == d and r['chosen'] == '1'),
                        key=lambda x: int(x['seed']))
            got = [int(r['seed']) for r in ch]
            vals = {k: [float(r[f'test_{k}']) for r in ch] for k in ('r20', 'n20', 'r10', 'n10')}

            def mean(v):
                return st.mean(v) if v else None

            def sd(v):
                return st.stdev(v) if len(v) > 1 else None
            w.writerow([m, d, cfg, len(ch), '/'.join(map(str, got)), '/'.join(str(s) for s in SEEDS if s not in got),
                        f6(mean(vals['r20'])), f6(mean(vals['n20'])), f6(sd(vals['r20'])), f6(sd(vals['n20'])),
                        f6(mean(vals['r10'])), f6(mean(vals['n10'])), 'HOST_A', ''])
            print(f'{m} {d} {cfg}: n={len(ch)} seeds {got}', file=sys.stderr)
    print(f'{len(runs)} runs, {sum(r["chosen"] == "1" for r in runs)} chosen', file=sys.stderr)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'scan':
        scan()
    else:
        main()
