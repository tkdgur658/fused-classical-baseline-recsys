"""Collect the top-K loss rows (SL@K, PSL, SSM) of the COHESION and SMORE blocks of Table I. Reads remote files only.

Packages objx_c (COHESION) and objx_sm (SMORE), branch cohsmo-topk c18533e (cosine rows added to dff26c1) (worktree HOST_A
<HOME>/P04_cohsmo_topk_wt): the top-K loss replaces the BPR term of the backbone, score function cosine or inner product (both searched),
temperature grids of the MSCA rows. Per (backbone, dataset, loss) cell the temperature (score, tau)* is the one recorded in
topk_selection.json by select_topk_cohsmo.py (seed-1 validation Recall@20, ties validation NDCG@20, then the smaller tau);
seed 1 is the grid run at tau*, seeds 2-5 are the runs of the tau* configuration.
Sources (one machine per backbone block on Sports, as collect_cohsmo.py):
  HOST_A  <RUN_ROOT_B>/P04_cohsmo_topk (runs_ctopk, topk_selection.json): COHESION B/P/S/Sports, SMORE B/P/S
  HOST_B  <RUN_ROOT_C>/P04_cohsmo_sports (runs_topk, topk_selection.json): SMORE Sports only
Eligible run: status SUCCESS, mode train, test_eval best_primed, data canonical_v2. One run per
seed is expected; if a seed was run twice, the earliest started run is used (value-independent, never the better one).
Output: tables/cohsmo_topk_perseed.csv (backbone, row, dataset, seed, r20, n20, tau, source), read by build_table1_main.py
and build_table_cohsmo.py. Only cells with a recorded tau* are written; a cell with fewer than five seeds stays a dash.
Usage (repository root): python -s tables/src/collect_cohsmo_topk.py
"""
import csv
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'tables' / 'cohsmo_topk_perseed.csv'
BB = {'objx_c': 'COHESION', 'objx_sm': 'SMORE'}
ROW = {'slk': 'SL@K', 'psl': 'PSL', 'ssm': 'SSM'}
HOSTS = [('HOST_A', '<RUN_ROOT_B>/P04_cohsmo_topk', 'any', 'HOST_A'),
         ('HOST_B', '<RUN_ROOT_C>/P04_cohsmo_sports', 'any', 'HOST_B')]

REMOTE = r'''
import glob, json, os, sys
D, TAG = sys.argv[1], sys.argv[2]
p = os.path.join(D, 'topk_selection.json')
sel = json.load(open(p)) if os.path.exists(p) else {}
print(json.dumps({'sel': sel}))
for f in glob.glob(D + '/runs*/objx_*/*/result.json'):
    r = json.load(open(f))
    b = r.get('best_test') or {}
    if (r.get('status') == 'SUCCESS' and r.get('mode') == 'train' and r.get('test_eval') == 'best_primed'
            and (TAG == 'any' or TAG in (r.get('gpu') or '')) and os.path.basename(str(r.get('data_root', '')).rstrip('/')) == 'canonical_v2'
            and b.get('recall@20') is not None):
        print(json.dumps({'pkg': r['package'], 'cfg': r['config'], 'ds': r['dataset'], 'seed': int(r['seed']),
                          'started': r.get('started') or '', 'r20': b['recall@20'], 'n20': b.get('ndcg@20'), 'f': f}))
'''


def main():
    rows = []
    for host, d, tag, src in HOSTS:
        out = subprocess.run(['ssh', '-o', 'BatchMode=yes', host, f'python3 - {d} {tag}'], input=REMOTE,
                             capture_output=True, text=True, check=True).stdout.splitlines()
        sel = json.loads(out[0])['sel']
        runs = {}
        for line in out[1:]:
            j = json.loads(line)
            runs.setdefault((j['pkg'], j['cfg'], j['seed']), []).append(j)
        for key, s in sel.items():
            pkg, ds, loss = key.split('/')
            if src == 'HOST_B' and not (pkg == 'objx_sm' and ds == 'sports'):
                continue
            if src == 'HOST_A' and pkg == 'objx_sm' and ds == 'sports':
                continue
            for seed in range(1, 6):
                got = sorted(runs.get((pkg, s['config'], seed), []), key=lambda j: j['started'])
                if not got:
                    continue
                j = got[0]
                rows.append(dict(backbone=BB[pkg], row=ROW[loss], dataset=ds, seed=seed, r20=j['r20'], n20=j['n20'],
                                 tau=s['tau'], source=src))
    keys = [(r['backbone'], r['row'], r['dataset'], r['seed']) for r in rows]
    assert len(keys) == len(set(keys)), 'duplicate cells'
    rows.sort(key=lambda r: (r['backbone'], r['row'], r['dataset'], r['seed']))
    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['backbone', 'row', 'dataset', 'seed', 'r20', 'n20', 'tau', 'source'],
                           lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    full = {}
    for r in rows:
        full[(r['backbone'], r['row'], r['dataset'])] = full.get((r['backbone'], r['row'], r['dataset']), 0) + 1
    print('wrote', OUT, len(rows), 'rows | complete cells:', sum(1 for v in full.values() if v == 5), 'of 24')


if __name__ == '__main__':
    main()
