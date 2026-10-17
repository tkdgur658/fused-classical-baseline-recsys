"""Collect the masked-view loss comparison (reviewer point: BPR, hard-negative BPR and SL@K under the masking and anchor
conditions of XRE). Reads remote result.json files, trains nothing.

Runs: HOST_A <RUN_ROOT_B>/P04_xmask/runs_xmask/xmask/*/result.json (repo <HOME>/P04_xmask_wt,
branch xmask 7951ebc, package xmask = XRES with the exchange term L_X replaced by L_alt; jobs P04_xmask/jobs_xmask.txt).
Rows (MSCA, lambda 0.1, anchor term A1, alternating-fold masked view, held-out positives of XRE):
  mbpr  L_alt = BPR of a held-out positive against one uniform non-positive item, both scored on the masked view
  mdns  L_alt = BPR against the highest-scoring of 50 uniform non-positive candidates on the masked view (DNS)
  mslk  L_alt = SL@20 on the masked view, with the dot score and temperature of the SL@K row of Table I
Unit tests (results/xmask/unit in the repo): L_alt weight 0 is bitwise the 'w/o exchange term' row; L_alt uses the same
masked-view tensors as XRE.
Sports (added 2026-10-07): HOST_B runs of <RUN_ROOT_C>/P04_sports_abl (branch xmask-sports f2947de =
xmask 7951ebc + Sports configs sports_xmask_{mbpr,mdns,mslk}, SL@K with the dot score and tau 0.2 of the Table I MSCA Sports
SL@K selection), synced to HOST_A <RUN_ROOT_B>/P04_sports_abl_HOST_B/runs_abl/xmask (README.txt of that work dir).
The device of every run is written out.
Output: tables/xmask_perseed.csv (row, dataset, seed, r20, n20, best_epoch, config, run_dir, gpu), test metrics of the
best-validation checkpoint (best_test of result.json). One run per seed (asserted).
Usage (repository root): python -s tables/src/collect_xmask.py
"""
import csv
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'tables' / 'xmask_perseed.csv'
REMOTE = '<RUN_ROOT_B>/P04_xmask/runs_xmask <RUN_ROOT_B>/P04_sports_abl_HOST_B/runs_abl/xmask'
CMD = (f"for f in $(find {REMOTE} -name result.json); do "
       "python3 -c \"import json,sys,os; j=json.load(open(sys.argv[1])); b=j.get('best_test',{}); "
       "print(json.dumps(dict(cfg=j['config'],ds=j['dataset'],seed=j['seed'],status=j.get('status'),"
       "r20=b.get('recall@20'),n20=b.get('ndcg@20'),ep=j.get('best_epoch'),run=os.path.dirname(sys.argv[1]),"
       "gpu=j.get('gpu'),te=j.get('test_eval'),mode=j.get('mode'))))\" $f; done")


def main():
    out = subprocess.run(['ssh', '-o', 'BatchMode=yes', 'HOST_A', CMD], capture_output=True, text=True, check=True).stdout
    rows = []
    for line in out.splitlines():
        j = json.loads(line)
        assert j['status'] in ('SUCCESS', None), j
        if j['cfg'].endswith('_xmask_mdnsfv'):   # E1 runs live elsewhere (collect_e1.py); never mixed in here
            continue
        assert j['mode'] == 'train' and j['te'] == 'best_primed', j
        gpu = (j['gpu'] or '').replace('NVIDIA ', '')
        rows.append(dict(row=j['cfg'].split('_xmask_')[1], dataset=j['ds'].split('@')[0], seed=int(j['seed']), r20=j['r20'],
                         n20=j['n20'], best_epoch=j['ep'], config=j['cfg'], run_dir=j['run'], gpu=gpu))
    rows.sort(key=lambda r: (r['row'], r['dataset'], r['seed']))
    keys = {(r['row'], r['dataset'], r['seed']) for r in rows}
    assert len(keys) == len(rows) == 60, (len(keys), len(rows))   # 3 rows x 4 datasets x 5 seeds
    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    print('wrote', OUT, len(rows))


if __name__ == '__main__':
    main()
