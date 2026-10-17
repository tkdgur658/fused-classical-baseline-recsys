"""Collect test Recall@K / NDCG@K (K = 5, 10, 20, 50) of MSCA Native and MSCA + XRE (lambda 0.1, K 40) for the Recall@K
figure (figures/src/fig_recallk.py) and the recall* macros (tables/src/build_analysis_macros.py). Reads result.json only,
trains nothing.

Usage (local machine, repository root):  python -s tables/src/collect_recallk.py
The script sends itself over ssh to HOST_A (argument `remote`, plain python3) and writes

  tables/recallk_listana.csv   one row per (dataset, arm, seed, k): recall = best_test['recall@k'], ndcg = best_test['ndcg@k']
      of result.json (test split, full ranking, best-valid checkpoint; the values as stored, 4 dp). Sources:
      Baby, Pantry, Scientific  <RUN_ROOT_B>/P04_listana/runs_listana/{objx_s,xres}/   (P04_listana reruns of the
                                Table I configuration, repo P04_xres_wt 406a7a7; the checkpoints of the self-message and
                                crowding analyses)
      Sports                    <RUN_ROOT_B>/P04_sports/runs_xres/{objx_s,xres}/       (the Table I Native and XRE
                                runs of Sports themselves, same repo commit 406a7a7)
      Native = objx_s <ds>_native, XRE = xres <ds>_xres_xre_l010_k40, seeds 1-5. Exactly one run directory per
      (dataset, arm, seed) is required; the collector stops otherwise.
Before writing, the Baby / Pantry / Scientific rows are compared with the existing tables/recallk_listana.csv (written
before this collector existed) and the script stops on any difference.
"""
import csv
import glob
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] if Path(__file__).exists() else Path('.')   # remote side: stdin
OUT = ROOT / 'tables' / 'recallk_listana.csv'
SSH = ['ssh', '-o', 'BatchMode=yes', 'HOST_A']
ROOTS = {'baby': '<RUN_ROOT_B>/P04_listana/runs_listana', 'pantry': '<RUN_ROOT_B>/P04_listana/runs_listana',
         'scientific': '<RUN_ROOT_B>/P04_listana/runs_listana', 'sports': '<RUN_ROOT_B>/P04_sports/runs_xres'}
DSS = ['baby', 'pantry', 'scientific', 'sports']
OLD_DSS = ['baby', 'pantry', 'scientific']
ARMS = [('Native', 'objx_s', '{ds}_native'), ('XRE', 'xres', '{ds}_xres_xre_l010_k40')]
SEEDS = [1, 2, 3, 4, 5]
KS = [5, 10, 20, 50]
COLS = ['dataset', 'arm', 'seed', 'k', 'recall', 'ndcg']


def remote():
    w = csv.writer(sys.stdout, lineterminator='\n')
    w.writerow(COLS + ['path'])
    for ds in DSS:
        for arm, pkg, cfg in ARMS:
            for s in SEEDS:
                c = cfg.format(ds=ds)
                hits = sorted(glob.glob(f'{ROOTS[ds]}/{pkg}/{pkg}__{c}__{ds}__s{s}__train__*/result.json'))
                if len(hits) != 1:
                    sys.exit(f'{ds} {arm} s{s}: {len(hits)} run directories {hits}')
                r = json.load(open(hits[0]))
                if r.get('status') != 'SUCCESS':
                    sys.exit(f'{hits[0]}: status {r.get("status")}')
                bt = r['best_test']
                for k in KS:
                    w.writerow([ds, arm, s, k, bt[f'recall@{k}'], bt[f'ndcg@{k}'], hits[0]])


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'remote':
        return remote()
    src = open(__file__, encoding='utf-8').read()
    out = subprocess.run(SSH + ['python3', '-', 'remote'], input=src.encode('utf-8'), capture_output=True, check=True)
    if out.stderr:
        sys.stderr.write(out.stderr.decode('utf-8', 'replace'))
    rows = list(csv.DictReader(out.stdout.decode('utf-8').splitlines()))
    if len(rows) != len(DSS) * len(ARMS) * len(SEEDS) * len(KS):
        sys.exit(f'{len(rows)} rows')
    key = lambda r: (r['dataset'], r['arm'], r['seed'], r['k'])
    if OUT.exists():
        with open(OUT, newline='', encoding='utf-8') as f:
            old = {key(r): r for r in csv.DictReader(f)}
        new = {key(r): r for r in rows}
        for k, r in old.items():
            if k[0] in OLD_DSS and (k not in new or (new[k]['recall'], new[k]['ndcg']) != (r['recall'], r['ndcg'])):
                sys.exit(f'existing row {k} {r} differs from the source {new.get(k)}')
    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=COLS, lineterminator='\n', extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    print('wrote', OUT.relative_to(ROOT), len(rows), 'rows')


if __name__ == '__main__':
    main()
