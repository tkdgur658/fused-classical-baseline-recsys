"""Apply the CACHEFIX build-correction re-runs of Baby (PREREG_CACHEFIX, revision log 20) to the per-seed sources of tab:main and tab:sota.

Rule (frozen in PREREG_CACHEFIX section 1): "build-correction re-run; the previous values are replaced in the table, not pooled".
Each replaced (row, seed) keeps only the new run; the previous run is kept for the supplementary build table only.

Reads   tables/src/data/cachefix_perseed.csv       (collect_cachefix.py: MSCA DA-MRS / UltraGCN-LI on Baby, seeds 1-5)
        tables/src/data/cachefix_sota_perseed.csv  (collect_cachefix.py: COHESION on Baby, seeds 1-5)
Writes  tables/src/data/cachefix_previous_perseed.csv  the replaced runs (written once, before the first replacement; never overwritten)
        tables/table1_perseed.csv, tables/table1_cells.csv, tables/table2_runs.tsv, tables/table2_cells.csv  through
        collect_cachefix.apply_new(), the same replacement that the simulation of collect_cachefix.py ran on a temporary copy.
Idempotent: a second run finds the rows already replaced (root = cachefix, queue = cachefix) and changes nothing.
Then re-run the builders that read these files (revision log 20 lists them).
Usage: python -s tables/src/apply_cachefix.py [tables_dir]
"""
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import collect_cachefix as cf  # noqa: E402


def main():
    tables = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE.parent
    data = tables / 'src' / 'data'
    new1 = {}
    for r in csv.DictReader(open(data / 'cachefix_perseed.csv', newline='', encoding='utf-8')):
        new1[(r['row'], int(r['seed']))] = r
    new2 = {int(r['seed']): r for r in csv.DictReader(open(data / 'cachefix_sota_perseed.csv', newline='', encoding='utf-8'))}
    assert sorted({k[0] for k in new1}) == ['DA-MRS', 'UltraGCN-LI'] and len(new1) == 10 and sorted(new2) == [1, 2, 3, 4, 5]

    t1, t1cols = cf.rd(tables / 'table1_perseed.csv')
    old1 = [r for r in t1 if r['backbone'] == 'MSCA' and r['dataset'] == 'baby' and (r['row'], int(r['seed'])) in new1]
    t2, t2cols = cf.rd(tables / 'table2_runs.tsv', '\t')
    old2 = [r for r in t2 if r['model'] == 'COHESION' and r['dataset'] == 'baby' and r['chosen'] == '1' and int(r['seed']) in new2]
    done1 = all(r['root'] == 'cachefix' for r in old1)
    done2 = all(r['queue'] == 'cachefix' for r in old2)
    if done1 and done2:
        print('already applied; nothing changed')
        return
    assert not done1 and not done2, 'partially applied state; restore the four files from the backup first'
    prev = data / 'cachefix_previous_perseed.csv'
    if not prev.exists():
        cols = ['table', 'backbone', 'row', 'dataset', 'seed', 'r20', 'n20', 'gpu', 'root', 'fingerprint', 'data']
        with open(prev, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, cols, lineterminator='\n')
            w.writeheader()
            for r in sorted(old1, key=lambda x: (x['row'], int(x['seed']))):
                w.writerow(dict(table='tab:main', backbone='MSCA', row=r['row'], dataset='baby', seed=r['seed'], r20=r['r20'], n20=r['n20'],
                                gpu=r['gpu'], root=r['root'], fingerprint=r['fingerprint'], data='canonical_v2'))
            for r in sorted(old2, key=lambda x: int(x['seed'])):
                w.writerow(dict(table='tab:sota', backbone='COHESION', row='Native', dataset='baby', seed=r['seed'], r20=r['test_r20'],
                                n20=r['test_n20'], gpu=r['gpu'], root=r['queue'],
                                fingerprint=r.get('origin', '').replace('\\', '/').split('/result.json')[0].split('__')[-1], data=r['data']))
        print('wrote', prev, len(old1) + len(old2), 'rows')
    new1d = {k: dict(v) for k, v in new1.items()}
    new2d = {k: dict(v) for k, v in new2.items()}
    cf.apply_new(tables, new1d, new2d)
    print('replaced', len(new1d), 'tab:main seeds and', len(new2d), 'tab:sota seeds in', tables)


if __name__ == '__main__':
    main()
