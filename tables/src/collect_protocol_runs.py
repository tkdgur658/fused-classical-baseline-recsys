"""Collect the per-run results behind the appendix training-protocol table (tables/protocol_runs.tsv).

Runs on HOST_A and only reads files. Usage (from the repository root, on the local machine):
    ssh HOST_A python3 - < tables/src/collect_protocol_runs.py > tables/protocol_runs.tsv

Rows: the MSCA and FREEDOM cells of Table 1 (native backbone and six plug-ins; Baby, Pantry, Scientific; seeds 1-3).
The (package, config) pair of each cell is read from the Table 1 cell file (source column), so all three protocols
use exactly the configurations of Table 1 and differ only in the training protocol:
  common  : Table 1 runs (batch 256, AdamW, cosine schedule, 100 epochs, patience 10, AMP)
            <- <RUN_ROOT_B>/P04_table1/table1_runs.tsv
  b2048   : batch 2048, 1000 epochs, patience 20, rest as common (configs <cfg>_b2048L)
            <- <RUN_ROOT_B>/P04_main_runs/qb2/*/*/*/result.json
  plugin  : MMRec defaults used by the plug-in papers (batch 2048, Adam, constant lr, no AMP, 1000 epochs,
            patience 20; configs <cfg>_pb) <- <RUN_ROOT_B>/P04_main_runs/qb3/*/*/*/result.json
Test metrics are the best_primed test metrics at the epoch selected by validation Recall@20 (best_test in result.json).
Duplicates or missing runs are reported on stderr.
"""
import csv
import glob
import json
import sys

CELLS = '<RUN_ROOT_B>/P04_table1/table1_cells_merged4.csv'
COMMON = '<RUN_ROOT_B>/P04_table1/table1_runs.tsv'
PROTOCOLS = [('b2048', '<RUN_ROOT_B>/P04_main_runs/qb2', '_b2048L'),
             ('plugin', '<RUN_ROOT_B>/P04_main_runs/qb3', '_pb')]
BACKBONES = ('MSCA', 'FREEDOM')
SEEDS = (1, 2, 3)

src = {}  # (package, config, dataset) -> (backbone, row)
for r in csv.DictReader(open(CELLS)):
    if r['backbone'] in BACKBONES:
        pkg, cfg = r['source'].split('/')
        src[(pkg, cfg, r['dataset'])] = (r['backbone'], r['row'])

out = {}  # (protocol, backbone, row, dataset, seed) -> record


def add(proto, pkg, cfg, ds, seed, rec, origin):
    if (pkg, cfg, ds) not in src:
        return
    bb, row = src[(pkg, cfg, ds)]
    key = (proto, bb, row, ds, seed)
    if key in out:
        print(f'DUPLICATE {key}: {out[key]["origin"]} | {origin}', file=sys.stderr)
        return
    out[key] = dict(rec, package=pkg, config=cfg, origin=origin)


for r in csv.DictReader(open(COMMON), delimiter='\t'):
    add('common', r['package'], r['config'], r['dataset'], int(r['seed']),
        {'test_r20': float(r['test_r20']), 'test_n20': float(r['test_n20']), 'valid_r20': float(r['valid_r20']),
         'best_epoch': int(r['best_epoch'])}, r['run_id'])

for proto, root, suffix in PROTOCOLS:
    for f in sorted(glob.glob(f'{root}/*/*/*/result.json')):
        d = json.load(open(f))
        if d.get('status') != 'SUCCESS':
            print(f'NOT_SUCCESS {f}', file=sys.stderr)
            continue
        cfg = d['config']
        if not cfg.endswith(suffix):
            print(f'UNEXPECTED_CONFIG {f}', file=sys.stderr)
            continue
        add(proto, d['package'], cfg[:-len(suffix)], d['dataset'].split('@')[0], int(d['seed']),
            {'test_r20': d['best_test']['recall@20'], 'test_n20': d['best_test']['ndcg@20'],
             'valid_r20': d['best_valid_score'], 'best_epoch': d.get('best_epoch')}, f.split(root + '/')[1])

for proto in ['common'] + [p for p, _, _ in PROTOCOLS]:
    for (pkg, cfg, ds), (bb, row) in sorted(src.items()):
        for s in SEEDS:
            if (proto, bb, row, ds, s) not in out:
                print(f'MISSING {proto} {bb} {row} {ds} seed {s} ({pkg}/{cfg})', file=sys.stderr)

cols = ['protocol', 'backbone', 'row', 'dataset', 'seed', 'package', 'config', 'test_r20', 'test_n20', 'valid_r20',
        'best_epoch', 'origin']
w = csv.writer(sys.stdout, delimiter='\t', lineterminator='\n')
w.writerow(cols)
for key in sorted(out):
    rec = out[key]
    w.writerow(list(key) + [rec[c] for c in cols[5:]])
