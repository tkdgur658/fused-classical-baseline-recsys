"""Collect the Table-2 baseline runs under the two batch-2048 protocols (appendix protocol table, lower block).

Runs on the HOST_D pod and only reads files. Usage (from the repository root, on the local machine):
    ssh -i <SSH_KEY> -p <PORT> user@HOST_D python3 - < tables/src/collect_protocol_sota_runs.py \
        > tables/protocol_sota_runs.tsv

Source: <POD_ROOT>/p04e/q2b/runs/<package>/<run>/result.json (data v2, author configurations of Table 2).
  b2048  : configs *_b2048L (batch 2048, 1000 epochs, patience 20, rest as the common protocol)
  plugin : configs *_pb (MMRec defaults: batch 2048, Adam, constant lr, no AMP, 1000 epochs, patience 20)
Test metrics are the best_primed test metrics at the epoch selected by validation Recall@20.
"""
import csv
import glob
import json
import sys

ROOT = '<POD_ROOT>/p04e/q2b/runs'
NAMES = {'cohesion': 'COHESION', 'diffmm': 'DiffMM', 'freedom': 'FREEDOM', 'lgmrec': 'LGMRec', 'mentor': 'MENTOR',
         'msca': 'MSCA', 'pgl': 'PGL', 'smore': 'SMORE'}
out = {}
for f in sorted(glob.glob(f'{ROOT}/*/*/result.json')):
    d = json.load(open(f))
    if d.get('status') != 'SUCCESS':
        print(f'NOT_SUCCESS {f}', file=sys.stderr)
        continue
    cfg = d['config']
    proto = 'b2048' if cfg.endswith('_b2048L') else ('plugin' if cfg.endswith('_pb') else None)
    if proto is None:
        print(f'UNEXPECTED_CONFIG {f}', file=sys.stderr)
        continue
    key = (proto, NAMES[d['package']], d['dataset'].split('@')[0], int(d['seed']))
    if key in out:
        print(f'DUPLICATE {key}', file=sys.stderr)
        continue
    out[key] = [d['package'], cfg, d['best_test']['recall@20'], d['best_test']['ndcg@20'], d['best_valid_score'],
                d.get('best_epoch'), f.split(ROOT + '/')[1]]
w = csv.writer(sys.stdout, delimiter='\t', lineterminator='\n')
w.writerow(['protocol', 'model', 'dataset', 'seed', 'package', 'config', 'test_r20', 'test_n20', 'valid_r20',
            'best_epoch', 'origin'])
for key in sorted(out):
    w.writerow(list(key) + out[key])
