"""Collect the official-code reproduction on FREEDOM-Baby (appendix table) into two files.

Runs on HOST_A and only reads files. First refresh the parsed logs with the read-only collector of the reproduction
workspace, then run this script (from the repository root, on the local machine):
    ssh HOST_A 'cd <RUN_ROOT_B>/P04_official_repro && \
        python runner/collect.py > /dev/null'
    ssh HOST_A python3 - combos < tables/src/collect_official_repro.py > tables/official_repro_combos.tsv
    ssh HOST_A python3 - reported < tables/src/collect_official_repro.py > tables/official_repro_reported.csv

combos   : one row per finished grid combination of the released code, parsed by runner/collect.py from the logs of
           quick_start (valid = best validation result, test = test result at that epoch), from
           <RUN_ROOT_B>/P04_official_repro/results_all_combos.json.
reported : the FREEDOM-Baby values printed in each plug-in paper (arXiv text in P04_official_repro/papers/). Each
           value below is checked to occur in the extracted paper text, and the row is written only if all its
           values are found. DA-MRS reports no FREEDOM result, so its LightGCN-Baby row is used.
"""
import csv
import json
import sys

ROOT = '<RUN_ROOT_B>/P04_official_repro'
# group in results_all_combos.json -> (plug-in, arm, backbone, seed). Groups not listed (serial identity check
# MG_native_asis) are not exported.
GROUPS = {
    'MGn_s999': ('MG', 'native', 'FREEDOM', 999), 'MGp_s999': ('MG', 'plugin', 'FREEDOM', 999),
    'MGn_s1': ('MG', 'native', 'FREEDOM', 1), 'MGp_s1': ('MG', 'plugin', 'FREEDOM', 1),
    'MGn_s2': ('MG', 'native', 'FREEDOM', 2), 'MGp_s2': ('MG', 'plugin', 'FREEDOM', 2),
    'MGn_s3': ('MG', 'native', 'FREEDOM', 3), 'MGp_s3': ('MG', 'plugin', 'FREEDOM', 3),
    'MGn_s4': ('MG', 'native', 'FREEDOM', 4), 'MGp_s4': ('MG', 'plugin', 'FREEDOM', 4),
    'MDVTn_s999': ('MDVT-D', 'native', 'FREEDOM', 999), 'MDVTd_s999': ('MDVT-D', 'plugin', 'FREEDOM', 999),
    'MDVTs_s999': ('MDVT-S', 'plugin', 'FREEDOM', 999),
    'DAMRSn_s999_rawLightGCNv2': ('DA-MRS', 'native', 'LightGCN', 999),
    'DAMRSp_s999_asis': ('DA-MRS', 'plugin', 'LightGCN', 999),
}
# Grid size of the released run script per (plug-in, arm): FREEDOM grid 8 (dropout x reg_weight); MG 8 x alpha2 3;
# MDVT (D) 8 x k 2 x s_weight 5 x threshold 4; MDVT (S) 8 x 2 x 5 x warm_up 6; DA-MRS 1; LightGCN n_ui_layers 3.
EXPECTED = {('MG', 'native'): 8, ('MG', 'plugin'): 24, ('MDVT-D', 'native'): 8, ('MDVT-D', 'plugin'): 320,
            ('MDVT-S', 'plugin'): 480, ('DA-MRS', 'native'): 3, ('DA-MRS', 'plugin'): 1}
# Early-stopping metric (valid_metric) of each released configuration; quick_start ranks combinations by its test value.
SEL_METRIC = {'MG': 'recall@5', 'MDVT-D': 'recall@20', 'MDVT-S': 'recall@20', 'DA-MRS': 'recall@20'}
METRICS = ['recall@5', 'recall@10', 'recall@20', 'ndcg@5', 'ndcg@10', 'ndcg@20']

# (plug-in, variant, backbone, metric, native, +plug-in, paper file, location in the paper)
REPORTED = [
    ('MG', 'MG', 'FREEDOM', 'recall@5', '0.0374', '0.0397', '2402.11262.txt', 'Table 3, Baby, FREEDOM / FREEDOM + MG'),
    ('MDVT-D', 'MDVT (D)', 'FREEDOM', 'recall@10', '0.0627', '0.0648', '2505.16665.txt', 'Table 2, Baby, FREEDOM / +MDVT (D)'),
    ('MDVT-S', 'MDVT (S)', 'FREEDOM', 'recall@10', '0.0627', '0.0652', '2505.16665.txt', 'Table 2, Baby, FREEDOM / +MDVT (S)'),
    ('VI-MMRec', 'VI-MMRec-O', 'FREEDOM', 'recall@10', '0.0627', '0.0663', '2512.08702.txt', 'Table 3, Baby, FREEDOM / +VI-MMRec-O'),
    ('DA-MRS', 'DA-MRS', 'LightGCN', 'recall@20', '0.0738', '0.0994', '2406.12501v1.txt', 'Table 2, Baby, LightGCN / DA-MRS +LightGCN'),
]


def combos():
    groups = json.load(open(f'{ROOT}/results_all_combos.json'))
    w = csv.writer(sys.stdout, delimiter='\t', lineterminator='\n')
    w.writerow(['plugin', 'arm', 'backbone', 'seed', 'expected', 'sel_metric', 'params']
               + [f'valid_{m}' for m in METRICS] + [f'test_{m}' for m in METRICS])
    for g in sorted(groups):
        if g not in GROUPS:
            print(f'SKIP group {g}', file=sys.stderr)
            continue
        plugin, arm, bb, seed = GROUPS[g]
        for r in groups[g]['rows']:
            p = {k: v for k, v in r['params'].items() if k != 'seed'}
            w.writerow([plugin, arm, bb, seed, EXPECTED[(plugin, arm)], SEL_METRIC[plugin],
                        json.dumps(p, sort_keys=True)]
                       + [r['valid'][m] for m in METRICS] + [r['test'][m] for m in METRICS])


def reported():
    w = csv.writer(sys.stdout, lineterminator='\n')
    w.writerow(['plugin', 'variant', 'backbone', 'metric', 'native', 'plugin_value', 'paper_text', 'location'])
    for row in REPORTED:
        text = open(f'{ROOT}/papers/{row[6]}', errors='ignore').read()
        missing = [v for v in row[4:6] if v not in text]
        if missing:
            print(f'NOT FOUND in {row[6]}: {missing} ({row[0]})', file=sys.stderr)
            continue
        w.writerow(row)


{'combos': combos, 'reported': reported}[sys.argv[1]]()
