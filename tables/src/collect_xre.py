"""Collect the per-run results of the proposed method (XRE) and its variants. Only reads remote files.

Usage (local machine, repository root):  python tables/src/collect_xre.py
The script sends itself over ssh to HOST_A (argument `scan` for result.json, `logs` for the per-step event monitor),
gathers one row per candidate run, applies the eligibility rules below, and writes
  tables/xre_runs.tsv     every candidate run with its eligibility (ineligible_reason empty = used)
  tables/xre_perseed.csv  one row per (backbone, row, dataset, seed): mean over the eligible runs of that seed
  tables/xre_cells.csv    one row per (backbone, row, dataset, seed set): mean and sample SD over the seeds of the
                          per-seed means (seed sets 1-5 and 1-3; the Table-1 builder uses the seed set of its Native cell)
  tables/xre_events.tsv   the [R12] / [XRE] monitor lines (every 500 training steps) of the exchange runs
Rows that have no eligible run yet (pending queues) are simply absent; the builders print \\todo for them, so re-running
this collector and the builders fills every pending cell once the runs are synced to HOST_A.

Sources (HOST_A, synced run roots; `rsync -rt` copies of the pod volumes). The adopted setting XRE_SETTING (below)
selects the configurations and roots of every XRE row; SETTINGS lists both settings.
  l003_k20  MSCA: <RUN_ROOT_B>/P04_main_runs/{q12,q12r}/{A,B,E}   package r12 (XRE and its controls)
                  A1: the runs of every root of R14_JUDGE_COMMAND.md (packages objx_s, r8-r14, row <ds>_<pkg>_a1_l003 and
                  objx_s <ds>_aux_a1_existing_l003; bitwise-identical A1 code in every package, unit-tested per package)
            other backbones: P04_main_runs/qtransX/{A,B,E}   packages xret_f (FREEDOM), xret_m (MMGCN), xret_n (MENTOR)
  l010_k40  MSCA: P04_main_runs/qxs2/A, qxs2r/B   package xres, <ds>_xres_{xre,xrf,xen}_l010_k40 and <ds>_xres_a1_l010
            other backbones: P04_main_runs/qtransX/{A,B,E}, qtx2/{A,B}, qtransX2/E and <HOME>/P04_qtx2gB/runs
                  (xret_* <ds>_xre_l010_k40, <ds>_xrf_l010_k40, <ds>_a1_l010)
  sensitivity (both settings): P04_main_runs/qxs/{A,B,E}, qxs2/A, qxs2r/B   package xres (rows of SETTINGS[..]['sens'])
  Instruments (Table I MSCA block, row XRE only; user decision 2026-10-01): <RUN_ROOT_B>/P04_instr/runs*
            package xres instruments_xres_xre_l010_k40 (xre_s2 70225e8, the template of the other datasets; job files and
            repos in P04_instr/README.txt). Instruments runs count from these roots only, and these roots count for
            Instruments only. Every other row (ablation, transfer, sensitivity, events) stays baby / pantry / scientific.
A root that is not synced yet is reported as absent on stderr and read once it exists.
Rules (fixed before the collection)
  1. Eligible run: status SUCCESS, mode train, test_eval best_primed (test metrics of the checkpoint selected by
     validation Recall@20, scoring cache rebuilt), seed 1-5, data canonical_v2 (Baby also accepts canonical
     v1, whose Baby files are byte-identical).
  2. The same physical run seen twice (copied roots: same package, config, seed, test R@20 and elapsed seconds) counts once.
  3. Every eligible run enters: a cell first averages all eligible runs of each seed, then averages over the seeds
     (the rule of the judgement tool a14_select.py, extended to NDCG@20). Standard deviations are over the per-seed means.
"""
import csv
import glob
import json
import os
import re
import statistics as st
import subprocess
import sys

DATASETS = ['baby', 'pantry', 'scientific']
# Extra datasets of the Table I MSCA block (row XRE only): each reads its own roots <base>runs* only, and those roots
# count for it only. Instruments: 2026-10-01 (later removed from the manuscript); Clothing: 2026-10-02 (P04_clothing).
# Table II Clothing (user decision 2026-10-02): Ours (MENTOR + XRE) = xret_n clothing_xre_l010_k40 under P04_t2_clothing.
EXTRA = {'instruments': ['<RUN_ROOT_B>/P04_instr/', '<RUN_ROOT_B>/P04_t2_instruments/'],
         'clothing': ['<RUN_ROOT_B>/P04_clothing/', '<RUN_ROOT_B>/P04_t2_clothing/']}
EXTRA['sports'] = ['<RUN_ROOT_B>/P04_sports/']   # Sports: Table I and Table II in one work dir (2026-10-03)
# Sports component ablation (2026-10-07): HOST_B runs of <RUN_ROOT_C>/P04_sports_abl (branch xres-sports-abl
# 6003cba, sports_xres_a1_l010 and sports_xres_{xro,xfo,xen,xrf}_l010_k40, one run per seed), synced to HOST_A with the
# commands of that README.txt.
SPORTS_ABL_ROOT = '<RUN_ROOT_B>/P04_sports_abl_HOST_B/'
EXTRA['sports'].append(SPORTS_ABL_ROOT)
EXTRA_DATASETS = {('MSCA', 'XRE'): list(EXTRA), ('MENTOR', 'XRE'): ['clothing', 'instruments', 'sports']}
for _row in ('A1', 'wo_A1', 'wo_A1_masked', 'wo_retention', 'wo_masked'):
    EXTRA_DATASETS[('MSCA', _row)] = ['sports']


def datasets_of(bb, row):
    return DATASETS + EXTRA_DATASETS.get((bb, row), [])


def extra_ds_of_root(path):
    for ds, bases in EXTRA.items():
        if any(path.startswith(b + 'runs') for b in bases):
            return ds
    return None
SEEDS = [1, 2, 3, 4, 5]
SEEDSETS = {'12345': [1, 2, 3, 4, 5], '123': [1, 2, 3]}
R = '<RUN_ROOT_B>/P04_main_runs'
F = '<RUN_ROOT_B>/P04_table1fix_runs'

# ================================================================================================================
# XRE_SETTING: the single switch of the adopted XRE hyper-setting (lambda, list size K).
# It selects the configurations (and run roots) behind every XRE row of the manuscript: both Ours rows of Table I
# (MSCA + XRE, MENTOR + XRE), '+ Ours' of Table II, the component ablation figure (and its controls), the exchange-event
# figure, the appendix transfer table (A1 / w/o masked view / XRE per backbone), the marked row of the sensitivity table,
# and the lambda / K macros of the text (build_macros.py). Rows keep their names (XRE, wo_masked, wo_retention, A1, ...)
# in xre_cells.csv for every setting; xre_cells.csv records the setting in its column `setting`, and the builders stop
# when that column differs from XRE_SETTING (re-run this collector after switching).
#   l003_k20  lambda 0.03, K 20: MSCA package r12 (b64663a, r12 xre and its controls), A1 = RULE 4 pool,
#             transfer xret_f / xret_m / xret_n <ds>_{a1,xre,xrf}_l003 (xre_t 1fbb5ca)
#   l010_k40  lambda 0.1, K 40: MSCA package xres (xre_s2 cb67090: <ds>_xres_{xre,xrf,xen}_l010_k40, A1 <ds>_xres_a1_l010),
#             transfer xret_* <ds>_xre_l010_k40 / <ds>_xrf_l010_k40 / <ds>_a1_l010 (xre_t2 2d46fc7)
XRE_SETTING = 'l010_k40'   # adopted (user decision 2026-09-30)
# REPLICATES: the single switch of the replicate rule for the MSCA exchange rows (XRE and its controls).
#   'all'   every eligible run of a seed enters (per-seed mean over all replicates, then the mean over seeds 1-5;
#           amendment 14-1). l010_k40: replicate 1 = qxs2/A, qxs4/A, replicate 2 = qxs2r/B, qxs4r/B.
#   'rep1'  replicate 1 only: every run under a replicate-2 root (SETTINGS[..]['rep2_roots']) is ineligible, so
#           each MSCA row has one run per seed. Rows without a second replicate (A1, transfer, sensitivity) are unchanged.
# The collected setting is recorded as setting_id() in xre_cells.csv / xre_events.tsv ('l010_k40' or 'l010_k40/rep1'),
# and every builder stops when it differs (check_setting). Switch here, then re-run python -s tables/src/make_all.py.
REPLICATES = 'rep1'   # user decision 2026-09-30: one run per seed for every row (replicate 1 roots only)
# ================================================================================================================

A1_PKGS = ['r8', 'r9', 'r10', 'r11', 'r12', 'r13', 'r14']
TRANSFER = (('FREEDOM', 'xret_f'), ('MMGCN', 'xret_m'), ('MENTOR', 'xret_n'))
SETTINGS = {
    'l003_k20': dict(
        lam='0.03', K='20',
        msca={'XRE': [('r12', '{ds}_r12_xre_l003')],
              'wo_masked': [('r12', '{ds}_r12_xrf_l003')],
              'wo_retention': [('r12', '{ds}_r12_xen_l003')],
              'wo_A1': [('r12', '{ds}_r12_xro_l003')],
              'wo_A1_masked': [('r12', '{ds}_r12_xfo_l003')],
              'A1': [('objx_s', '{ds}_aux_a1_existing_l003')] + [(p, '{ds}_' + p + '_a1_l003') for p in A1_PKGS]},
        transfer={'A1': '{ds}_a1_l003', 'XRE': '{ds}_xre_l003', 'wo_masked': '{ds}_xrf_l003'},
        a1_rule4=True,
        rep2_roots=[f'{R}/q12r/{v}' for v in ('A', 'B', 'E')],   # r12 replicate 2 (the A1 RULE-4 pool is unchanged)
        roots=[f'{R}/{q}/{v}' for q in ('q12', 'q12r') for v in ('A', 'B', 'E')]
        + [f'{R}/qtransX/{v}' for v in ('A', 'B', 'E')],
        # sensitivity table: (group, lambda, K, row); row 'XRE' = the adopted setting itself
        sens=[('lambda', '0.01', '20', 'sens_l001_k20'), ('lambda', '0.03', '20', 'XRE'),
              ('lambda', '0.1', '20', 'sens_l010_k20'),
              ('K', '0.03', '10', 'sens_l003_k10'), ('K', '0.03', '20', 'XRE'), ('K', '0.03', '40', 'sens_l003_k40')]),
    'l010_k40': dict(
        lam='0.1', K='40',
        msca={'XRE': [('xres', '{ds}_xres_xre_l010_k40')],
              'wo_masked': [('xres', '{ds}_xres_xrf_l010_k40')],
              'wo_retention': [('xres', '{ds}_xres_xen_l010_k40')],
              'wo_A1': [('xres', '{ds}_xres_xro_l010_k40')],
              'wo_A1_masked': [('xres', '{ds}_xres_xfo_l010_k40')],
              'A1': [('xres', '{ds}_xres_a1_l010')]},
        transfer={'A1': '{ds}_a1_l010', 'XRE': '{ds}_xre_l010_k40', 'wo_masked': '{ds}_xrf_l010_k40'},
        a1_rule4=False,
        rep2_roots=[f'{R}/qxs2r/B', f'{R}/qxs4r/B'],
        roots=[f'{R}/qxs2/A', f'{R}/qxs2r/B', f'{R}/qxs4/A', f'{R}/qxs4r/B'] + [f'{R}/qtransX/{v}' for v in ('A', 'B', 'E')]
        + [f'{R}/qtx2/A', f'{R}/qtx2/B', f'{R}/qtransX2/E', '<HOME>/P04_qtx2gB/runs'],
        sens=[('lambda', '0.03', '40', 'sens_l003_k40'), ('lambda', '0.1', '40', 'XRE'),
              ('lambda', '0.3', '40', 'sens_l030_k40'),
              ('K', '0.1', '20', 'sens_l010_k20'), ('K', '0.1', '40', 'XRE'), ('K', '0.1', '80', 'sens_l010_k80')]),
}
SET = SETTINGS[XRE_SETTING]
assert REPLICATES in ('all', 'rep1'), REPLICATES


def setting_id():
    """identifier of the collected setting: XRE_SETTING, plus '/rep1' under the replicate-1 rule"""
    return XRE_SETTING if REPLICATES == 'all' else f'{XRE_SETTING}/{REPLICATES}'
# sensitivity rows (package xres, roots SENS_ROOTS); only the rows named in SET['sens'] are collected
SENS_CONFIGS = {'sens_l001_k20': '{ds}_xres_l001_k20', 'sens_l010_k20': '{ds}_xres_l010_k20',
                'sens_l003_k10': '{ds}_xres_l003_k10', 'sens_l003_k40': '{ds}_xres_l003_k40',
                'sens_l030_k40': '{ds}_xres_xre_l030_k40', 'sens_l010_k80': '{ds}_xres_xre_l010_k80'}
SENS_ROOTS = [f'{R}/qxs/{v}' for v in ('A', 'B', 'E')] + [f'{R}/qxs2/A', f'{R}/qxs2r/B', f'{R}/qxs3/A', f'{R}/qxs3/B']

ROOTS = list(dict.fromkeys(
    SET['roots'] + SENS_ROOTS
    + [f'{R}/{q}' for q in ('q7/A', 'q7/B', 'q45A', 'q45B', 'q8/A', 'q8/B', 'q8b/A', 'q8b/B', 'q8c/A', 'q8c/B',
                             'q8c/E', 'q9/A', 'q9/B', 'q10/E', 'q10r/A', 'q10r/B', 'q10r/E', 'q11/A', 'q11/B',
                             'q13/A', 'q13/B', 'q13/E', 'q14/A', 'q14/B', 'q14/E', 'q14r/A', 'q14r/B', 'q14r/E')]
    + [f'{F}/{q}' for q in ('qfixE', 'qfixA', 'qfixB')]))
STAGE_A = '<RUN_ROOT_B>/P04_objxs_runs/stageA'   # seed-1 objx_s A1 references of the a14 judgement
ROOTS.append(STAGE_A)
# RULE 4 (official cell definition, <TOOLS>/thesis_share/OFFICIAL_CELLS.md; setting l003_k20): the A1 row is the
# a14_select A1 pool of the Round-14 judgement, i.e. the A1 runs under the roots of q14_select/roots.txt plus the seed-1
# stage-A objx_s A1 run (a14 --tsv prereg/r7_refs_seed1_HOST_A.tsv), n = 45 runs. Other A1 runs (e.g. q10r) are not pooled.
A1_ROOTS = [f'{R}/{q}' for q in ('q7/A', 'q7/B', 'q45A', 'q45B', 'q8/A', 'q8/B', 'q8b/A', 'q8b/B', 'q9/A', 'q9/B', 'q11/A',
                                  'q11/B', 'q8c/A', 'q8c/B', 'q8c/E', 'q12/A', 'q12/B', 'q12r/A', 'q12r/B', 'q12r/E',
                                  'q14/A', 'q14/B', 'q14r/A', 'q14r/B', 'q10/E', 'q13/A', 'q13/B')] \
    + [f'{F}/{q}' for q in ('qfixE', 'qfixA', 'qfixB')]
PACKAGES = {'r12', 'xres', 'xret_f', 'xret_m', 'xret_n', 'objx_s'} | set(A1_PKGS)

# (backbone, row) -> list of (package, config template). The printed labels live in the builders.
ROWS = {('MSCA', k): v for k, v in SET['msca'].items()}
for _bb, _pkg in TRANSFER:
    ROWS[(_bb, 'Native')] = [(_pkg, '{ds}_native_l000')]   # information (bitwise = the Native runs of Tables I / II)
    for _row, _tmpl in SET['transfer'].items():
        ROWS[(_bb, _row)] = [(_pkg, _tmpl)]
for _g, _l, _k, _row in SET['sens']:
    if _row != 'XRE':
        ROWS[('MSCA', _row)] = [('xres', SENS_CONFIGS[_row])]
if XRE_SETTING == 'l003_k20':
    ROWS[('MSCA', 'XRE_r14')] = [('r14', '{ds}_r14_xre_l003')]   # independent replicate of XRE in package r14 (information)
SENS_ROWS = {row for _, _, _, row in SET['sens'] if row != 'XRE'}
# exchange rows whose [R12] / [XRE] monitor lines are collected (event figure: MSCA XRE vs wo_masked)
EXCHANGE_ROWS = ('XRE', 'wo_masked', 'wo_retention', 'wo_A1', 'wo_A1_masked')
EXCHANGE_CONFIGS = {tmpl.format(ds=ds) for (bb, row), alts in ROWS.items() if row in EXCHANGE_ROWS
                    for _, tmpl in alts for ds in DATASETS}

FIELDS = ['package', 'config', 'dataset', 'seed', 'status', 'mode', 'test_eval', 'data', 'gpu', 'started', 'elapsed_sec',
          'n_batches', 'best_epoch', 'test_r10', 'test_n10', 'test_r20', 'test_n20', 'path']
EV_FIELDS = ['package', 'config', 'dataset', 'seed', 'run', 'train_interactions', 'step', 'held_rows', 'entry_rows',
             'p_in_list', 'retained_per_row', 'mass_entry', 'mass_retention', 'scale', 'term']


# ----------------------------------------------------------------------------------------------- remote side
def run_dirs():
    for root in ROOTS + sorted(x for bs in EXTRA.values() for b in bs for x in glob.glob(b + 'runs*') if os.path.isdir(x)):
        if not os.path.isdir(root):
            print(f'root absent: {root}', file=sys.stderr)
            continue
        for f in glob.glob(os.path.join(glob.escape(root), '**', 'result.json'), recursive=True):
            name = os.path.basename(os.path.dirname(f))
            pkg = name.split('__')[0]
            if pkg in PACKAGES:
                yield f, pkg, name


def scan():
    w = csv.writer(sys.stdout, delimiter='\t', lineterminator='\n')
    w.writerow(FIELDS)
    for f, pkg, name in run_dirs():
        try:
            with open(f, encoding='utf-8') as fh:
                d = json.load(fh)
        except Exception as e:  # e.g. a file caught mid-rsync: reported, skipped
            print(f'UNREADABLE {f} {type(e).__name__}', file=sys.stderr)
            continue
        bt = d.get('best_test') or {}
        w.writerow([d.get('package'), d.get('config'), str(d.get('dataset', '')).split('@')[0], d.get('seed'),
                    d.get('status'), d.get('mode'), d.get('test_eval'),
                    os.path.basename(str(d.get('data_root', '')).rstrip('/')), (d.get('gpu') or '').replace('NVIDIA ', ''),
                    d.get('started'), d.get('elapsed_sec'), d.get('n_batches'), d.get('best_epoch'), bt.get('recall@10'),
                    bt.get('ndcg@10'), bt.get('recall@20'), bt.get('ndcg@20'), f])


def logs():
    w = csv.writer(sys.stdout, delimiter='\t', lineterminator='\n')
    w.writerow(EV_FIELDS)
    head = re.compile(r'history_keys=(\d+)')
    step = re.compile(r'\[(?:R12|XRE)\] step=(\d+) (.*)')
    for f, pkg, name in run_dirs():
        parts = name.split('__')
        if len(parts) < 4 or parts[1] not in EXCHANGE_CONFIGS:
            continue
        log = os.path.join(os.path.dirname(f), 'stdout.log')
        if not os.path.isfile(log):
            continue
        n_train = ''
        with open(log, encoding='utf-8', errors='replace') as fh:
            for chunk in fh:
                for line in chunk.split('\r'):
                    m = head.search(line)
                    if m and not n_train:
                        n_train = m.group(1)
                    m = step.search(line)
                    if not m:
                        continue
                    kv = dict(x.split('=', 1) for x in m.group(2).split() if '=' in x)
                    w.writerow([pkg, parts[1], parts[2].split('@')[0], parts[3].lstrip('s'), f, n_train, m.group(1)]
                               + [kv.get(k, '') for k in EV_FIELDS[7:]])


# ------------------------------------------------------------------------------------------------ local side
HERE = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.dirname(HERE)
SSH = ['ssh', '-o', 'BatchMode=yes', 'HOST_A']


def check_setting(path=None):
    """Builders call this first: stop when xre_cells.csv was collected under another XRE_SETTING."""
    path = path or os.path.join(TABLES, 'xre_cells.csv')
    if not os.path.exists(path):
        return
    with open(path, newline='', encoding='utf-8') as fh:
        got = {r.get('setting') or 'unrecorded' for r in csv.DictReader(fh)}
    if got - {setting_id()}:
        raise SystemExit(f'{path} was collected with setting {sorted(got)}, the switches give {setting_id()}: '
                         're-run tables/src/collect_xre.py (python -s tables/src/make_all.py)')


def remote(mode):
    with open(os.path.abspath(__file__), 'rb') as fh:
        out = subprocess.run(SSH + ['python3', '-', mode], stdin=fh, capture_output=True, check=True)
    if out.stderr.strip():
        print(out.stderr.decode(errors='replace'), file=sys.stderr)
    return list(csv.DictReader(out.stdout.decode().splitlines(), delimiter='\t'))


def ineligible(r):
    why = []
    if r['status'] != 'SUCCESS':
        why.append(f"status {r['status']}")
    if r['mode'] != 'train':
        why.append(f"mode {r['mode']}")
    if r['test_eval'] != 'best_primed':
        why.append(f"test_eval {r['test_eval'] or 'none'}")
    if r['test_r20'] in ('', 'None'):
        why.append('no test R@20')
    try:
        if int(r['seed']) not in SEEDS:
            why.append('seed outside 1-5')
    except ValueError:
        why.append('seed unreadable')
    # (device-name filter of the original collector removed in the public copy; other rules unchanged)
    if not (r['data'] == 'canonical_v2' or (r['dataset'] == 'baby' and r['data'] == 'canonical')):
        why.append(f"data {r['data']}")
    return '; '.join(why)


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def sd(xs):
    return st.stdev(xs) if len(xs) > 1 else None


def f6(x):
    return '' if x is None else f'{x:.6f}'


def main():
    rows = remote('scan')
    wanted = {}
    for (bb, row), alts in ROWS.items():
        for pkg, tmpl in alts:
            for ds in datasets_of(bb, row):
                wanted[(pkg, tmpl.format(ds=ds))] = (bb, row, ds)
    seen = set()
    for r in rows:
        key = wanted.get((r['package'], r['config']))
        r['backbone'], r['row'] = (key[0], key[1]) if key else ('', '')
        r['ineligible_reason'] = ineligible(r) if key else 'not a collected row'
        need = r['dataset'] if r['dataset'] in EXTRA else None      # the extra dataset whose roots this row must read
        if key and not r['ineligible_reason'] and extra_ds_of_root(r['path']) != need:
            r['ineligible_reason'] = 'extra-dataset rows read their own roots only, and those roots count for them only'
        if SET['a1_rule4'] and key and key[:2] == ('MSCA', 'A1') and not r['ineligible_reason']:   # RULE 4
            in_roots = any(r['path'].startswith(x + '/') for x in A1_ROOTS)
            stage_a = r['path'].startswith(STAGE_A + '/') and r['seed'] == '1'
            if not (in_roots or stage_a):
                r['ineligible_reason'] = 'A1 run outside the a14 judgement roots (RULE 4)'
        if key and r['path'].startswith(STAGE_A + '/') and key[:2] != ('MSCA', 'A1'):
            r['ineligible_reason'] = 'stage-A root is read for the A1 row only'
        if (REPLICATES == 'rep1' and key and not r['ineligible_reason']
                and any(r['path'].startswith(x + '/') for x in SET['rep2_roots'])):
            r['ineligible_reason'] = 'replicate 2 (REPLICATES = rep1)'
        if not r['ineligible_reason']:
            ident = (r['package'], r['config'], r['seed'], r['test_r20'], round(float(r['elapsed_sec'] or 0)))
            if ident in seen:
                r['ineligible_reason'] = 'duplicate copy of an already counted run'
            seen.add(ident)
    rows.sort(key=lambda r: (r['backbone'], r['row'], r['dataset'], r['seed'], r['path']))
    cols = ['backbone', 'row', 'ineligible_reason'] + FIELDS
    with open(os.path.join(TABLES, 'xre_runs.tsv'), 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter='\t', lineterminator='\n', extrasaction='ignore')
        w.writeheader()
        w.writerows(r for r in rows if r['backbone'])
    per = {}
    for r in rows:
        if r['backbone'] and not r['ineligible_reason']:
            per.setdefault((r['backbone'], r['row'], r['dataset'], int(r['seed'])), []).append(r)
    with open(os.path.join(TABLES, 'xre_perseed.csv'), 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh, lineterminator='\n')
        w.writerow(['backbone', 'row', 'dataset', 'seed', 'n_runs', 'r20', 'n20', 'r10', 'n10'])
        for k in sorted(per):
            rr = per[k]
            w.writerow(list(k) + [len(rr)] + [f6(mean([float(x[m]) for x in rr]))
                                              for m in ('test_r20', 'test_n20', 'test_r10', 'test_n10')])
    with open(os.path.join(TABLES, 'xre_cells.csv'), 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh, lineterminator='\n')
        w.writerow(['backbone', 'row', 'dataset', 'seedset', 'n', 'seeds', 'runs_per_seed', 'r20_mean', 'r20_std',
                    'n20_mean', 'n20_std', 'r10_mean', 'n10_mean', 'packages', 'setting'])
        for (bb, row) in ROWS:
            for ds in datasets_of(bb, row):
                for name, sset in SEEDSETS.items():
                    got = [(s, per[(bb, row, ds, s)]) for s in sset if (bb, row, ds, s) in per]
                    if not got:
                        continue
                    ps = {m: [mean([float(x[m]) for x in rr]) for _, rr in got]
                          for m in ('test_r20', 'test_n20', 'test_r10', 'test_n10')}
                    w.writerow([bb, row, ds, name, len(got), '/'.join(str(s) for s, _ in got),
                                '/'.join(str(len(rr)) for _, rr in got), f6(mean(ps['test_r20'])), f6(sd(ps['test_r20'])),
                                f6(mean(ps['test_n20'])), f6(sd(ps['test_n20'])), f6(mean(ps['test_r10'])),
                                f6(mean(ps['test_n10'])),
                                '/'.join(sorted({x['package'] for _, rr in got for x in rr})), setting_id()])
    ev = remote('logs')
    used = {r['path']: (r['backbone'], r['row']) for r in rows if r['backbone'] and not r['ineligible_reason']}
    for e in ev:
        e['backbone'], e['row'] = used.get(e['run'], ('', ''))
        e['setting'] = setting_id()
    with open(os.path.join(TABLES, 'xre_events.tsv'), 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=['backbone', 'row', 'setting'] + EV_FIELDS, delimiter='\t',
                           lineterminator='\n')
        w.writeheader()
        w.writerows(e for e in ev if e['run'] in used)
    n_ok = sum(1 for r in rows if r['backbone'] and not r['ineligible_reason'])
    print(f"{sum(1 for r in rows if r['backbone'])} candidate runs, {n_ok} eligible, "
          f"{len(per)} (row, dataset, seed) groups, {sum(1 for e in ev if e['run'] in used)} event lines", file=sys.stderr)


if __name__ == '__main__':
    if len(sys.argv) >= 2 and sys.argv[1] in ('scan', 'logs'):
        {'scan': scan, 'logs': logs}[sys.argv[1]]()
    else:
        main()
