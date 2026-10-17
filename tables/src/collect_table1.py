"""Collect the per-run results of Table II (backbone x plug-in objective: rows Native and the six plug-ins; the + Ours
row comes from tables/xre_cells.csv). Only reads remote files.

Usage (local machine, repository root):  python tables/src/collect_table1.py
The script sends itself over ssh to HOST_A (argument `scan`), gathers one row per candidate run, applies the rules
below, and writes
  tables/table1_runs.tsv     every candidate run with its eligibility and the chosen flag (ineligible_reason empty and
                             chosen 1 = used)
  tables/table1_perseed.csv  one row per (backbone, row, dataset, seed): the chosen run of that seed
  tables/table1_cells.csv    one row per (backbone, row, dataset): mean and sample SD over the chosen seeds
                             (columns of the earlier HOST_A file + gpus, roots). A cell with fewer than five seeds is
                             still written (n < 5); the builders print it as \\todo{pending} until every seed exists.
The same Native cells are read by every table that prints a Native row (Table I MSCA row, Table II, appendix transfer
and protocol tables) through tables/src/native_rows.py, so a Native value is printed from one run set everywhere.

Configurations (fixed before this collection; package / config, <ds> = baby, pantry, scientific)
  MSCA     Native objx_s <ds>_native | MG objx_s <ds>_mg_<a*> | MDVT plg_s <ds>_mdvt_n2_l02_g01 |
           VI-MMRec plg_s <ds>_vio_k10_l001 | DA-MRS plg2_s <ds>_damrs_a1b2_k10_x2 |
           UltraGCN-LI plg2_s <ds>_ugcn_li_k10_l0007 | INA plg2_s <ds>_ina_cooc_k10_d04_l06
  FREEDOM  plg_f <ds>_a4_{native, mg_<a*>, mdvt_n2_l02_g01, vio_k10_l001} |
           plg2_f <ds>_a4_{damrs_a1b2_k10_x2, ugcn_li_k10_l0007, ina_cooc_k10_d04_l06}
  MMGCN    plg2_m <ds>_a4_{native, mg_<a*>, mdvt_n4_l02_g01, vio_k10_l001, damrs_a1b2_k10_x2, ugcn_li_k10_l0007,
           ina_cooc_k10_d04_l06}
  DA-MRS (alpha 1, beta 2) and UltraGCN L_I (lambda 0.007) are the configurations fixed by the pre-set port rules
  (author decision log 2026-09-29T10:08:19Z, branch table1fix 198b975; amendment 14 plug-in bar).
  MG a* = the alpha2 chosen by plugins_msf_plan.select_mg (highest seed-1 VALID Recall@20 of the {0.1, 0.2, 0.3} grid),
  as recorded in the source column of the earlier table1_cells.csv (2026-09-29); it is frozen here (MG_ASTAR) so that a
  re-run of a seed cannot change the selected configuration.
  Instruments (MSCA block only, user decision 2026-10-01; the FREEDOM / MMGCN blocks stay baby / pantry / scientific):
  the same configurations with <ds> = instruments (objx_s instruments_native / instruments_mg_<a*>, plg_s, plg2_s; the
  instruments configs equal the scientific ones up to the dataset name; DA-MRS / UltraGCN L_I configs table1fix 60b5b92).
  Its runs live under the Instruments roots INSTR_GLOB (<RUN_ROOT_B>/P04_instr/runs*; job files and repos in
  P04_instr/README.txt). MG a* on Instruments is not frozen yet: mg_astar_instruments() applies the select_mg rule (seed-1
  VALID Recall@20 of best_valid_primed, ties VALID NDCG@20, then smaller alpha2) to the eligible seed-1 grid runs (earliest
  completion per alpha2, RULE 3) and leaves the MG cell pending until all three grid runs exist. Freeze it in MG_ASTAR
  once chosen.
Rules (fixed before this collection)
  1. Eligible run: status SUCCESS, mode train, test_eval best_primed (test metrics of the checkpoint selected by
     validation Recall@20, scoring cache rebuilt), seed 1-5, data copy canonical_v2_vimmrec for the
     VI-MMRec rows and canonical_v2 otherwise (Baby also accepts canonical v1, whose Baby files are byte-identical).
  2. Only runs under the designated Table-1 roots (ROOTS) count: the Table-1 queues of the three backbones (seeds 1-3),
     the seed 4-5 extension q45, the corrected-port queue qfix, the re-runs of the MSCA Native seeds 2-3 (q7,
     the a14 references) and the seed-1 stage-A root (MSCA Native seed 1, the a14 reference), and qt2s (runs
     added for this table: MSCA MG a* seed 1, MMGCN MG a* seeds 2-3). Archived pod runs, the original-
     protocol check (origcheck), the plug-in fix-arm runs (plugfix) and other queues (e.g. qtrans) are not read.
     Instruments cells read the Instruments roots only, and those roots are read for Instruments runs only.
  3. One run per (cell, seed): if a designated root holds more than one eligible run of the same seed (other than the
     same physical run seen twice), the earliest completion (started + elapsed_sec) is chosen. Value-independent.
  4. Cell = mean over the chosen seeds, SD = sample SD over seeds (ddof 1).
"""
import csv
import datetime as dt
import glob
import json
import os
import statistics as st
import subprocess
import sys

DATASETS = ['baby', 'pantry', 'scientific']
# Extra datasets (MSCA block only): each reads its own roots <base>runs* only, and those roots count for it only.
# Instruments: user decision 2026-10-01 (later removed from the manuscript); Clothing: user decision 2026-10-02
# (configs = the scientific ones up to the dataset name, P04_clothing/README.txt).
EXTRA = {'instruments': '<RUN_ROOT_B>/P04_instr/', 'clothing': '<RUN_ROOT_B>/P04_clothing/',
         'sports': '<RUN_ROOT_B>/P04_sports/'}   # Sports: 2026-10-03
DATASETS_OF = {'MSCA': DATASETS + list(EXTRA), 'FREEDOM': DATASETS, 'MMGCN': DATASETS}
MG_GRID = {'a01': 0.1, 'a02': 0.2, 'a03': 0.3}
SEEDS = [1, 2, 3, 4, 5]
R = '<RUN_ROOT_B>/P04_main_runs'
F = '<RUN_ROOT_B>/P04_table1fix_runs'
ROOTS = ([f'{R}/{q}' for q in ('volA', 'volB', 'q2A', 'q2B', 'qcA', 'qcB', 'q45A', 'q45B', 'q7/A', 'q7/B',
                               'qt2s/A', 'qt2s/B')]
         + [f'{F}/{q}' for q in ('qfixA', 'qfixB', 'qfixE')]
         + ['<RUN_ROOT_B>/P04_objxs_runs/stageA'])
PACKAGES = {'objx_s', 'plg_s', 'plg2_s', 'plg_f', 'plg2_f', 'plg2_m'}
MG_ASTAR = {('MSCA', 'baby'): 'a03', ('MSCA', 'pantry'): 'a02', ('MSCA', 'scientific'): 'a02',
            ('FREEDOM', 'baby'): 'a02', ('FREEDOM', 'pantry'): 'a02', ('FREEDOM', 'scientific'): 'a02',
            ('MMGCN', 'baby'): 'a03', ('MMGCN', 'pantry'): 'a01', ('MMGCN', 'scientific'): 'a03'}
DAMRS, UGCN, INA = 'damrs_a1b2_k10_x2', 'ugcn_li_k10_l0007', 'ina_cooc_k10_d04_l06'
SPEC = {  # backbone -> [(row, package, config template)]; '{g}' = MG a*
    'MMGCN': [('Native', 'plg2_m', '{ds}_a4_native'), ('MG', 'plg2_m', '{ds}_a4_mg_{g}'),
              ('MDVT', 'plg2_m', '{ds}_a4_mdvt_n4_l02_g01'), ('VI-MMRec', 'plg2_m', '{ds}_a4_vio_k10_l001'),
              ('DA-MRS', 'plg2_m', '{ds}_a4_' + DAMRS), ('UltraGCN-LI', 'plg2_m', '{ds}_a4_' + UGCN),
              ('INA', 'plg2_m', '{ds}_a4_' + INA)],
    'FREEDOM': [('Native', 'plg_f', '{ds}_a4_native'), ('MG', 'plg_f', '{ds}_a4_mg_{g}'),
                ('MDVT', 'plg_f', '{ds}_a4_mdvt_n2_l02_g01'), ('VI-MMRec', 'plg_f', '{ds}_a4_vio_k10_l001'),
                ('DA-MRS', 'plg2_f', '{ds}_a4_' + DAMRS), ('UltraGCN-LI', 'plg2_f', '{ds}_a4_' + UGCN),
                ('INA', 'plg2_f', '{ds}_a4_' + INA)],
    'MSCA': [('Native', 'objx_s', '{ds}_native'), ('MG', 'objx_s', '{ds}_mg_{g}'),
             ('MDVT', 'plg_s', '{ds}_mdvt_n2_l02_g01'), ('VI-MMRec', 'plg_s', '{ds}_vio_k10_l001'),
             ('DA-MRS', 'plg2_s', '{ds}_' + DAMRS), ('UltraGCN-LI', 'plg2_s', '{ds}_' + UGCN),
             ('INA', 'plg2_s', '{ds}_' + INA)],
}
FIELDS = ['package', 'config', 'dataset', 'seed', 'status', 'mode', 'test_eval', 'data', 'gpu', 'started', 'elapsed_sec',
          'best_epoch', 'fingerprint', 'test_r10', 'test_n10', 'test_r20', 'test_n20', 'valid_r20', 'valid_n20', 'path']


# ----------------------------------------------------------------------------------------------- remote side
def extra_roots():
    return sorted(x for b in EXTRA.values() for x in glob.glob(b + 'runs*') if os.path.isdir(x))


def scan():
    w = csv.writer(sys.stdout, delimiter='\t', lineterminator='\n')
    w.writerow(FIELDS)
    for root in ROOTS + extra_roots():
        if not os.path.isdir(root):
            print(f'root absent: {root}', file=sys.stderr)
            continue
        for pkg in sorted(PACKAGES):
            for f in glob.glob(os.path.join(glob.escape(root), pkg, '*', 'result.json')):
                try:
                    with open(f, encoding='utf-8') as fh:
                        d = json.load(fh)
                except Exception as e:  # e.g. a file caught mid-rsync: reported, skipped
                    print(f'UNREADABLE {f} {type(e).__name__}', file=sys.stderr)
                    continue
                bt = d.get('best_test') or {}
                bv = d.get('best_valid_primed') or {}
                w.writerow([d.get('package'), d.get('config'), str(d.get('dataset', '')).split('@')[0], d.get('seed'),
                            d.get('status'), d.get('mode'), d.get('test_eval'),
                            os.path.basename(str(d.get('data_root', '')).rstrip('/')),
                            (d.get('gpu') or '').replace('NVIDIA ', ''), d.get('started'), d.get('elapsed_sec'),
                            d.get('best_epoch'), d.get('fingerprint'), bt.get('recall@10'), bt.get('ndcg@10'),
                            bt.get('recall@20'), bt.get('ndcg@20'), bv.get('recall@20'), bv.get('ndcg@20'), f])


# ------------------------------------------------------------------------------------------------ local side
HERE = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.dirname(HERE)
SSH = ['ssh', '-o', 'BatchMode=yes', 'HOST_A']


def remote(mode):
    with open(os.path.abspath(__file__), 'rb') as fh:
        out = subprocess.run(SSH + ['python3', '-', mode], stdin=fh, capture_output=True, check=True)
    if out.stderr.strip():
        print(out.stderr.decode(errors='replace'), file=sys.stderr)
    return list(csv.DictReader(out.stdout.decode().splitlines(), delimiter='\t'))


def wanted():
    out = {}
    for bb, rows in SPEC.items():
        for row, pkg, tmpl in rows:
            for ds in DATASETS_OF[bb]:
                if '{g}' in tmpl and not MG_ASTAR.get((bb, ds)):
                    continue                                   # MG a* not selected yet (Instruments grid pending)
                out[(pkg, tmpl.format(ds=ds, g=MG_ASTAR.get((bb, ds))))] = (bb, row, ds)
    return out


def extra_ds_of_root(path):
    """The extra dataset whose roots hold this path, else None."""
    for ds, b in EXTRA.items():
        if path.startswith(b + 'runs'):
            return ds
    return None


def mg_astar_extra(rows, ds):
    """plugins_msf_plan.select_mg on the seed-1 grid of MSCA (objx_s <ds>_mg_a0{1,2,3}) of an extra dataset: eligible
    runs under its roots, earliest completion per alpha2; (None, {}) until all three exist."""
    c = []
    for tag, a in MG_GRID.items():
        rr = [r for r in rows if r['package'] == 'objx_s' and r['config'] == f'{ds}_mg_{tag}' and r['seed'] == '1'
              and extra_ds_of_root(r['path']) == ds and not ineligible(r, 'MG') and r['valid_r20'] not in ('', 'None')]
        if not rr:
            return None, {}
        r = sorted(rr, key=lambda x: (completed(x), x['path']))[0]
        c.append((-float(r['valid_r20']), -float(r['valid_n20'] or 0.0), a, tag))
    return sorted(c)[0][3], {t: -v for v, _, _, t in c}


def ineligible(r, row):
    why = []
    if r['status'] != 'SUCCESS':
        why.append(f"status {r['status']}")
    if r['mode'] != 'train':
        why.append(f"mode {r['mode']}")
    if r['test_eval'] != 'best_primed':
        why.append(f"test_eval {r['test_eval'] or 'none'}")
    if r['test_r20'] in ('', 'None') or r['test_n20'] in ('', 'None'):
        why.append('no test metrics')
    try:
        if int(r['seed']) not in SEEDS:
            why.append('seed outside 1-5')
    except ValueError:
        why.append('seed unreadable')
    # (device-name filter of the original collector removed in the public copy; other rules unchanged)
    ok_data = (r['data'] == 'canonical_v2_vimmrec' if row == 'VI-MMRec' else
               (r['data'] == 'canonical_v2' or (r['dataset'] == 'baby' and r['data'] == 'canonical')))
    if not ok_data:
        why.append(f"data {r['data']}")
    return '; '.join(why)


def completed(r):
    try:
        t = dt.datetime.strptime(r['started'], '%Y-%m-%d %H:%M:%S')
        return t + dt.timedelta(seconds=float(r['elapsed_sec'] or 0))
    except (TypeError, ValueError):
        return dt.datetime.max


def root_of(path):
    for x in ROOTS:
        if path.startswith(x + '/'):
            return x.replace(R + '/', '').replace(F + '/', '').replace('<RUN_ROOT_B>/P04_objxs_runs/', '')
    ds = extra_ds_of_root(path)
    if ds:
        b = EXTRA[ds]
        return os.path.basename(b.rstrip('/')) + '/' + path[len(b):].split('/')[0]
    return '?'


def f6(x):
    return '' if x is None else f'{x:.6f}'


def main():
    rows = remote('scan')
    for ds in EXTRA:
        if not MG_ASTAR.get(('MSCA', ds)):
            a, grid = mg_astar_extra(rows, ds)
            print(f"MSCA MG a* on {ds}: {a or 'pending (seed-1 grid incomplete)'}; seed-1 valid R@20 {grid}",
                  file=sys.stderr)
            if a:
                MG_ASTAR[('MSCA', ds)] = a
    want = wanted()
    seen = set()
    for r in rows:
        key = want.get((r['package'], r['config']))
        r['backbone'], r['row'] = (key[0], key[1]) if key else ('', '')
        r['root'] = root_of(r['path'])
        r['chosen'] = '0'
        r['ineligible_reason'] = ineligible(r, key[1]) if key else 'not a Table-II configuration'
        need = r['dataset'] if r['dataset'] in EXTRA else None      # the extra dataset whose roots this cell must read
        if key and not r['ineligible_reason'] and extra_ds_of_root(r['path']) != need:
            r['ineligible_reason'] = 'extra-dataset cells read their own roots only, and those roots count for them only'
        if not r['ineligible_reason']:
            ident = (r['package'], r['config'], r['seed'], r['test_r20'], round(float(r['elapsed_sec'] or 0)))
            if ident in seen:
                r['ineligible_reason'] = 'duplicate copy of an already counted run'
            seen.add(ident)
    groups = {}
    for r in rows:
        if r['backbone'] and not r['ineligible_reason']:
            groups.setdefault((r['backbone'], r['row'], r['dataset'], int(r['seed'])), []).append(r)
    chosen = {}
    for k, rr in groups.items():
        rr.sort(key=lambda x: (completed(x), x['path']))                                         # RULE 3
        rr[0]['chosen'] = '1'
        for x in rr[1:]:
            x['ineligible_reason'] = f"later duplicate of seed {k[3]} (RULE 3: earliest completion {rr[0]['root']})"
        chosen[k] = rr[0]
    rows = [r for r in rows if r['backbone']]
    rows.sort(key=lambda r: (r['backbone'], r['row'], r['dataset'], r['seed'], r['path']))
    cols = ['backbone', 'row', 'chosen', 'ineligible_reason', 'root'] + FIELDS
    with open(os.path.join(TABLES, 'table1_runs.tsv'), 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter='\t', lineterminator='\n', extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(TABLES, 'table1_perseed.csv'), 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh, lineterminator='\n')
        w.writerow(['backbone', 'row', 'dataset', 'seed', 'r20', 'n20', 'r10', 'n10', 'gpu', 'root', 'fingerprint'])
        for k in sorted(chosen):
            c = chosen[k]
            w.writerow(list(k) + [c['test_r20'], c['test_n20'], c['test_r10'], c['test_n10'], c['gpu'], c['root'],
                                  c['fingerprint']])
    missing = []
    with open(os.path.join(TABLES, 'table1_cells.csv'), 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh, lineterminator='\n')
        w.writerow(['backbone', 'row', 'dataset', 'n', 'r20_mean', 'r20_std', 'n20_mean', 'n20_std', 'seeds', 'source',
                    'gpus', 'roots'])
        for bb, spec in SPEC.items():
            for row, pkg, tmpl in spec:
                for ds in DATASETS_OF[bb]:
                    got = [chosen[(bb, row, ds, s)] for s in SEEDS if (bb, row, ds, s) in chosen]
                    missing += [f'{bb}/{row}/{ds}/s{s}' for s in SEEDS if (bb, row, ds, s) not in chosen]
                    src = f"{pkg}/{tmpl.format(ds=ds, g=MG_ASTAR.get((bb, ds)) or '<a* pending>')}"
                    if not got:
                        continue
                    r20 = [float(x['test_r20']) for x in got]
                    n20 = [float(x['test_n20']) for x in got]
                    sd = (lambda v: st.stdev(v) if len(v) > 1 else None)
                    w.writerow([bb, row, ds, len(got), f6(st.mean(r20)), f6(sd(r20)), f6(st.mean(n20)), f6(sd(n20)),
                                '/'.join(x['seed'] for x in got), src, '/'.join(sorted({x['gpu'] for x in got})),
                                '/'.join(x['root'] for x in got)])
    n_ok = sum(r['chosen'] == '1' for r in rows)
    print(f'{len(rows)} candidate runs, {n_ok} chosen; missing (cell/seed): {len(missing)}'
          + (' -> ' + ', '.join(missing) if missing else ''), file=sys.stderr)


if __name__ == '__main__':
    if len(sys.argv) >= 2 and sys.argv[1] == 'scan':
        scan()
    else:
        main()
