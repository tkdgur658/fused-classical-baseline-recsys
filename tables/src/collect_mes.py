"""Collect the MES runs (masked entry supervision = entry pairs only, no retention, no anchor term) into
tables/mes_perseed.csv, the file that tables/src/mes_source.py reads to replace the provisional red values.

Runs (HOST_A): <RUN_ROOT_B>/P04_xen0/runs_xen0/xres0/*/result.json, package xres0, config
<ds>_xres_xeoa0_l010_k40 (lambda 0.1, K 40), repo <HOME>/P04_novctrl_wt@75d079a (README in P04_xen0).
Eligible run: status SUCCESS, mode train, test_eval best_primed; seeds 1-5, one run per seed (a second eligible run of
one (dataset, seed) stops the script: no replicates). Metrics: best_test recall@20 / ndcg@20 (best-validation checkpoint).
Only MSCA is collected here; COHESION / SMORE MES runs do not exist yet (mes_source.py keeps them provisional).
The file is written only when all four datasets have seeds 1-5, unless --partial is given (mes_source.py then switches
each complete (backbone, dataset) cell separately).
Added 2026-10-10 (r20_R1/MANU queue, HOST_A <DATA_ROOT>/MANU/runs, one run per seed):
  MSCA row MES_wd   xres0 <ds>_xres_xeoa0_l010_k40_wd01 (P04_novctrl_wt 6bea495; MES + decoupled AdamW weight decay 0.1 =
                    the P04_wdcmp protocol; read by build_ctrl.py macros_wd, not by mes_source.py)
  COHESION / SMORE row MES   mes_c / mes_s <ds>_mes_l010_k40 (branch mescs cac56bd, P04_mescs_wt; MesMixin over XRET_C /
                    XRET_S: entry part only, no retention, anchor not added, lambda 0.1, K 40)
  These cells are written as soon as a (backbone, row, dataset) cell has seeds 1-5 (mes_source.py switches per cell).
Changed 2026-10-10 10:xx (author: mestune adopted by the validation rule; approved runs A):
  MSCA row MES      the main-table MES = the lambda / K selected per dataset by seed-1 VALIDATION Recall@20
                    (trainer_best_valid of P04_mestune and P04_xen0; test never used): SELECTED below. Runs from P04_mestune
                    (<RUN_ROOT_B>/P04_mestune/runs_mestune), P04_xen0 or the MANU queue (Sports K 80 seeds 2-5).
  MSCA row MES_k40  lambda 0.1, K 40 (P04_xen0): the setting of every Table III control row (build_ctrl.py uses this row)
  MSCA row MES_wd   (from 11:1x) the main setting + weight decay 0.1 (configs <ds>_xres_xeoa0_<SELECTED>_wd01); the earlier
                    lambda 0.1, K 40 + wd runs are kept as row MES_wd_k40
  MSCA row MES_fv   xres0f <ds>_xres_xeoa0f_l010_k40 (novctrl 8563895): MES with the entry pairs on the full view (Table III)
  FREEDOM / MMGCN / MENTOR row MES   mes_f / mes_m / mes_n <ds>_mes_l010_k40 (branch mesfmn c84a522; tab:transfer)
  Every row is written per complete (backbone, row, dataset) cell (seeds 1-5); --partial is kept for compatibility.
Usage (repository root):  python -s tables/src/collect_mes.py [--partial]
then:                     python -s tables/src/make_all.py --no-collect
"""
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'tables' / 'mes_perseed.csv'
W = '<RUN_ROOT_B>/P04_xen0'
MANU = '<DATA_ROOT>/MANU/runs'
DS = ['baby', 'pantry', 'scientific', 'sports']
SEEDS = [1, 2, 3, 4, 5]
MT = '<RUN_ROOT_B>/P04_mestune/runs_mestune'
# valid-selected main setting per dataset (seed-1 trainer_best_valid recall@20 over l005_k40, l010_k20, l010_k40, l010_k80,
# l020_k40, l030_k40; recorded 2026-10-10 in r20_R1/MANU/STATUS.md): Baby .1045, Pantry .1309, Scientific .1919, Sports .1208
SELECTED = {'baby': 'l030_k40', 'pantry': 'l010_k20', 'scientific': 'l020_k40', 'sports': 'l010_k80'}
SCAN = (f"for f in $(find -L {W}/runs_xen0 {MT} {MANU} -name result.json 2>/dev/null); do "
        "python3 -c \"import json,sys,os; j=json.load(open(sys.argv[1])); b=j.get('best_test') or {}; "
        "print(json.dumps(dict(cfg=j['config'],pkg=j['package'],ds=j['dataset'],seed=j['seed'],status=j.get('status'),"
        "mode=j.get('mode'),te=j.get('test_eval'),r20=b.get('recall@20'),n20=b.get('ndcg@20'),r10=b.get('recall@10'),"
        "n10=b.get('ndcg@10'),run=os.path.dirname(sys.argv[1]))))\" $f; done")


def kinds(r):
    """[(backbone, row), ...] of an eligible MES run (a run may stand for two rows), else []"""
    p, c, ds = r['pkg'], r['cfg'], r['ds'].split('@')[0]
    out = []
    if p == 'xres0' and c == f'{ds}_xres_xeoa0_{SELECTED[ds]}':
        out.append(('MSCA', 'MES'))
    if p == 'xres0' and c.endswith('_xres_xeoa0_l010_k40') and r['run'].startswith(W):
        out.append(('MSCA', 'MES_k40'))
    # weight decay 0.1 (user decision 2026-10-10 11:1x): row MES_wd = the main setting + wd (novctrl dba2ba3, queue meswd2);
    # the earlier lambda 0.1, K 40 + wd runs (queue meswd, novctrl 6bea495) are kept as row MES_wd_k40
    if p == 'xres0' and c == f'{ds}_xres_xeoa0_{SELECTED[ds]}_wd01' and r['run'].startswith(MANU):
        out.append(('MSCA', 'MES_wd'))
    if p == 'xres0' and c.endswith('_xres_xeoa0_l010_k40_wd01') and r['run'].startswith(MANU):
        out.append(('MSCA', 'MES_wd_k40'))
    if p == 'xres0f' and c.endswith('_xres_xeoa0f_l010_k40') and r['run'].startswith(MANU):
        out.append(('MSCA', 'MES_fv'))
    bb = {'mes_c': 'COHESION', 'mes_s': 'SMORE', 'mes_f': 'FREEDOM', 'mes_m': 'MMGCN', 'mes_n': 'MENTOR'}
    if p in bb and c.endswith('_mes_l010_k40') and r['run'].startswith(MANU):
        out.append((bb[p], 'MES'))
    return out


def main():
    partial = '--partial' in sys.argv
    out = subprocess.run(['ssh', '-o', 'BatchMode=yes', 'HOST_A', SCAN], capture_output=True, text=True, check=True).stdout
    cells = {}
    for line in out.splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        ds = r['ds'].split('@')[0]
        if r['status'] != 'SUCCESS' or r['mode'] != 'train' or r['te'] != 'best_primed':
            continue
        for k in kinds(r):
            key = k + (ds, int(r['seed']))
            if key in cells:
                sys.exit(f'ERROR replicate {key}: {cells[key]["run"]} and {r["run"]}')
            cells[key] = r
    groups = sorted({key[:2] for key in cells} | {('MSCA', 'MES')})
    write = []
    _ = partial   # every row is written per complete cell (2026-10-10)
    for b, row in groups:
        dss = DS if b in ('MSCA', 'COHESION', 'SMORE') else DS[:3]
        complete = [d for d in dss if all((b, row, d, s) in cells for s in SEEDS)]
        for d in dss:
            have = sorted(s for (bb, rr, dd, s) in cells if (bb, rr, dd) == (b, row, d))
            print(b, row, d, 'seeds', have, 'complete' if d in complete else 'INCOMPLETE')
        write += [(b, row, d) for d in complete]
    if not write:
        print('nothing complete; nothing written')
        return
    with OUT.open('w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['backbone', 'row', 'dataset', 'seed', 'r20', 'n20', 'r10', 'n10', 'n_runs', 'run_dir'])
        for b, row, d in write:
            for s in SEEDS:
                r = cells[(b, row, d, s)]
                w.writerow([b, row, d, s, r['r20'], r['n20'], r['r10'], r['n10'], 1, r['run']])
    print('wrote', OUT, 'cells:', write)


if __name__ == '__main__':
    main()
