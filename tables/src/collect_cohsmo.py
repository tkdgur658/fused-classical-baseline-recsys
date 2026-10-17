"""Collect the Table I-style blocks of COHESION and SMORE (XRE, the five plug-ins of Table I, Native). Reads remote files only.

Sources (HOST_A):
  plug-ins  <RUN_ROOT_B>/P04_cohsmo/runs_cohsmo/{plg2_c,plg2_sm}/  (repo P04_cohsmo_wt, branch cohsmo 77bb243; MG a*
            per backbone-dataset from <RUN_ROOT_B>/P04_cohsmo/mg_selection.json, seed-1 validation rule of Table I)
  XRE       <RUN_ROOT_B>/P04_main_runs/qtransX3/E/{xret_c,xret_s}/   (configs <ds>_xre_l010_k40; RunPod GPU)
  Native    tables/table2_runs.tsv (Baby / Pantry / Scientific) and tables/table2_runs_extra.tsv (Sports), chosen
            COHESION / SMORE rows = the Table II runs, one run set per Native value (2026-10-06 review P0-1); the Native
            runs inside the Sports blocks (HOST_Asp, HOST_B) are not read
  Sports    one machine per backbone block (branch cohsmo-sports 52fddc6; XRE, plug-ins, MG grid and Native of that block):
            COHESION: HOST_A, <RUN_ROOT_B>/P04_cohsmo_spA/runs_cohsp/ (moved from HOST_B on 2026-10-06;
                      MG a* from <RUN_ROOT_B>/P04_cohsmo_spA/mg_selection.json)
            SMORE:    HOST_B, <RUN_ROOT_B>/P04_cohsmo/runs_sports_HOST_B/ (synced without checkpoints;
                      MG a* from HOST_B mg_selection.json); COHESION runs of HOST_B are not read
Output: tables/cohsmo_perseed.csv (backbone, row, dataset, seed, r20, n20, source) and tables/cohsmo_mg_astar.tsv
        (the chosen MG alpha2 per block and dataset, write_astar(); --astar-only writes only this file)
Usage (repository root): python -s tables/src/collect_cohsmo.py
"""
import csv
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / 'tables'
OUT = T / 'cohsmo_perseed.csv'
BB = {'plg2_c': 'COHESION', 'plg2_sm': 'SMORE', 'xret_c': 'COHESION', 'xret_s': 'SMORE'}
ROWNAME = {'ina_cooc_k10_d04_l06': 'INA', 'mdvt_n2_l02_g01': 'MDVT', 'damrs_a1b2_k10_x2': 'DA-MRS',
           'ugcn_li_k10_l0007': 'UltraGCN-LI', 'xre_l010_k40': 'XRE', 'native': 'Native'}
DIRS = {'HOST_A': '<RUN_ROOT_B>/P04_cohsmo/runs_cohsmo <RUN_ROOT_B>/P04_main_runs/qtransX3/E',
        'HOST_B': '<RUN_ROOT_B>/P04_cohsmo/runs_sports_HOST_B',
        'HOST_Asp': '<RUN_ROOT_B>/P04_cohsmo_spA/runs_cohsp'}


def remote(cmd):
    return subprocess.run(['ssh', '-o', 'BatchMode=yes', 'HOST_A', cmd], capture_output=True, text=True, check=True).stdout


def main():
    sel = json.loads(remote('cat <RUN_ROOT_B>/P04_cohsmo/mg_selection.json'))
    astar = {(k.split('/')[0], k.split('/')[1]): v['a_star'] for k, v in sel.items()}
    rows = []
    for src, dirs in DIRS.items():
        cmd = (f"for d in {dirs}; do [ -d $d ] && find $d -name result.json; done | while read f; do "
               "python3 -c \"import json,sys; j=json.load(open(sys.argv[1])); b=j.get('best_test',{}); "
               "print(json.dumps(dict(pkg=j['package'],cfg=j['config'],ds=j['dataset'],seed=j['seed'],st=j.get('status'),"
               "r20=b.get('recall@20'),n20=b.get('ndcg@20'))))\" $f; done")
        for line in remote(cmd).splitlines():
            j = json.loads(line)
            if j['pkg'] not in BB or j['r20'] is None:
                continue
            if src == 'HOST_B' and j['pkg'] not in ('plg2_sm', 'xret_s'):      # COHESION Sports runs on HOST_A only
                continue
            if src == 'HOST_Asp' and j['pkg'] not in ('plg2_c', 'xret_c'):
                continue
            tail = j['cfg'].split(j['ds'] + '_', 1)[1]
            if tail.startswith('mg_'):
                key = (j['pkg'] if j['pkg'].startswith('plg') else None, j['ds'])
                a = astar.get(key) if src == 'HOST_A' else None
                if src == 'HOST_B':                       # Sports MG selection of HOST_B (select_mg_sports74.py)
                    a = SPORTS_ASTAR.get(j['pkg'])
                if src == 'HOST_Asp':
                    a = SPA_ASTAR.get(j['pkg'])
                if tail != f'mg_{a}':
                    continue
                row = 'MG'
            elif tail in ROWNAME:
                row = ROWNAME[tail]
                if row == 'Native' and j['ds'] == 'sports':     # Sports Native = Table II runs (below)
                    continue
            else:
                continue
            if src == 'HOST_A' and j['ds'] == 'sports':
                continue
            rows.append(dict(backbone=BB[j['pkg']], row=row, dataset=j['ds'], seed=int(j['seed']), r20=j['r20'], n20=j['n20'],
                             source=src))
    for r in csv.DictReader(open(T / 'table2_runs.tsv', encoding='utf-8'), delimiter='\t'):
        if r['chosen'] == '1' and r['model'] in ('COHESION', 'SMORE') and r['dataset'] in ('baby', 'pantry', 'scientific'):
            rows.append(dict(backbone=r['model'], row='Native', dataset=r['dataset'], seed=int(r['seed']),
                             r20=float(r['test_r20']), n20=float(r['test_n20']), source='table2'))
    for r in csv.DictReader(open(T / 'table2_runs_extra.tsv', encoding='utf-8'), delimiter='\t'):
        if r['chosen'] == '1' and r['model'] in ('COHESION', 'SMORE') and r['dataset'] == 'sports':
            rows.append(dict(backbone=r['model'], row='Native', dataset='sports', seed=int(r['seed']),
                             r20=float(r['test_r20']), n20=float(r['test_n20']), source='table2_extra'))
    keys = [(r['backbone'], r['row'], r['dataset'], r['seed']) for r in rows]
    assert len(keys) == len(set(keys)), 'duplicate cells'
    rows.sort(key=lambda r: (r['backbone'], r['row'], r['dataset'], r['seed']))
    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    print('wrote', OUT, len(rows))


SPORTS_ASTAR = {}
SPA_ASTAR = {}
try:
    _s = json.loads(subprocess.run(['ssh', '-o', 'BatchMode=yes', 'HOST_B',
                                    'cat <RUN_ROOT_C>/P04_cohsmo_sports/mg_selection.json 2>/dev/null || echo {}'],
                                   capture_output=True, text=True, timeout=60).stdout or '{}')
    SPORTS_ASTAR = {k.split('/')[0]: v['a_star'] for k, v in _s.items() if 'sports' in k}
except Exception:
    pass
try:
    _s = json.loads(subprocess.run(['ssh', '-o', 'BatchMode=yes', 'HOST_A',
                                    'cat <RUN_ROOT_B>/P04_cohsmo_spA/mg_selection.json 2>/dev/null || echo {}'],
                                   capture_output=True, text=True, timeout=60).stdout or '{}')
    SPA_ASTAR = {k.split('/')[0]: v['a_star'] for k, v in _s.items()}
except Exception:
    pass

def write_astar():
    """tables/cohsmo_mg_astar.tsv: the MG coefficient alpha2 chosen per COHESION / SMORE block and dataset (the a* of the
    MG selection files read by main(), and alpha2 read from the selected YAML of the package worktree on HOST_A:
    P04_cohsmo_wt 77bb243 for Baby / Pantry / Scientific, P04_cohsmo_sports_wt 52fddc6 for Sports, the branch run on
    HOST_B as well). Read by tables/src/build_table_hparams.py. Usage: python -s tables/src/collect_cohsmo.py --astar-only"""
    sel = json.loads(remote('cat <RUN_ROOT_B>/P04_cohsmo/mg_selection.json'))
    pick = {(k.split('/')[0], k.split('/')[1]): v['a_star'] for k, v in sel.items()}
    for pkg, a in SPORTS_ASTAR.items():          # SMORE Sports block of HOST_B
        if pkg == 'plg2_sm':
            pick[(pkg, 'sports')] = a
    for pkg, a in SPA_ASTAR.items():           # COHESION Sports block of HOST_A
        if pkg == 'plg2_c':
            pick[(pkg, 'sports')] = a
    rows = []
    for (pkg, ds), a in sorted(pick.items()):
        wt = '<HOME>/P04_cohsmo_sports_wt' if ds == 'sports' else '<HOME>/P04_cohsmo_wt'
        y = f'{wt}/models/{pkg}/configs/{ds}_mg_{a}.yaml'
        val = remote(f"grep '^alpha2:' {y}").split(':', 1)[1].strip()
        rows.append(dict(backbone=BB[pkg], dataset=ds, a_star=a, alpha2=val, yaml=y))
    with open(T / 'cohsmo_mg_astar.tsv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter='\t', lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    print('wrote', T / 'cohsmo_mg_astar.tsv', len(rows))


if __name__ == '__main__':
    import sys
    if '--astar-only' in sys.argv:
        write_astar()
    else:
        main()
        write_astar()
