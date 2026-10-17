"""Collect the analysis data of the self-message attenuation (Proposition prop:atten) and of the gradient-direction split
of the exchange term. Only reads remote files, trains nothing.

Usage (local machine, repository root):  python -s tables/src/collect_analysis.py [--selfmsg-only]
  --selfmsg-only (2026-10-10): write tables/analysis_selfmsg.csv only, leave tables/gradsplit_perseed.csv untouched.
Arm 'mes' (2026-10-10): MES = xres0 MAIN-table setting per dataset (Baby baby_xres_xeoa0_l030_k40, Pantry l010_k20,
Scientific l020_k40: P04_mestune/runs_mestune/xres0; Sports l010_k80: s1 P04_mestune, s2-5 run_data/r20/MANU/runs/xres0),
cells <ds>_mes_s<k> written by prop2.py / prop2_nl.py via P04_newprin/prop2/p2_driver_mes.sh with the same training-positive
pairs and negatives as the native cells (tp_u, tp_i, tp_n identical, checked on HOST_A).
The script sends itself over ssh to HOST_A (argument `selfmsg` or `prune`, run by REMOTE_PY, which has numpy) and writes

  tables/analysis_selfmsg.csv   one row per (dataset, arm, seed, history bin); bin 'all' = every pair of the cell.
      Source: <RUN_ROOT_B>/P04_newprin/prop2/out/<ds>_<arm>_s<k>.npz|json (prop2.py, notes 41_prop2_check.md),
      MSCA Native (objx_s) and XRE (xres l010_k40) checkpoints of <RUN_ROOT_B>/P04_listana/runs_listana/, seeds
      1-5 (Baby, Pantry, Scientific). Sports: <RUN_ROOT_B>/P04_newprin/prop2_sports/out/sports_<arm>_s<k>.npz|json,
      written by prop2_sports/prop2_nl.py (= prop2.py generalized to the 3 LightGCN layers of the Sports MSCA config; its
      2-layer path reproduces out/baby_native_s1.npz bitwise) via prop2_sports/p2_driver_sports.sh on the Table I Native
      (objx_s sports_native) and XRE (xres sports_xres_xre_l010_k40) runs of <RUN_ROOT_B>/P04_sports/runs_xres/,
      seeds 1-5 (repo P04_xres_wt 406a7a7, the listana commit). Each cell holds the same training-positive sample (rng 0, eval users of the test split, at most five training
      positives per user) and one uniform negative n per pair outside the training history.
      Per pair: S = self-message R~_up m_up^T e_p, c = explicit cavity score over H_u \\ {p}, s_up = model.forward() score,
      s_un = score of the negative, du = training history length of u.
      Columns: n_pairs, n_pos (pairs with s_up > 0, the denominator of the share), share_mean = mean of S / s_up over the
      pairs with s_up > 0, share_median, share_rom = sum S / sum s_up, S_mean, Ewf = E[sigma(s_un - s_up)] (BPR positive
      weight, full score), Ewc = E[sigma(s_un - c)] (same weight with the self-message removed), wratio = Ewf / Ewc,
      rho_mean = E[Ewf_pair / Ewc_pair], repro_ok and test_r20 (the restored checkpoint reproduces result.json to 4 dp).
      History bins (du, training interactions of the user): <=3, 4-7, 8-15, >=16.

  tables/gradsplit_perseed.csv  one row per (dataset, arm, seed): test Recall@20 / NDCG@20 of the best-valid checkpoint.
      Source: <RUN_ROOT_B>/P04_prune/results_prune.json (collect_prune.py of that folder, notes 31_prune.md),
      Pantry and Scientific, seeds 1-5. Arms e3 (full-view exchange + anchor, E3), pos (entry gradient raises the
      held-out positive only), disp (entry gradient lowers the displaced item only), bdrand (disp with a random list
      item instead of the displaced item), bdnoanc (disp without the anchor term). Native, A1 and XRE rows are copied
      from the manuscript tables (tables/table1_perseed.csv row MSCA Native, tables/xre_perseed.csv rows MSCA A1 / XRE);
      the collector stops if results_prune.json disagrees with them on any seed (4 dp), so both sources are the same runs.
"""
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] if Path(__file__).exists() else Path('.')   # remote side: stdin
TABLES = ROOT / 'tables'
REMOTE_PY = 'python'
SSH = ['ssh', '-o', 'BatchMode=yes', 'HOST_A']
P2 = '<RUN_ROOT_B>/P04_newprin/prop2/out'
P2_DS = {'sports': '<RUN_ROOT_B>/P04_newprin/prop2_sports/out'}   # other datasets: P2
PRUNE = '<RUN_ROOT_B>/P04_prune/results_prune.json'
DSS = ['baby', 'pantry', 'scientific', 'sports']
ARMS = ['native', 'xre', 'mes']   # 'mes' added 2026-10-10 (MAIN-table MES runs; absent cells are skipped)
SEEDS = [1, 2, 3, 4, 5]
BINS = [('le3', 0, 3), ('4to7', 4, 7), ('8to15', 8, 15), ('ge16', 16, 10 ** 9)]
PRUNE_DS = ['pantry', 'scientific']
PRUNE_ARMS = ['e3', 'pos', 'disp', 'bdrand', 'bdnoanc']
SEL_COLS = ['dataset', 'arm', 'seed', 'bin', 'n_pairs', 'n_pos', 'share_mean', 'share_median', 'share_rom', 'S_mean', 'Ewf',
            'Ewc', 'wratio', 'rho_mean', 'repro_ok', 'test_r20']


def remote_selfmsg():
    import numpy as np
    w = csv.writer(sys.stdout, lineterminator='\n')
    w.writerow(SEL_COLS)
    for ds in DSS:
        for arm in ARMS:
            for s in SEEDS:
                t = f'{ds}_{arm}_s{s}'
                p = Path(P2_DS.get(ds, P2)) / f'{t}.npz'
                if not p.exists():
                    sys.stderr.write(f'absent {p}\n')
                    continue
                z = np.load(p)
                meta = json.loads((Path(P2_DS.get(ds, P2)) / f'{t}.json').read_text())
                S, c, sup, sun, du = z['S'], z['c'], z['s_up'], z['s_un'], z['du']
                wf = 1.0 / (1.0 + np.exp(-(sun - sup)))
                wc = 1.0 / (1.0 + np.exp(-(sun - c)))
                for b, lo, hi in [('all', 0, 10 ** 9)] + BINS:
                    m = (du >= lo) & (du <= hi)
                    mp = m & (sup > 0)
                    sh = S[mp] / sup[mp]
                    w.writerow([ds, arm, s, b, int(m.sum()), int(mp.sum()), f'{sh.mean():.6f}', f'{np.median(sh):.6f}',
                                f'{S[m].sum() / sup[m].sum():.6f}', f'{S[m].mean():.6f}', f'{wf[m].mean():.6f}',
                                f'{wc[m].mean():.6f}', f'{wf[m].mean() / wc[m].mean():.6f}', f'{(wf[m] / wc[m]).mean():.6f}',
                                int(bool(meta['repro_ok'])), meta['reeval_test_recall20']])


def remote_prune():
    d = json.load(open(PRUNE))
    w = csv.writer(sys.stdout, lineterminator='\n')
    w.writerow(['dataset', 'arm', 'seed', 'r20', 'n20', 'path'])
    for ds in PRUNE_DS:
        for arm, cells in d[ds]['cells'].items():
            for s, v in sorted(cells.items()):
                w.writerow([ds, arm, s, v['r20'], v['n20'], v['path']])


def fetch(mode):
    src = open(__file__, encoding='utf-8').read()
    out = subprocess.run(SSH + [REMOTE_PY, '-', mode], input=src.encode('utf-8'), capture_output=True, check=True)
    if out.stderr:
        sys.stderr.write(out.stderr.decode('utf-8', 'replace'))
    return list(csv.DictReader(out.stdout.decode('utf-8').splitlines()))


def manuscript_rows():
    """Native / A1 / XRE per seed from the manuscript tables (the values of Table I and the ablation figure)."""
    out = {}
    with open(TABLES / 'table1_perseed.csv', newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['backbone'] == 'MSCA' and r['row'] == 'Native' and r['dataset'] in PRUNE_DS:
                out[(r['dataset'], 'Native', int(r['seed']))] = (float(r['r20']), float(r['n20']), 'tables/table1_perseed.csv')
    with open(TABLES / 'xre_perseed.csv', newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['backbone'] == 'MSCA' and r['row'] in ('A1', 'XRE') and r['dataset'] in PRUNE_DS:
                out[(r['dataset'], r['row'], int(r['seed']))] = (float(r['r20']), float(r['n20']), 'tables/xre_perseed.csv')
    return out


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'selfmsg':
        return remote_selfmsg()
    if len(sys.argv) > 1 and sys.argv[1] == 'prune':
        return remote_prune()
    rows = fetch('selfmsg')
    bad = [r for r in rows if r['repro_ok'] != '1']
    if bad:
        sys.exit(f'restore check failed: {bad}')
    n_cells = len({(r['dataset'], r['arm'], r['seed']) for r in rows})
    if n_cells != len(DSS) * len(ARMS) * len(SEEDS):
        sys.stderr.write(f'warning: {n_cells} self-message cells\n')
    with open(TABLES / 'analysis_selfmsg.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=SEL_COLS, lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    print('wrote tables/analysis_selfmsg.csv', len(rows), 'rows')
    if '--selfmsg-only' in sys.argv:
        return

    pr = fetch('prune')
    ms = manuscript_rows()
    remote = {(r['dataset'], r['arm'], int(r['seed'])): r for r in pr}
    for (ds, arm, s), (r20, n20, _) in ms.items():
        r = remote.get((ds, arm, s))
        if r is None or round(float(r['r20']), 4) != round(r20, 4) or round(float(r['n20']), 4) != round(n20, 4):
            sys.exit(f'results_prune.json disagrees with the manuscript tables at {ds} {arm} s{s}: {r} vs {r20} {n20}')
    out = []
    for ds in PRUNE_DS:
        for arm in ['Native', 'A1', 'XRE']:
            for s in SEEDS:
                if (ds, arm, s) in ms:
                    r20, n20, src = ms[(ds, arm, s)]
                    out.append(dict(dataset=ds, arm=arm, seed=s, r20=f'{r20:.4f}', n20=f'{n20:.4f}', source=src))
        for arm in PRUNE_ARMS:
            for s in SEEDS:
                r = remote.get((ds, arm, s))
                if r:
                    out.append(dict(dataset=ds, arm=arm, seed=s, r20=f"{float(r['r20']):.4f}", n20=f"{float(r['n20']):.4f}",
                                    source=r['path']))
    with open(TABLES / 'gradsplit_perseed.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['dataset', 'arm', 'seed', 'r20', 'n20', 'source'], lineterminator='\n')
        w.writeheader()
        w.writerows(out)
    print('wrote tables/gradsplit_perseed.csv', len(out), 'rows')


if __name__ == '__main__':
    main()
