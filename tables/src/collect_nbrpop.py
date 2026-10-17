"""Neighbor x popularity cross of the held-out test targets (internal review 2026-10-07, Table 6 proposal 4): top-20 hit
rate of MSCA and of a compared model in a 2 x 5 grid, rows = co-occurrence neighbor of the training history or not,
columns = popularity quintile Q1..Q5 of the target training degree. Reads existing per-target cells, trains nothing.

Source cells (one run per seed, seeds 1-5, inference done once by P04_hyp_strat/hyp_strat.py, CPU only; see
tables/src/collect_hyp_strat.py for the checkpoints and the reproduction check that every cell passed):
  HOST_A:<RUN_ROOT_B>/P04_hyp_strat/cells/<ds>_<arm>_s<k>.npz|json
  arm 'native' = MSCA (objx_s <ds>_native); arm 'xre' = the earlier design E + R + A (old XRE, xres
  <ds>_xres_xre_l010_k40). Per target: P_u, P_i, P_rank (evaluator rank, training items excluded), P_deg (training degree
  of the target), P_link_struct (co-occurrence neighbor of >= 1 training-history item of its user in the MSCA
  co-occurrence graph), P_link_any (same with the union of co-occurrence, image kNN and text kNN graphs).
Quintile rule = the rule of tables/hyp_strat_pop.csv (fig_h_strat_pop): targets sorted stably by (degree, cell order) and
cut into five equal parts (q = floor(5 * position / n)), computed over all targets of the seed, then crossed with the
neighbor flag. With graph 'cooc' the numbers equal the 'structpop' rows of tables/hyp_strat_pop.csv (checked here).

Output tables/hyp_nbrpop.csv, one row per (dataset, seed, graph, link, quintile):
  model_a = MSCA, model_b = label of the compared model, n_targets, deg_lo, deg_hi, hit20_a, hit20_b (share of the targets
  of the cell ranked <= 20), gain_pp = 100 * (hit20_b - hit20_a) on the same seed.

Usage (repository root):
  python -s tables/src/collect_nbrpop.py              # arm xre (E + R + A), ssh to HOST_A (CPU, nice 15)
  python -s tables/src/collect_nbrpop.py --arm mes    # once cells <ds>_mes_s<k> exist (MES checkpoints run through
                                                      # P04_hyp_strat/hyp_strat.py); label MES
  python -s tables/src/collect_nbrpop.py --from-csv   # offline: derive the cooc rows from tables/hyp_strat_pop.csv
  python -s tables/src/collect_nbrpop.py --arm mes --partial   # skip a dataset whose five <ds>_mes_s<k> cells are not
                                                      # all present yet (2026-10-10: Sports MES seeds 2-5 in training)
MES cells (2026-10-10): P04_hyp_strat/hyp_strat_mes.py (= hyp_strat.py with arm 'mes', repo P04_novctrl_wt) via
P04_hyp_strat/run_mes.sh on the MAIN-table MES runs (xres0; Baby l030_k40, Pantry l010_k20, Scientific l020_k40, Sports
l010_k80; P04_mestune/runs_mestune/xres0 and run_data/r20/MANU/runs/xres0); every cell reproduced result.json test
Recall@20 / NDCG@20 to 4 dp (repro_ok and recomputed_ok, checked again below).
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
CELLS = '<RUN_ROOT_B>/P04_hyp_strat/cells'
DSS = ['baby', 'pantry', 'scientific', 'sports']
SEEDS = [1, 2, 3, 4, 5]
NQ = 5
LABEL = {'xre': 'E + R + A', 'mes': 'MES'}
COLS = ['dataset', 'seed', 'model_a', 'model_b', 'graph', 'link', 'quintile', 'n_targets', 'deg_lo', 'deg_hi',
        'hit20_a', 'hit20_b', 'gain_pp']


def remote(arm, partial=False):
    import numpy as np
    out = []
    for ds in DSS:
        if partial and not all((Path(CELLS) / f'{ds}_{arm}_s{s}.npz').exists() for s in SEEDS):
            sys.stderr.write(f'partial: {ds} skipped (not all five {arm} cells exist)\n')
            continue
        for s in SEEDS:
            z = {}
            for a in ('native', arm):
                p = Path(CELLS) / f'{ds}_{a}_s{s}'
                m = json.loads(p.with_suffix('.json').read_text())
                if not (m['repro_ok'] and m['recomputed_ok']):
                    sys.exit(f'{p}: reproduction check failed')
                z[a] = dict(np.load(str(p) + '.npz'))
            A, B = z['native'], z[arm]
            for k in ('P_u', 'P_i', 'P_deg', 'P_link_struct', 'P_link_any'):
                if not np.array_equal(A[k], B[k]):
                    sys.exit(f'{ds} s{s}: {k} differs between arms')
            n = len(A['P_rank'])
            order = np.argsort(A['P_deg'], kind='stable')
            qid = np.empty(n, np.int64)
            qid[order] = (np.arange(n) * NQ) // n
            ha, hb = A['P_rank'] <= 20, B['P_rank'] <= 20
            for graph, key in (('cooc', 'P_link_struct'), ('union', 'P_link_any')):
                for link in ('nbr', 'non'):
                    lm = A[key] if link == 'nbr' else ~A[key]
                    for q in range(NQ):
                        msk = lm & (qid == q)
                        if not msk.any():
                            continue
                        dg = A['P_deg'][msk]
                        out.append([ds, s, graph, link, f'q{q + 1}', int(msk.sum()), int(dg.min()), int(dg.max()),
                                    float(ha[msk].mean()), float(hb[msk].mean())])
    json.dump(out, sys.stdout)


def from_csv():
    """cooc rows from the structpop grouping of tables/hyp_strat_pop.csv (arm xre only)"""
    v = {}
    for r in csv.DictReader(open(TABLES / 'hyp_strat_pop.csv', encoding='utf-8')):
        if r['grouping'] == 'structpop':
            v[(r['dataset'], int(r['seed']), r['arm'], r['group'])] = r
    out = []
    for ds in DSS:
        for s in SEEDS:
            for link in ('nbr', 'non'):
                for q in range(1, NQ + 1):
                    a, b = v[(ds, s, 'native', f'{link}_q{q}')], v[(ds, s, 'xre', f'{link}_q{q}')]
                    assert a['n_targets'] == b['n_targets']
                    out.append([ds, s, 'cooc', link, f'q{q}', int(a['n_targets']), int(a['deg_lo']), int(a['deg_hi']),
                                float(a['hit20']), float(b['hit20'])])
    return out


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'remote':
        return remote(sys.argv[2], len(sys.argv) > 3 and sys.argv[3] == 'partial')
    arm = sys.argv[sys.argv.index('--arm') + 1] if '--arm' in sys.argv else 'xre'
    if '--from-csv' in sys.argv:
        rows = from_csv()
    else:
        src = open(__file__, encoding='utf-8').read()
        res = subprocess.run(SSH + [f'CUDA_VISIBLE_DEVICES= nice -n 15 {REMOTE_PY} - remote {arm}'
                                    + (' partial' if '--partial' in sys.argv else '')],
                             input=src.encode('utf-8'), capture_output=True)
        if res.returncode:
            sys.exit(res.stderr.decode('utf-8', 'replace'))
        if res.stderr:
            sys.stderr.write(res.stderr.decode('utf-8', 'replace'))
        rows = json.loads(res.stdout.decode('utf-8'))
        if arm == 'xre':   # the cooc rows must equal the structpop rows of hyp_strat_pop.csv (6 dp)
            ref = {tuple(r[:5]): r for r in from_csv()}
            k = 0
            for r in rows:
                if r[2] == 'cooc':
                    c = ref[tuple(r[:5])]
                    assert r[5:8] == c[5:8], (r, c)
                    assert abs(r[8] - c[8]) < 1e-6 and abs(r[9] - c[9]) < 1e-6, (r, c)
                    k += 1
            assert k == len(ref), (k, len(ref))
            print('checked', k, 'cooc cells against tables/hyp_strat_pop.csv (structpop)')
    with open(TABLES / 'hyp_nbrpop.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(COLS)
        for ds, s, graph, link, q, n, lo, hi, a, b in rows:
            w.writerow([ds, s, 'MSCA', LABEL[arm], graph, link, q, n, lo, hi, f'{a:.6f}', f'{b:.6f}',
                        f'{100 * (b - a):.4f}'])
    print('wrote', TABLES / 'hyp_nbrpop.csv', len(rows), 'rows')


if __name__ == '__main__':
    main()
