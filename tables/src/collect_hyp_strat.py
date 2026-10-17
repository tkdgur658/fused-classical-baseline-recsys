"""Collect the stratified hypothesis tests of MSCA versus MSCA + XRE (history length, target group, rank shift) from
per-user / per-target checkpoint statistics on HOST_A, and write their CSV files and macros. Inference was done once by the
remote tool, this collector only reads its cells and trains nothing.

Remote tool: <RUN_ROOT_B>/P04_hyp_strat/hyp_strat.py (driver run_all.sh; CPU only, nice 10). It restores the
best-valid checkpoint with load_cell of <RUN_ROOT_B>/P04_listana/tools/list_stats_full.py (the loader of the
crowding and self-message analyses) and stops unless the re-evaluated and the rank-recomputed test Recall@20 / NDCG@20
equal result.json best_test to 4 dp. Cells <RUN_ROOT_B>/P04_hyp_strat/cells/<ds>_<arm>_s<k>.npz|json.
Checkpoints (one run per seed, seeds 1-5, the same runs as tables/recallk_listana.csv and tables/analysis_selfmsg.csv):
  Baby, Pantry, Scientific  <RUN_ROOT_B>/P04_listana/runs_listana/{objx_s,xres}/ (P04_listana reruns of the
                            Table I configuration, repo P04_xres_wt 406a7a7)
  Sports                    <RUN_ROOT_B>/P04_sports/runs_xres/{objx_s,xres}/ (the Table I runs of Sports)
  arm native = objx_s <ds>_native (MSCA), arm xre = xres <ds>_xres_xre_l010_k40 (MSCA + XRE, lambda 0.1, K 40).
Ranking = evaluator: training items excluded, validation items stay candidates, rank = 1 + #{candidates scoring higher}.

Usage (local machine, repository root):  python -s tables/src/collect_hyp_strat.py
The script sends itself over ssh to HOST_A (argument `remote`, run by REMOTE_PY, which has numpy) and writes

  tables/hyp_strat_hist.csv   one row per (dataset, arm, seed, bin): n_users, recall20, ndcg20 = mean over the test users
      whose training history length d_u falls in the bin (<=3, 4-7, 8-15, >=16, the bins of fig:selfmsg (a)); bin 'all'.
  tables/hyp_strat_pop.csv    one row per (dataset, arm, seed, grouping, group): n_targets, hit20 = share of the held-out
      test targets (user, item) ranked inside the top 20, mrr = mean 1 / rank. Groupings:
        link   'nbr' = the target is adjacent to >= 1 training-history item of its user in the union of the MSCA
               item-item graphs (co-occurrence, image kNN, text kNN; link_any of fig:crowding (a)), 'non' = otherwise
        struct same with the co-occurrence graph only
        pop    quintiles q1 (least popular) .. q5 of the training degree of the targets: targets sorted stably by
               (degree, order of the cells) and cut into five equal parts; deg_lo / deg_hi = degree range of the part
        structpop  'nbr_q<k>' / 'non_q<k>' = co-occurrence neighbors / non-neighbors inside popularity quintile k
  tables/hyp_rankshift.csv    one row per (dataset, seed, pair, nat_bin, xre_bin): n_targets with MSCA rank (seed s) in
      nat_bin and rank in xre_bin under the second model, rank bins 1-20, 21-40, 41-60, 61-100, 101-200, 201-500, >500;
      pair 'xre' = MSCA + XRE of seed s, pair 'ctrl' = MSCA of seed s mod 5 + 1 (control: two runs of one method);
      n_all = test targets of the seed.
  tables/hyp_rankshift_trans.csv  one row per (dataset, seed, pair, msca_bin, other_bin), coarse bins top = rank 1-20 (the
      evaluation cutoff of Recall@20), near = 21-40 (just below the cutoff; 40 is also the training list size K of XRE),
      deep = >40. n_targets = test targets with the MSCA rank (seed s) in msca_bin and the rank under the second model in
      other_bin; recall_mass = sum over these targets of 1 / (n_test(u) * n_users), i.e. their share of Recall@20, so the
      recall_mass of a cell is what the cell adds to the per-user averaged Recall@20; recall_a / recall_b = test Recall@20 of
      MSCA (seed s) / the second model from the cells (checked against sum of recall_mass over the top column).
  tables/macros_hyp_strathist.tex, macros_hyp_stratpop.tex, macros_hyp_rankshift.tex  (macro names start with hz)
Rank shift definitions (fixed 2026-10-07; the first version binned entering targets by the MSCA rank but leaving targets by
the rank under MSCA + XRE and subtracted the two per bin, which is not a flow of any one cohort, see
<TOOLS>/hyp_figs_20261007/NOTES_fix_rankshift_forest.md):
  recovered hits from bin b = targets with MSCA rank in b (b = near, deep) and rank <= 20 under the second model
  lost hits into bin b      = targets with MSCA rank <= 20 and rank in b (b = near, deep) under the second model
  net gain = recovered - lost = (hit rate of the second model - hit rate of MSCA) at cutoff 20, per-target counting
  entry rate of cohort b = recovered from b / targets with MSCA rank in b
Usage for the rank shift only (does not touch the history / popularity files):  python -s tables/src/collect_hyp_strat.py rank
Paired seed statistics: gain_s = XRE - Native on seed s (same users / targets in both arms of a seed), mean, min, max,
number of positive seeds, two-sided paired t-test over the five seeds (scipy.stats.ttest_rel).
"""
import csv
import io
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
DSN = {'baby': 'Baby', 'pantry': 'Pantry', 'scientific': 'Scientific', 'sports': 'Sports'}
ARMS = ['native', 'xre']
SEEDS = [1, 2, 3, 4, 5]
BINS = [('le3', 0, 3), ('4to7', 4, 7), ('8to15', 8, 15), ('ge16', 16, 10 ** 9)]
POOLED = [('le7', 0, 7), ('ge8', 8, 10 ** 9)]   # short / long histories, the two halves of BINS
RBINS = [('1-20', 1, 20), ('21-40', 21, 40), ('41-60', 41, 60), ('61-100', 61, 100), ('101-200', 101, 200),
         ('201-500', 201, 500), ('>500', 501, 10 ** 9)]
NQ = 5
H_COLS = ['dataset', 'arm', 'seed', 'bin', 'n_users', 'recall20', 'ndcg20']
P_COLS = ['dataset', 'arm', 'seed', 'grouping', 'group', 'n_targets', 'deg_lo', 'deg_hi', 'hit20', 'mrr']
R_COLS = ['dataset', 'seed', 'pair', 'nat_bin', 'xre_bin', 'n_targets', 'n_all']
CB = [('top', 1, 20), ('near', 21, 40), ('deep', 41, 10 ** 9)]   # coarse bins: cutoff 20, list size K = 40 of XRE
T_COLS = ['dataset', 'seed', 'pair', 'msca_bin', 'other_bin', 'n_targets', 'n_all', 'recall_mass', 'recall_a', 'recall_b']


# ------------------------------------------------------------------------------------------------------------ remote
def remote(only_rank=False):
    import numpy as np
    out = {'hist': [], 'pop': [], 'rank': [], 'trans': []}
    for ds in DSS:
        for s in SEEDS:
            z, m = {}, {}
            for arm in ARMS:
                p = Path(CELLS) / f'{ds}_{arm}_s{s}'
                m[arm] = json.loads(p.with_suffix('.json').read_text())
                if not (m[arm]['repro_ok'] and m[arm]['recomputed_ok']):
                    sys.exit(f'{p}: reproduction check failed')
                z[arm] = dict(np.load(str(p) + '.npz'))
            a, b = z['native'], z['xre']
            for k in ('users', 'n_hist', 'n_test', 'P_u', 'P_i', 'P_deg', 'P_link_struct', 'P_link_any', 'item_deg'):
                if not np.array_equal(a[k], b[k]):
                    sys.exit(f'{ds} s{s}: {k} differs between arms')
            for arm in ARMS:
                d = z[arm]
                if round(float(d['recall20'].mean()), 4) != round(float(m[arm]['result_json_test_recall20']), 4):
                    sys.exit(f'{ds} {arm} s{s}: recall20')
                if only_rank:
                    continue
                for bn, lo, hi in [('all', 0, 10 ** 9)] + BINS + POOLED:
                    u = (d['n_hist'] >= lo) & (d['n_hist'] <= hi)
                    out['hist'].append([ds, arm, s, bn, int(u.sum()), f"{d['recall20'][u].mean():.6f}",
                                        f"{d['ndcg20'][u].mean():.6f}"])
                hit = d['P_rank'] <= 20
                rr = 1.0 / d['P_rank']
                n = len(hit)
                groups = [('all', 'all', np.ones(n, bool)),
                          ('link', 'nbr', d['P_link_any']), ('link', 'non', ~d['P_link_any']),
                          ('struct', 'nbr', d['P_link_struct']), ('struct', 'non', ~d['P_link_struct'])]
                order = np.argsort(d['P_deg'], kind='stable')
                qid = np.empty(n, np.int64)
                qid[order] = (np.arange(n) * NQ) // n
                groups += [('pop', f'q{q + 1}', qid == q) for q in range(NQ)]
                groups += [('structpop', f'{t}_q{q + 1}', (qid == q) & (d['P_link_struct'] == (t == 'nbr')))
                           for t in ('nbr', 'non') for q in range(NQ)]
                for gname, g, msk in groups:
                    if not msk.any():
                        continue
                    dg = d['P_deg'][msk]
                    out['pop'].append([ds, arm, s, gname, g, int(msk.sum()), int(dg.min()), int(dg.max()),
                                       f'{hit[msk].mean():.6f}', f'{rr[msk].mean():.6f}'])
            # control: MSCA seed s against MSCA seed s' = s mod 5 + 1 (two different runs of the same method)
            s2 = s % len(SEEDS) + 1
            c2 = dict(np.load(f'{CELLS}/{ds}_native_s{s2}.npz'))
            if not (np.array_equal(c2['P_i'], a['P_i']) and np.array_equal(c2['P_u'], a['P_u'])):
                sys.exit(f'{ds}: targets differ between seeds {s} and {s2}')
            # weight of a target in Recall@20 = 1 / (n_test of its user * number of test users)
            w = 1.0 / (a['n_test'][a['P_u']] * len(a['users']))
            for pair, ra, rb, recb in (('xre', a['P_rank'], b['P_rank'], b['recall20']),
                                       ('ctrl', a['P_rank'], c2['P_rank'], c2['recall20'])):
                for nb, nlo, nhi in RBINS:
                    for xb, xlo, xhi in RBINS:
                        c = int(((ra >= nlo) & (ra <= nhi) & (rb >= xlo) & (rb <= xhi)).sum())
                        out['rank'].append([ds, s, pair, nb, xb, c, len(ra)])
                rec_a, rec_b = float(a['recall20'].mean()), float(recb.mean())
                for r_, rr in ((ra, rec_a), (rb, rec_b)):   # Recall@20 = sum of the weights of the hits
                    if abs(float(w[r_ <= 20].sum()) - rr) > 1e-9:
                        sys.exit(f'{ds} s{s} {pair}: recall mass {w[r_ <= 20].sum()} != Recall@20 {rr}')
                for nb, nlo, nhi in CB:
                    for xb, xlo, xhi in CB:
                        msk = (ra >= nlo) & (ra <= nhi) & (rb >= xlo) & (rb <= xhi)
                        out['trans'].append([ds, s, pair, nb, xb, int(msk.sum()), len(ra), f'{w[msk].sum():.9f}',
                                             f'{rec_a:.9f}', f'{rec_b:.9f}'])
    json.dump(out, sys.stdout)


# ------------------------------------------------------------------------------------------------------------ local
def write_macros(name, L, ranges=()):
    """write a macro file; a value that rounds to zero is printed without a minus sign. For each stem in `ranges`,
    <stem>Min / <stem>Max = smallest / largest of the four dataset macros <stem><Dataset> (as printed)."""
    vals = {}
    for x in L:
        if x.startswith('\\newcommand{\\'):
            k, val = x[len('\\newcommand{\\'):-1].split('}{')
            vals[k] = val
    for stem in ranges:
        xs = [vals[stem + DSN[ds]] for ds in DSS]
        L = L + [f'\\newcommand{{\\{stem}Min}}{{{min(xs, key=float)}}}', f'\\newcommand{{\\{stem}Max}}{{{max(xs, key=float)}}}']
    text = '\n'.join(L) + '\n'
    for z in ('0.0', '0.00', '0.000'):
        text = text.replace('{-' + z + '}', '{' + z + '}')
    (TABLES / name).write_text(text, encoding='utf-8')
    print('wrote', name, sum(x.startswith('\\newcommand') for x in L), 'macros')


def write(name, cols, rows):
    with open(TABLES / name, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(cols)
        w.writerows(rows)
    print('wrote', name, len(rows), 'rows')


def paired(nat, xre):
    """nat, xre: dict seed -> value. Returns gains, mean, min, max, n_pos, p (two-sided paired t over seeds)."""
    from scipy import stats
    g = [xre[s] - nat[s] for s in SEEDS]
    p = float(stats.ttest_rel([xre[s] for s in SEEDS], [nat[s] for s in SEEDS]).pvalue)
    return dict(g=g, mean=sum(g) / len(g), min=min(g), max=max(g), npos=sum(x > 0 for x in g), p=p)


WORD = {'le3': 'LeThree', '4to7': 'FourSeven', '8to15': 'EightFifteen', 'ge16': 'GeSixteen', 'all': 'All',
        'le7': 'LeSeven', 'ge8': 'GeEight'}


def macros_hist():
    rows = list(csv.DictReader(open(TABLES / 'hyp_strat_hist.csv', encoding='utf-8')))
    v = {(r['dataset'], r['arm'], int(r['seed']), r['bin']): r for r in rows}
    L = ['% generated by tables/src/collect_hyp_strat.py from tables/hyp_strat_hist.csv; do not edit',
         '% relative gain (%) of MSCA + XRE over MSCA per history bin = 100 * mean over seeds of (XRE - Native) / Native']
    rel_by = {}
    for ds in DSS:
        D = DSN[ds]
        for b in ['all'] + [x[0] for x in BINS + POOLED]:
            for met, M in (('recall20', 'Rec'), ('ndcg20', 'Ndcg')):
                nat = {s: float(v[(ds, 'native', s, b)][met]) for s in SEEDS}
                xre = {s: float(v[(ds, 'xre', s, b)][met]) for s in SEEDS}
                rel = [100 * (xre[s] - nat[s]) / nat[s] for s in SEEDS]
                st = paired(nat, xre)
                rel_by[(ds, b, met)] = sum(rel) / 5
                L.append(f'\\newcommand{{\\hzHist{M}Rel{WORD[b]}{D}}}{{{sum(rel) / 5:.1f}}}')
                L.append(f'\\newcommand{{\\hzHist{M}Npos{WORD[b]}{D}}}{{{st["npos"]}}}')
                L.append(f'\\newcommand{{\\hzHist{M}P{WORD[b]}{D}}}{{{st["p"]:.3f}}}')
            share = 100 * int(v[(ds, 'native', 1, b)]['n_users']) / int(v[(ds, 'native', 1, 'all')]['n_users'])
            L.append(f'\\newcommand{{\\hzHistShare{WORD[b]}{D}}}{{{share:.1f}}}')
    # range over datasets of the relative Recall@20 gain in the shortest and the longest bin
    for b in ('le3', 'ge16', 'le7', 'ge8'):
        xs = [rel_by[(ds, b, 'recall20')] for ds in DSS]
        L.append(f'\\newcommand{{\\hzHistRecRel{WORD[b]}Min}}{{{min(xs):.1f}}}')
        L.append(f'\\newcommand{{\\hzHistRecRel{WORD[b]}Max}}{{{max(xs):.1f}}}')
    # number of datasets whose relative Recall@20 / NDCG@20 gain is larger for short (<= 7) than for long (>= 8) histories
    for met, M in (('recall20', 'Rec'), ('ndcg20', 'Ndcg')):
        k = sum(rel_by[(ds, 'le7', met)] > rel_by[(ds, 'ge8', met)] for ds in DSS)
        L.append(f'\\newcommand{{\\hzHist{M}ShortWins}}{{{k}}}')
    write_macros('macros_hyp_strathist.tex', L)


GW = {'nbr': 'Nbr', 'non': 'Non', 'q1': 'QOne', 'q2': 'QTwo', 'q3': 'QThree', 'q4': 'QFour', 'q5': 'QFive', 'all': 'All'}
GG = {'link': 'Link', 'struct': 'Struct', 'pop': 'Pop', 'all': ''}


def macros_pop():
    rows = list(csv.DictReader(open(TABLES / 'hyp_strat_pop.csv', encoding='utf-8')))
    v = {(r['dataset'], r['arm'], int(r['seed']), r['grouping'], r['group']): r for r in rows}
    L = ['% generated by tables/src/collect_hyp_strat.py from tables/hyp_strat_pop.csv; do not edit',
         '% Hit = share (%) of test targets in the top 20, mean over seeds; Gain = XRE - Native (points); Rel = relative gain (%)']
    for ds in DSS:
        D = DSN[ds]
        for gname, g in [('all', 'all'), ('link', 'nbr'), ('link', 'non'), ('struct', 'nbr'), ('struct', 'non')] + \
                        [('pop', f'q{q}') for q in range(1, NQ + 1)]:
            nat = {s: float(v[(ds, 'native', s, gname, g)]['hit20']) for s in SEEDS}
            xre = {s: float(v[(ds, 'xre', s, gname, g)]['hit20']) for s in SEEDS}
            st = paired(nat, xre)
            tag = f'{GG[gname]}{GW[g]}{D}'
            L.append(f'\\newcommand{{\\hzPopHitNat{tag}}}{{{100 * sum(nat.values()) / 5:.1f}}}')
            L.append(f'\\newcommand{{\\hzPopHitXre{tag}}}{{{100 * sum(xre.values()) / 5:.1f}}}')
            L.append(f'\\newcommand{{\\hzPopGain{tag}}}{{{100 * st["mean"]:.2f}}}')
            L.append(f'\\newcommand{{\\hzPopRel{tag}}}{{{100 * sum((xre[s] - nat[s]) / nat[s] for s in SEEDS) / 5:.1f}}}')
            L.append(f'\\newcommand{{\\hzPopNpos{tag}}}{{{st["npos"]}}}')
            L.append(f'\\newcommand{{\\hzPopP{tag}}}{{{st["p"]:.3f}}}')
            if gname != 'all':
                share = 100 * int(v[(ds, 'native', 1, gname, g)]['n_targets']) / int(v[(ds, 'native', 1, 'all', 'all')]['n_targets'])
                L.append(f'\\newcommand{{\\hzPopShare{tag}}}{{{share:.1f}}}')
    # share (%) of the targets outside the MSCA top 20 that are not co-occurrence neighbors of the history, mean over seeds
    for ds in DSS:
        sh = []
        for s in SEEDS:
            o = {g: int(v[(ds, 'native', s, 'struct', g)]['n_targets']) * (1 - float(v[(ds, 'native', s, 'struct', g)]['hit20']))
                 for g in ('nbr', 'non')}
            sh.append(100 * o['non'] / (o['nbr'] + o['non']))
        L.append(f'\\newcommand{{\\hzPopOutNonShare{DSN[ds]}}}{{{sum(sh) / 5:.1f}}}')
        # hit-rate gain (points) of co-occurrence neighbors / non-neighbors inside the top and the fourth popularity quintile
        for g, G in (('nbr_q5', 'NbrQFive'), ('non_q5', 'NonQFive'), ('nbr_q4', 'NbrQFour'), ('non_q4', 'NonQFour')):
            gs = [float(v[(ds, 'xre', s, 'structpop', g)]['hit20']) - float(v[(ds, 'native', s, 'structpop', g)]['hit20'])
                  for s in SEEDS]
            L.append(f'\\newcommand{{\\hzPopGainCross{G}{DSN[ds]}}}{{{100 * sum(gs) / 5:.2f}}}')
    write_macros('macros_hyp_stratpop.tex', L, ['hzPopGainStructNbr', 'hzPopGainStructNon', 'hzPopShareStructNbr', 'hzPopHitNatStructNbr', 'hzPopHitNatStructNon',
                                                'hzPopGainLinkNbr', 'hzPopGainLinkNon', 'hzPopGainPopQFive', 'hzPopGainPopQFour',
                                                'hzPopGainPopQThree', 'hzPopOutNonShare', 'hzPopGainAll', 'hzPopGainCrossNbrQFive',
                                                'hzPopGainCrossNonQFive', 'hzPopGainCrossNbrQFour', 'hzPopGainCrossNonQFour'])


def load_trans():
    """(dataset, seed, pair) -> dict with n[(msca_bin, other_bin)], w[(msca_bin, other_bin)] (Recall@20 mass), n_all,
    recall_a, recall_b"""
    T = {}
    for r in csv.DictReader(open(TABLES / 'hyp_rankshift_trans.csv', encoding='utf-8')):
        k = (r['dataset'], int(r['seed']), r['pair'])
        t = T.setdefault(k, dict(n={}, w={}, n_all=int(r['n_all']), recall_a=float(r['recall_a']),
                                 recall_b=float(r['recall_b'])))
        t['n'][(r['msca_bin'], r['other_bin'])] = int(r['n_targets'])
        t['w'][(r['msca_bin'], r['other_bin'])] = float(r['recall_mass'])
    return T


def rank_stats(t):
    """per-seed quantities (% of all test targets unless stated) from one 3 x 3 transition matrix t"""
    n, N = t['n'], t['n_all']
    pc = lambda x: 100 * x / N   # noqa: E731
    coh = {b: sum(n[(b, c)] for c, _, _ in CB) for b, _, _ in CB}
    q = dict(recnear=pc(n[('near', 'top')]), recdeep=pc(n[('deep', 'top')]),
             lostnear=pc(n[('top', 'near')]), lostdeep=pc(n[('top', 'deep')]))
    q['rec'], q['lost'] = q['recnear'] + q['recdeep'], q['lostnear'] + q['lostdeep']
    q['gain'] = q['rec'] - q['lost']
    q['recnearshare'] = 100 * q['recnear'] / q['rec']
    q['lostnearshare'] = 100 * q['lostnear'] / q['lost']
    q['ratenear'] = 100 * n[('near', 'top')] / coh['near']   # % of the MSCA 21-40 cohort that enters the top 20
    q['ratedeep'] = 100 * n[('deep', 'top')] / coh['deep']
    q['keep'] = 100 * n[('top', 'top')] / coh['top']          # % of the MSCA top 20 that stays in the top 20
    for b, _, _ in CB:
        q['coh' + b] = pc(coh[b])
        for c, _, _ in CB:
            q['t' + b + c] = pc(n[(b, c)])
    # Recall@20 (per-user averaging) in points: gain, and the same recovered / lost split with the user weights
    w = t['w']
    q['recallgain'] = 100 * (t['recall_b'] - t['recall_a'])
    q['recallnet'] = 100 * (w[('near', 'top')] + w[('deep', 'top')] - w[('top', 'near')] - w[('top', 'deep')])
    assert abs(q['recallgain'] - q['recallnet']) < 1e-5, (q['recallgain'], q['recallnet'])
    q['recallrecdeepshare'] = 100 * w[('deep', 'top')] / (w[('near', 'top')] + w[('deep', 'top')])
    return q


BW = {'top': 'Top', 'near': 'Near', 'deep': 'Deep'}
RNM = {'recnear': 'RecNear', 'recdeep': 'RecDeep', 'lostnear': 'LostNear', 'lostdeep': 'LostDeep', 'rec': 'Rec',
       'lost': 'Lost', 'gain': 'Gain', 'recnearshare': 'RecNearShare', 'lostnearshare': 'LostNearShare',
       'ratenear': 'RateNear', 'ratedeep': 'RateDeep', 'keep': 'Keep'}
ONE_DP = {'recnearshare', 'lostnearshare', 'ratenear', 'keep', 'recallrecdeepshare'}


def macros_rank():
    T = load_trans()
    L = ['% generated by tables/src/collect_hyp_strat.py from tables/hyp_rankshift_trans.csv; do not edit',
         '% Rank bins: Top = rank 1-20 (evaluation cutoff), Near = 21-40 (just below the cutoff; 40 = list size K of XRE),',
         '% Deep = beyond 40. MSCA of seed s against MSCA + XRE of seed s (no infix) or MSCA of seed s mod 5 + 1 (infix Ctrl).',
         '% RecNear / RecDeep = recovered hits = targets with MSCA rank 21-40 / >40 that the second model ranks in the top 20;',
         '% LostNear / LostDeep = lost hits = targets in the MSCA top 20 that the second model ranks 21-40 / >40;',
         '% Rec, Lost = their sums, Gain = Rec - Lost (all in % of all test targets, per-target counting, mean over seeds 1-5);',
         '% RecNearShare = % of recovered hits from MSCA 21-40; LostNearShare = % of lost hits that land at 21-40;',
         '% RateNear / RateDeep = % of the MSCA 21-40 / >40 cohort that enters the top 20 (entry rate); Keep = % of the MSCA',
         '% top 20 that stays; Coh<Bin> = cohort size (%); T<MSCA bin><XRE bin> = transition matrix (% of test targets);',
         '% Lo / Hi (before the dataset) = smallest / largest seed; Npos = seeds with a positive value;',
         '% Ex<Q> = <Q> of MSCA + XRE minus <Q> of the seed-pair control, paired by seed s (points);',
         '% RecallGain = test Recall@20 of MSCA + XRE minus MSCA (per-user averaging, points); RecallRecDeepShare = share (%)',
         '% of the Recall@20 mass of the recovered hits that comes from MSCA ranks beyond 40; SignAgree = seeds on which',
         '% RecallGain and Gain have the same sign (SignAgreeAll: of the 20 dataset-seed pairs).']
    S = {}
    for ds in DSS:
        for pair in ('xre', 'ctrl'):
            for s in SEEDS:
                S[(ds, pair, s)] = rank_stats(T[(ds, s, pair)])
    f = lambda k, x: f'{x:.1f}' if k in ONE_DP else f'{x:.2f}'   # noqa: E731
    mac = lambda name, v: L.append('\\newcommand{\\' + name + '}{' + v + '}')   # noqa: E731
    ranges = []
    for ds in DSS:
        D = DSN[ds]
        for pair, P in (('xre', ''), ('ctrl', 'Ctrl')):
            for k, K in RNM.items():
                xs = [S[(ds, pair, s)][k] for s in SEEDS]
                mac(f'hzRank{P}{K}{D}', f(k, sum(xs) / 5))
                mac(f'hzRank{P}{K}Lo{D}', f(k, min(xs)))
                mac(f'hzRank{P}{K}Hi{D}', f(k, max(xs)))
                mac(f'hzRank{P}{K}Npos{D}', str(sum(x > 0 for x in xs)))
                if ds == DSS[0]:
                    ranges.append(f'hzRank{P}{K}')
        for k, K in RNM.items():   # excess of MSCA + XRE over the seed-pair control, paired by seed
            xs = [S[(ds, 'xre', s)][k] - S[(ds, 'ctrl', s)][k] for s in SEEDS]
            mac(f'hzRankEx{K}{D}', f(k, sum(xs) / 5))
            mac(f'hzRankEx{K}Npos{D}', str(sum(x > 0 for x in xs)))
            if ds == DSS[0]:
                ranges.append(f'hzRankEx{K}')
        for b, _, _ in CB:
            mac(f'hzRankCoh{BW[b]}{D}', f"{sum(S[(ds, 'xre', s)]['coh' + b] for s in SEEDS) / 5:.1f}")
            for c, _, _ in CB:
                mac(f'hzRankT{BW[b]}{BW[c]}{D}', f"{sum(S[(ds, 'xre', s)]['t' + b + c] for s in SEEDS) / 5:.2f}")
        rg = [S[(ds, 'xre', s)]['recallgain'] for s in SEEDS]
        hg = [S[(ds, 'xre', s)]['gain'] for s in SEEDS]
        rd = [S[(ds, 'xre', s)]['recallrecdeepshare'] for s in SEEDS]
        mac(f'hzRankRecallGain{D}', f'{sum(rg) / 5:.2f}')
        mac(f'hzRankRecallGainNpos{D}', str(sum(x > 0 for x in rg)))
        mac(f'hzRankRecallRecDeepShare{D}', f'{sum(rd) / 5:.1f}')
        mac(f'hzRankSignAgree{D}', str(sum((x > 0) == (y > 0) for x, y in zip(rg, hg))))
    ranges += ['hzRankRecallGain', 'hzRankRecallRecDeepShare'] + [f'hzRankCoh{BW[b]}' for b, _, _ in CB]
    mac('hzRankSignAgreeAll', str(sum((S[(ds, 'xre', s)]['recallgain'] > 0) == (S[(ds, 'xre', s)]['gain'] > 0)
                                      for ds in DSS for s in SEEDS)))
    # DEPRECATED (first version, mixed binning: targets entering from MSCA 21-40 minus targets leaving into XRE 21-40).
    # Kept only so that text/4_ex/4c_abl_1.tex, which still cites them, compiles; the quantity is no flow of one cohort.
    per_ds = [sum(S[(ds, 'xre', s)]['recnear'] - S[(ds, 'xre', s)]['lostnear'] for s in SEEDS) / 5 for ds in DSS]
    L.append('% DEPRECATED, do not cite: RecNear - LostNear (mixed binning of the first version)')
    mac('hzRankNetNearMin', f'{min(per_ds):.2f}')
    mac('hzRankNetNearMax', f'{max(per_ds):.2f}')
    write_macros('macros_hyp_rankshift.tex', L, ranges)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'remote':
        return remote(only_rank=len(sys.argv) > 2 and sys.argv[2] == 'rank')
    mode = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if mode in ('all', 'rank'):
        src = open(__file__, encoding='utf-8').read()
        out = subprocess.run(SSH + [REMOTE_PY, '-', 'remote'] + (['rank'] if mode == 'rank' else []),
                             input=src.encode('utf-8'), capture_output=True)
        if out.returncode:
            sys.exit(out.stderr.decode('utf-8', 'replace'))
        d = json.loads(out.stdout.decode('utf-8'))
        if mode == 'all':
            write('hyp_strat_hist.csv', H_COLS, d['hist'])
            write('hyp_strat_pop.csv', P_COLS, d['pop'])
        write('hyp_rankshift.csv', R_COLS, d['rank'])
        write('hyp_rankshift_trans.csv', T_COLS, d['trans'])
    if mode != 'rank':
        macros_hist()
        macros_pop()
    macros_rank()


if __name__ == '__main__':
    main()
