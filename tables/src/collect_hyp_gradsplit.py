"""Hypothesis figure fig_h_gradsplit: direction of the entry gradient (Pantry, Scientific).

Hypothesis. The exchange gain comes from demoting the list occupant that the entry pair displaces, rather than from raising
the held-out positive. The entry pair softplus(s(u,D) - s(u,p)) sends two gradients: one raises the positive p, the other
lowers the displaced occupant D. Each split arm keeps one of them with a stop-gradient on the other score.

Data: tables/gradsplit_perseed.csv (tables/src/collect_analysis.py; P04_prune on HOST_A, MSCA, lambda 0.1, K 40,
ONE run per seed, seeds 1-5; the Native, A1 and XRE rows are the manuscript runs, checked against results_prune.json).
Arms (manuscript label <- arm id of the CSV)
  Proposed (XRE)            <- XRE      masked-view exchange term + anchor term (Table I runs)
  Unmatched full-view XRE   <- e3       E3 = exchange term on the full graph + anchor term, no mass rescaling (the base of
                                         the split; P04_prune README: entry + retention + A1, full view). Pantry
                                         seed 1 is the P04_rev run xrev pantry_xrev_xfp_l010, seeds 2-5 are P04_prune
                                         xprune pantry_xprune_e3 runs; checked 2026-10-07 (R2335): the xprune config
                                         is the xrev config plus 'prune_arm: e3', overall.yaml and dataset yaml are
                                         identical, every shared model source has the same sha256, and the P04_prune
                                         unit test (off_e3) gives bitwise equal losses and gradients, so one
                                         configuration. Named 'XRE on full view' before R2335; renamed because Table
                                         III, Fig. 4 and supplementary Table S6 use that name for the mass-matched run.
  Raise positive only       <- pos      E3 with entry softplus(sg(s(u,D)) - s(u,p)): only the gradient raising p
  Lower list occupant only  <- disp     E3 with entry softplus(s(u,D) - sg(s(u,p))): only the gradient lowering D
  Lower random list item    <- bdrand   disp with D replaced by a uniform random non-positive item of the list
  Lower occupant, no anchor <- bdnoanc  disp without the anchor term
  Anchor term (A)           <- A1       anchor term alone
  Backbone (MSCA)           <- Native   MSCA without an auxiliary term
Tests (test Recall@20, same seed = same initial parameters within the MSCA packages)
  vs_e3     arm minus E3, paired over seeds 1-5 (Native: Welch, a different run set). Pre-set decision rule of P04_prune
            (README, fixed before the runs): "A < E3" if mean difference <= -0.0010 and A lower on >= 4 of 5 seeds,
            "A ~ E3" if mean difference > -0.0010 and A lower on <= 3 of 5 seeds, otherwise "undecided".
  disp_pos  Lower list occupant only minus Raise positive only, paired (the direct contrast of the two directions).
  vs_xre    arm minus XRE, paired.
Writes tables/hyp_gradsplit_cells.csv (mean, SD, seeds per arm), tables/hyp_gradsplit_tests.csv (one row per contrast),
tables/macros_hyp_gradsplit.tex: hyGrad<Arm><DS>{Delta,Rel,Lower,P,Verdict} (arm minus E3; Delta = signed Recall@20 in
units of 1e-4 printed as a signed decimal, Rel = percent of the E3 mean, Lower = seeds below E3), hyGradDispPos<DS>{Delta,
Higher,P}, hyGradXreFull<DS>{Delta,Higher,P} (XRE minus E3); Arm in Pos, Disp, Rand, Noanc, Aone; DS in Pantry, Scientific.
Usage: python -s tables/src/collect_hyp_gradsplit.py      (then python -s figures/src/fig_h_gradsplit.py)
"""
import csv
import statistics as st
from pathlib import Path

from scipy import stats

TABLES = Path(__file__).resolve().parents[1]
SRC = TABLES / 'gradsplit_perseed.csv'
DATASETS = [('pantry', 'Pantry'), ('scientific', 'Scientific')]
ARMS = [('XRE', 'Proposed (XRE)'), ('e3', 'Unmatched full-view XRE'), ('pos', 'Raise positive only'),
        ('disp', 'Lower list occupant only'), ('bdrand', 'Lower random list item'), ('bdnoanc', 'Lower occupant, no anchor'),
        ('A1', 'Anchor term (A)'), ('Native', 'Backbone (MSCA)')]
MACRO_ARM = {'pos': 'Pos', 'disp': 'Disp', 'bdrand': 'Rand', 'bdnoanc': 'Noanc', 'A1': 'Aone', 'Native': 'Native'}
SEEDS = [1, 2, 3, 4, 5]


def load():
    v = {}
    with open(SRC, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            v[(r['dataset'], r['arm'], int(r['seed']))] = float(r['r20'])
    return v


def contrast(v, ds, a, b, welch=False):
    x = [v[(ds, a, s)] for s in SEEDS]
    y = [v[(ds, b, s)] for s in SEEDS]
    d = [p - q for p, q in zip(x, y)]
    if welch:
        p = float(stats.ttest_ind(x, y, equal_var=False).pvalue)
        m = st.mean(x) - st.mean(y)
    else:
        p = float(stats.ttest_rel(x, y).pvalue) if any(abs(e) > 1e-12 for e in d) else 1.0
        m = st.mean(d)
    m = 0.0 if abs(m) < 5e-9 else m
    hi, lo = sum(e > 1e-12 for e in d), sum(e < -1e-12 for e in d)
    verdict = 'lower' if (m <= -0.0010 + 1e-12 and lo >= 4) else ('similar' if (m > -0.0010 and lo <= 3) else 'undecided')
    return dict(dataset=ds, a=a, b=b, test='welch' if welch else 'paired', n=len(SEEDS), diff=f'{m:.6f}',
                rel_pct=f'{100 * m / st.mean(y):.2f}', higher=hi, lower=lo, p=f'{p:.4f}', rule=verdict)


def sgn(x, nd=4):
    x = float(x)
    s = f'{abs(x):.{nd}f}'
    s = s.lstrip('0') if nd == 4 else s   # Recall differences as .0018 (manuscript style), percentages as 0.5
    return ('$-$' if x < -5 * 10 ** -(nd + 1) else ('+' if x > 5 * 10 ** -(nd + 1) else '')) + s


def main():
    v = load()
    cells, tests = [], []
    for ds, _ in DATASETS:
        for arm, lab in ARMS:
            x = [v[(ds, arm, s)] for s in SEEDS]
            cells.append(dict(dataset=ds, arm=arm, label=lab, n=len(x), r20_mean=f'{st.mean(x):.6f}', r20_sd=f'{st.stdev(x):.6f}',
                              seeds='/'.join(f'{e:.4f}' for e in x)))
        for arm, _ in ARMS:
            if arm != 'e3':
                tests.append(dict(family='vs_e3', **contrast(v, ds, arm, 'e3', welch=arm == 'Native')))
        for fam, a, b in (('disp_pos', 'disp', 'pos'), ('rand_disp', 'bdrand', 'disp')):
            tests.append(dict(family=fam, **dict(contrast(v, ds, a, b), rule='')))   # the pre-set rule covers arm vs E3 only
        for arm, _ in ARMS:
            if arm != 'XRE':
                tests.append(dict(family='vs_xre', **contrast(v, ds, arm, 'XRE', welch=arm == 'Native')))
    for path, rows in ((TABLES / 'hyp_gradsplit_cells.csv', cells), (TABLES / 'hyp_gradsplit_tests.csv', tests)):
        with open(path, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
            w.writeheader()
            w.writerows(rows)
    T = {(t['family'], t['dataset'], t['a']): t for t in tests}
    mac = []
    for ds, D in DATASETS:
        for arm, A in MACRO_ARM.items():
            t = T[('vs_e3', ds, arm)]
            mac += [(f'hyGrad{A}{D}Delta', sgn(t['diff'])), (f'hyGrad{A}{D}Rel', sgn(t['rel_pct'], 1)),
                    (f'hyGrad{A}{D}Lower', str(t['lower'])), (f'hyGrad{A}{D}P', f"{float(t['p']):.3f}"),
                    (f'hyGrad{A}{D}Verdict', t['rule'])]
        for fam, name in (('disp_pos', 'DispPos'), ('rand_disp', 'RandDisp')):
            t = T[(fam, ds, 'disp' if fam == 'disp_pos' else 'bdrand')]
            mac += [(f'hyGrad{name}{D}Delta', sgn(t['diff'])), (f'hyGrad{name}{D}Rel', sgn(t['rel_pct'], 1)),
                    (f'hyGrad{name}{D}Higher', str(t['higher'])), (f'hyGrad{name}{D}P', f"{float(t['p']):.3f}")]
        t = T[('vs_e3', ds, 'XRE')]
        mac += [(f'hyGradXreFull{D}Delta', sgn(t['diff'])), (f'hyGradXreFull{D}Higher', str(t['higher'])),
                (f'hyGradXreFull{D}P', f"{float(t['p']):.3f}")]
    lines = ['% Generated by tables/src/collect_hyp_gradsplit.py from tables/gradsplit_perseed.csv. Do not edit by hand.',
             '% Test Recall@20, seeds 1-5, one run per seed. hyGrad<Arm><DS>*: arm minus unmatched full-view XRE (E3), paired',
             '% (Native: Welch); Delta signed, Rel percent of the E3 mean, Lower seeds below E3, P two-sided t-test,',
             '% Verdict = pre-set P04_prune rule (lower / similar / undecided). DispPos = lower occupant only minus raise',
             '% positive only; RandDisp = random list item minus occupant; XreFull = XRE minus E3.']
    lines += [f'\\newcommand{{\\{k}}}{{{x}}}' for k, x in mac]
    (TABLES / 'macros_hyp_gradsplit.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
    for t in tests:
        print(t['family'], t['dataset'], t['a'], t['b'], t['diff'], t['rel_pct'], f"{t['higher']}/{t['lower']}", t['p'], t['rule'])


if __name__ == '__main__':
    main()
