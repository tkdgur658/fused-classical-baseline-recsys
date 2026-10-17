"""Hypothesis figure fig_h_forest: where the gain of XRE over the backbone MSCA comes from, as seed-paired contrasts.

Hypothesis. The gain comes from high-scoring partners. The anchor term (A) and the exchange term each add Recall@20, a
uniform partner on the masked view adds little over A, a sampled high-scoring partner (masked DNS) recovers most of the
exchange gain, and the masked view matters without the anchor term (reported as the data say, also where they disagree).
Data (exactly the runs of Table III tab:xmask and tables/paired_ci.csv, one run per seed, seeds 1-5)
  tables/xmask_perseed.csv   rows mbpr (A + masked BPR), mdns (A + masked DNS) (tables/src/collect_xmask.py, 45 runs,
                             one per row x dataset x seed)
  tables/xre_perseed.csv     MSCA rows XRE, A1 (anchor term A alone), wo_A1 (Exchange, masked view), wo_A1_masked
                             (Exchange, full view), wo_masked (XRE on full view) (tables/src/collect_xre.py, rep1 runs;
                             n_runs == 1 is asserted, as build_paired.py does)
  tables/m1probe_perseed.csv row A1first (anchor term alone with the first of the same 50 probes, collect_m1probe.py)
  tables/table1_perseed.csv  MSCA Native (Backbone (MSCA), the run set of tables/src/native_rows.py)
  tables/ablation_tests.csv  p of the contrasts masked / fullxre (cross-check only, figure family of fig_ablation.py)
Contrasts (a - b of test Recall@20, the same seed in a and b, d_s over seeds 1-5)
  1 Anchor  A - MSCA                         gain of the anchor term
  2 Exch    XRE - A                          gain added by the exchange term
  3 Unif    (A + masked BPR) - A             uniform partner
  4 Dns     (A + masked DNS) - A             high-scoring sampled partner
  5 List    XRE - (A + masked DNS)           list occupants versus a sampled hard partner
  6 MaskX   Exchange (masked) - Exchange (full view)    masked view without the anchor term
  7 MaskA   XRE - XRE on full view           masked view with the anchor term
  8 Inter   (7) - (6) per seed               interaction of the masked view with the anchor term
  9 DnsVsBpr  (A + masked DNS) - (A + masked BPR)   high-scoring sampled partner against a uniform partner
 10 XreVsBpr  XRE - (A + masked BPR)                list occupants against a uniform partner
  Rows 9 and 10 (direct partner contrasts, added 2026-10-07 on reviewer request) follow row 8, so the numbers (1)-(8)
  cited in the text stay valid.
 11 Probe   A - (A, single probe)        strongest of the M = 50 probes against the first of the same 50 draws (M = 1)
  Row 11 (reviewer control M = 1, added 2026-10-07): A1first = tables/m1probe_perseed.csv (tables/src/collect_m1probe.py,
  15 runs, one per dataset x seed); its CI must equal \\mpCIR<DS> of tables/macros_m1probe.tex.
Scale. Absolute differences of Recall@20 are used (plotted x 10^3), not differences relative to MSCA. Absolute values
reproduce the intervals of tables/macros_paired.tex and the means of Table III digit for digit, which the cross-check
below needs, and the eight contrasts share one axis because the three datasets differ by up to 0.04 in Recall@20 while
every contrast lies within about 0.006. Relative values would only rescale each dataset by one constant (MSCA mean
0.104 / 0.109 / 0.138), so no ordering or sign changes. The relative mean (% of the MSCA mean) is still stored in the CSV.
Statistics: mean(d), sample SD, 95% paired t-interval mean +- t_{0.975,4} sd / sqrt(5), two-sided paired t-test p
(scipy ttest_rel; for 8 the one-sample test of the per-seed interaction), seeds with d_s > 0 (k/5).
 12 MaskDns (A + masked DNS) - (A + full-view DNS)   masking with a sampled high-scoring partner
  Row 12 (reviewer control E1, added 2026-10-07): mdnsfv = tables/e1_perseed.csv (tables/src/collect_e1.py, 15 runs, one per
  dataset x seed); its CI must equal \\eCIR<DS> of tables/macros_e1.tex. Rows 11 and 12 form the group 'controls'.
Datasets (per contrast since 2026-10-07): a contrast enters a dataset once both of its rows have seeds 1-5 there. Every
contrast covers Baby, Pantry, Scientific and Sports, all treated alike (user decision 2026-10-07: the hardware of the
runs is not part of the analysis; the former Same / Mixed macros are removed). hfNDs / hfDsPhrase name the datasets of the core
contrasts, hf<C>NDs / hf<C>DsAll those of contrast C.
Cross-checks (printed, never hidden; mismatches are listed and the script exits with status 1):
  contrast 10 CI == pcBprCI<DS> (tables/macros_paired.tex),
  contrast 5 CI == \\pcDnsCI<DS>, contrast 2 CI == \\pcAoneCI<DS>, contrast 1 CI == \\pcAoneNativeCI<DS>,
  contrast 6 CI == \\pcAblMaskedCI<DS>, contrast 7 CI == \\pcAblFullxreCI<DS> (tables/macros_paired.tex), the p of 6 and 7
  against tables/ablation_tests.csv, and the DNS share against \\xmDnsShareMin / \\xmDnsShareMax (tables/macros_xmask.tex).
Contrast ids 1-12 stay internal (macros, stats_audit.csv). Fig. h_forest (fig_h_forest.py, since 2026-10-07) shows six of
them renumbered: (1) = id 1, (2) = id 2, (3) = id 3, (4) = id 4, (5) = id 9, (6) = id 5; the text cites ids 6, 7, 8, 10, 11, 12
by name through Table III / supplementary Table S4.
Writes tables/hyp_forest.csv and tables/macros_hyp_forest.tex (macro names hf..., no digits):
  hf<C>Mean<DS>, hf<C>CI<DS> (absolute Recall@20, 4 dp, the format of macros_paired.tex), hf<C>P<DS> (paired t p, 3 dp),
  hf<C>Pos<DS> (k/5), hf<C>NExcl (word: datasets whose CI excludes zero), hf<C>Min / hf<C>Max (means over datasets) for
  C in Anchor, Exch, Unif, Dns, List, MaskX, MaskA, Inter, DnsVsBpr, XreVsBpr, Probe;
  hfDnsShareMin/Max = 100 mean(4) / mean(2) (%), hfUnifShareMin/Max = 100 mean(3) / mean(2) (%),
  hfAnchorShareMin/Max = 100 mean(1) / (mean(1) + mean(2)) (% of the total XRE - MSCA gain), hfNDs, hfDsPhrase.
Usage: python -s tables/src/collect_hyp_forest.py      (then python -s figures/src/fig_h_forest.py)
"""
import csv
import math
import re
import statistics as st
import sys
from pathlib import Path

from scipy import stats

TABLES = Path(__file__).resolve().parents[1]
SRC = {'xmask': TABLES / 'xmask_perseed.csv', 'xre': TABLES / 'xre_perseed.csv', 't1': TABLES / 'table1_perseed.csv',
       'm1': TABLES / 'm1probe_perseed.csv', 'e1': TABLES / 'e1_perseed.csv',
       'abl': TABLES / 'ablation_tests.csv'}
OUT = TABLES / 'hyp_forest.csv'
MAC = TABLES / 'macros_hyp_forest.tex'
DS_ALL = [('baby', 'Baby'), ('pantry', 'Pantry'), ('scientific', 'Scientific'), ('sports', 'Sports')]
SEEDS = [1, 2, 3, 4, 5]
XRE_ROWS = ('XRE', 'A1', 'wo_A1', 'wo_A1_masked', 'wo_masked')
# (id, macro name, group, a, b, description); 'Inter' is built from 6 and 7
CONTRASTS = [
    (1, 'Anchor', 'gain components', 'A1', 'Native', 'Anchor term (A) - Backbone (MSCA)'),
    (2, 'Exch', 'gain components', 'XRE', 'A1', 'Proposed (XRE) - Anchor term (A)'),
    (3, 'Unif', 'partner type', 'mbpr', 'A1', '(A + masked BPR) - Anchor term (A)'),
    (4, 'Dns', 'partner type', 'mdns', 'A1', '(A + masked DNS) - Anchor term (A)'),
    (5, 'List', 'partner type', 'XRE', 'mdns', 'Proposed (XRE) - (A + masked DNS)'),
    (6, 'MaskX', 'masked view', 'wo_A1', 'wo_A1_masked', 'Exchange (masked view) - Exchange (full view)'),
    (7, 'MaskA', 'masked view', 'XRE', 'wo_masked', 'Proposed (XRE) - XRE on full view'),
    (8, 'Inter', 'masked view', None, None, '(7) - (6): masked-view gain with A minus without A'),
    (9, 'DnsVsBpr', 'partner vs BPR', 'mdns', 'mbpr', '(A + masked DNS) - (A + masked BPR)'),
    (10, 'XreVsBpr', 'partner vs BPR', 'XRE', 'mbpr', 'Proposed (XRE) - (A + masked BPR)'),
    (11, 'Probe', 'controls', 'A1', 'A1first', 'Anchor term (A) - single probe (first of the same 50 probes)'),
    (12, 'MaskDns', 'controls', 'mdns', 'mdnsfv', '(A + masked DNS) - (A + full-view DNS)'),
]
NEEDED = sorted({x for c in CONTRASTS for x in c[3:5] if x})
WORDS = 'none one two three four'.split()


def load():
    V = {}

    def put(row, ds, seed, r20):
        key = (row, ds, int(seed))
        assert key not in V, f'duplicate run {key}'   # one run per seed, never replicates
        V[key] = float(r20)
    for r in csv.DictReader(open(SRC['xmask'], encoding='utf-8')):
        put(r['row'], r['dataset'], r['seed'], r['r20'])
    for r in csv.DictReader(open(SRC['xre'], encoding='utf-8')):
        if r['backbone'] == 'MSCA' and r['row'] in XRE_ROWS:
            assert int(r['n_runs']) == 1, r
            put(r['row'], r['dataset'], r['seed'], r['r20'])
    for r in csv.DictReader(open(SRC['m1'], encoding='utf-8')):
        put(r['row'], r['dataset'], r['seed'], r['r20'])
    for r in csv.DictReader(open(SRC['e1'], encoding='utf-8')):
        put(r['row'], r['dataset'], r['seed'], r['r20'])
    for r in csv.DictReader(open(SRC['t1'], encoding='utf-8')):
        if r['backbone'] == 'MSCA' and r['row'] == 'Native':
            put('Native', r['dataset'], r['seed'], r['r20'])
    return V


def summary(d):
    n = len(d)
    mu, sd = st.mean(d), st.stdev(d)
    h = stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n)
    p = float(stats.ttest_1samp(d, 0.0).pvalue) if sd > 0 else float(mu == 0)
    return dict(n=n, mean=mu, sd=sd, ci_lo=mu - h, ci_hi=mu + h, ci_excl_zero=int(mu - h > 0 or mu + h < 0),
                pos=sum(x > 0 for x in d), p=p)


def fmt(v):
    return f'{v:+.4f}'.replace('-', '$-$')


def phrase(names):
    return names[0] if len(names) == 1 else ', '.join(names[:-1]) + (' and ' if len(names) == 2 else ', and ') + names[-1]


def read_macros(path):
    return dict(re.findall(r'\\newcommand\{\\(\w+)\}\{(.*)\}', path.read_text(encoding='utf-8')))


def main():
    V = load()
    def full(rows_, d):
        return all((k, d, s) in V for k in rows_ for s in SEEDS)
    CORE = sorted({x for c in CONTRASTS if c[0] <= 10 for x in c[3:5] if x})
    DS = [(d, D) for d, D in DS_ALL if full(CORE, d)]   # datasets of the core contrasts (1)-(10)
    for d, _ in DS_ALL[:3]:
        assert full(NEEDED, d), f'incomplete rows on {d}'
    DSC = {name: [(d, D) for d, D in DS if a is None or full((a, b), d)] for _, name, _, a, b, _ in CONTRASTS}
    print('datasets with every core row at seeds 1-5:', [D for _, D in DS],
          '| controls:', {n: [D for _, D in v] for n, v in DSC.items() if len(v) != len(DS)})
    rows, res = [], {}
    for d, D in DS:
        native = st.mean(V[('Native', d, s)] for s in SEEDS)
        diffs = {}
        for cid, name, grp, a, b, desc in CONTRASTS:
            if (d, D) not in DSC[name]:
                continue
            if a is None:
                dd = [x - y for x, y in zip(diffs[7], diffs[6])]
            else:
                dd = [V[(a, d, s)] - V[(b, d, s)] for s in SEEDS]
                if cid != 8:   # the paired t-test of a against b equals the one-sample test of d
                    pr = float(stats.ttest_rel([V[(a, d, s)] for s in SEEDS], [V[(b, d, s)] for s in SEEDS]).pvalue)
            diffs[cid] = dd
            s = summary(dd)
            if a is not None:
                assert abs(s['p'] - pr) < 1e-9, (cid, d)
            res[(name, d)] = s
            rows.append(dict(id=cid, contrast=name, group=grp, a=a or '(7)', b=b or '(6)', description=desc, dataset=d,
                             n=s['n'], mean=f"{s['mean']:.6g}", sd=f"{s['sd']:.6g}", ci_lo=f"{s['ci_lo']:.6g}",
                             ci_hi=f"{s['ci_hi']:.6g}", ci_excl_zero=s['ci_excl_zero'], pos=s['pos'],
                             p_paired=f"{s['p']:.6g}", mean_rel_pct=f"{100 * s['mean'] / native:.4g}",
                             msca_mean=f'{native:.6g}', **{f'd_s{k}': f'{x:.6g}' for k, x in zip(SEEDS, dd)}))
    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)

    # ---------------- macros
    out = ['% Generated by tables/src/collect_hyp_forest.py from tables/xmask_perseed.csv, xre_perseed.csv, '
           'm1probe_perseed.csv and table1_perseed.csv.',
           '% Do not edit by hand. Paired seed differences of test Recall@20 (a - b, seeds 1-5, one run per seed), mean and',
           '% 95% paired t-interval (df 4) in absolute Recall@20; P = paired t-test; Pos = seeds with a positive difference.']

    def mac(n, v):
        assert re.fullmatch(r'[A-Za-z]+', n), n
        out.append(f'\\newcommand{{\\{n}}}{{{v}}}')
    for _, name, *_ in CONTRASTS:
        DS_ = DSC[name]
        mac(f'hf{name}NDs', WORDS[len(DS_)])
        mac(f'hf{name}DsAll', phrase([D for _, D in DS_]))
        for d, D in DS_:
            s = res[(name, d)]
            mac(f'hf{name}Mean{D}', fmt(s['mean']))
            mac(f'hf{name}CI{D}', f"[{fmt(s['ci_lo'])}, {fmt(s['ci_hi'])}]")
            mac(f'hf{name}P{D}', f"{s['p']:.3f}" if s['p'] >= 0.001 else '$<$0.001')
            mac(f'hf{name}Pos{D}', f"{s['pos']}/5")
        mac(f'hf{name}NExcl', WORDS[sum(res[(name, d)]['ci_excl_zero'] for d, _ in DS_)])
        # R0013: names of the datasets whose interval excludes zero ('none of the datasets' if none)
        excl = [D for d, D in DS_ if res[(name, d)]['ci_excl_zero']]
        mac(f'hf{name}Ds', phrase(excl) if excl else 'none of the datasets')
        mac(f'hf{name}Min', fmt(min(res[(name, d)]['mean'] for d, _ in DS_)))
        mac(f'hf{name}Max', fmt(max(res[(name, d)]['mean'] for d, _ in DS_)))
        # dataset holding the largest / smallest mean (the prose names it, e.g. the largest uniform-partner share)
        mac(f'hf{name}ArgMax', max(DS_, key=lambda x: res[(name, x[0])]['mean'])[1])
        # 2026-10-07 (Sports): directional versions, since an interval can now exclude zero from below (prose 'above zero'
        # must use Up): DsUp / NUp = interval above zero, DsDown / NDown = interval below zero
        up = [D for d, D in DS_ if res[(name, d)]['ci_lo'] > 0]
        down = [D for d, D in DS_ if res[(name, d)]['ci_hi'] < 0]
        mac(f'hf{name}DsUp', phrase(up) if up else 'none of the datasets')
        mac(f'hf{name}NUp', WORDS[len(up)])
        mac(f'hf{name}DsDown', phrase(down) if down else 'none of the datasets')
        mac(f'hf{name}NDown', WORDS[len(down)])
    m = lambda name, d: res[(name, d)]['mean']   # noqa: E731
    shares = {'Dns': [100 * m('Dns', d) / m('Exch', d) for d, _ in DS],
              'Unif': [100 * m('Unif', d) / m('Exch', d) for d, _ in DS],
              'Anchor': [100 * m('Anchor', d) / (m('Anchor', d) + m('Exch', d)) for d, _ in DS]}
    for k, v in shares.items():
        mac(f'hf{k}ShareMin', f'{min(v):.0f}'.replace('-', '$-$'))
        mac(f'hf{k}ShareMax', f'{max(v):.0f}'.replace('-', '$-$'))
        mac(f'hf{k}ShareArgMax', DS[max(range(len(v)), key=lambda i: v[i])][1])
    # anchor share over the datasets on which the anchor term raises the mean (Sports 2026-10-07: A below MSCA there)
    pos = [i for i, (d, _) in enumerate(DS) if m('Anchor', d) > 0 and m('Anchor', d) + m('Exch', d) > 0]
    mac('hfAnchorSharePosMin', f"{min(shares['Anchor'][i] for i in pos):.0f}")
    mac('hfAnchorSharePosMax', f"{max(shares['Anchor'][i] for i in pos):.0f}")
    mac('hfAnchorSharePosDs', 'every dataset' if len(pos) == len(DS) else phrase([DS[i][1] for i in pos]))
    out_ = [DS[i][1] for i in range(len(DS)) if i not in pos]
    mac('hfAnchorShareScope', '' if not out_ else ' except on ' + phrase(out_))   # leading space, prose: XRE\hfAnchorShareScope,
    mac('hfNDs', WORDS[len(DS)])
    mac('hfDsPhrase', phrase([D for _, D in DS]))
    MAC.write_text('\n'.join(out) + '\n', encoding='utf-8', newline='\n')

    # ---------------- cross-checks
    pc, xm = read_macros(TABLES / 'macros_paired.tex'), read_macros(TABLES / 'macros_xmask.tex')
    pc.update({f'mpCIR{D}': v for D in ('Baby', 'Pantry', 'Scientific', 'Sports')
               for v in [read_macros(TABLES / 'macros_m1probe.tex').get(f'mpCIR{D}')] if v})
    pc.update({f'eCIR{D}': v for D in ('Baby', 'Pantry', 'Scientific', 'Sports')
               for v in [read_macros(TABLES / 'macros_e1.tex').get(f'eCIR{D}')] if v})
    hf = read_macros(MAC)
    bad = []
    for name, ref in (('XreVsBpr', 'pcBprCI'), ('List', 'pcDnsCI'), ('Exch', 'pcAoneCI'), ('Anchor', 'pcAoneNativeCI'),
                      ('MaskX', 'pcAblMaskedCI'), ('MaskA', 'pcAblFullxreCI'), ('Probe', 'mpCIR'), ('MaskDns', 'eCIR')):
        for d, D in DSC[name]:
            a, b = hf[f'hf{name}CI{D}'], pc.get(f'{ref}{D}')
            print(f'check {name:6s} {D:10s} hf {a:28s} {ref}{D} {b}')
            if a != b:
                bad.append(f'{name} {D}: hf {a} vs {ref}{D} {b}')
    abl = {(r['contrast'], r['dataset']): float(r['p']) for r in csv.DictReader(open(SRC['abl'], encoding='utf-8'))}
    for name, c in (('MaskX', 'masked'), ('MaskA', 'fullxre')):
        for d, D in DS:
            p, q = res[(name, d)]['p'], abl.get((c, d))
            print(f'check p {name:6s} {D:10s} {p:.4g} vs ablation_tests {q}')
            if q is None or abs(p - q) > 1e-3 * max(q, 1e-6):
                bad.append(f'p {name} {D}: {p:.4g} vs ablation_tests.csv {q}')
    for k in ('Min', 'Max'):
        a, b = hf[f'hfDnsShare{k}'], xm.get(f'xmDnsShare{k}')
        print(f'check DNS share {k}: hf {a} vs xm {b}')
        if a != b:
            bad.append(f'DNS share {k}: hf {a} vs xmDnsShare{k} {b}')
    # R0110: qualitative statements of the prose that combine macros (lines 'forest:' of tables/claims_check.txt)
    cc = TABLES / 'claims_check.txt'
    keep = [x for x in (cc.read_text(encoding='utf-8').splitlines() if cc.exists() else []) if not x.startswith('forest:')]
    keep += [f"forest: c) 'On {xm.get('xmExchAboveXreDs')}, both exchange-only rows exceed XRE ... Fig. 4 row (1) (A - MSCA) below zero' needs "
             f"hfAnchorDsDown == xmExchAboveXreDs: {hf['hfAnchorDsDown']} == {xm.get('xmExchAboveXreDs')} -> "
             f"{hf['hfAnchorDsDown'] == xm.get('xmExchAboveXreDs')}",
             f"forest: g) XRE - XRE on full view (Table S4; old forest row 7): intervals above zero on {hf['hfMaskADsUp']}, below zero on {hf['hfMaskADsDown']}",
             f"forest: abstract/intro/conclusion masking with the anchor term: XRE - XRE on full view up {hf['hfMaskANUp']} of {hf['hfMaskANDs']}, "
             f"masked - full-view DNS up {hf['hfMaskDnsNUp']} of {hf['hfMaskDnsNDs']}",
             f"forest: abstract S7 / (ii) / conclusion 'Holm-significant gain over the anchor term alone on xmSigAone' and Fig. 4 row (2) (XRE - A) "
             f"interval above zero: {xm.get('xmSigAone')} / {hf['hfExchDsUp']}",
             f"forest: g) exchange-only masked - full view (Table III; old forest row 6) interval above zero on {hf['hfMaskXDsUp']}; Sports mean {hf['hfMaskXMeanSports']}"]
    cc.write_text('\n'.join(keep) + '\n', encoding='utf-8', newline='\n')
    print('wrote', OUT, len(rows), 'rows and', MAC)
    for (name, d), s in res.items():
        print(f"{name:6s} {d:10s} mean {1e3 * s['mean']:+.2f} CI [{1e3 * s['ci_lo']:+.2f}, {1e3 * s['ci_hi']:+.2f}] "
              f"pos {s['pos']}/5 p {s['p']:.4f} excl {s['ci_excl_zero']}")
    print('shares (%):', {k: [round(x, 1) for x in v] for k, v in shares.items()})
    if bad:
        print('MISMATCH:', *bad, sep='\n  ')
        sys.exit(1)
    print('all cross-checks pass')


if __name__ == '__main__':
    main()
