"""Figure 1 (teaser B, page width), thesis-B version.  [added at revision log 23; replaces fig_teaser.py in the main text]

Panel (a)  fusion F lifts every model: Recall@20 of every compared model without (x) and with (y) the training-free fused
           baseline F, one dot per model-dataset pair (50 pairs, 8 datasets), MSCA + MES as stars.
           Source: tables/src/data/fusion_summary_9ds_mes.csv.
           Seeds 1-5 per pair, except COHESION Scientific (3), LGMRec Pantry (3), Scientific (2), Sports (4).
Panel (b)  significant cells under the protocol (two-sided paired t over seeds 1-5, Holm over datasets of each contrast):
           top: plug-in objectives against their own backbone (5 plug-ins x 3 backbones x 4 datasets = 60 cells; up / not
           significant / down) and MES against MSCA without F (4 datasets) and with F (6 datasets).
           Sources: tables/src/data/classic_plugincount.csv (cross-checked with tables/macros_plugincount.tex),
           tables/sigtests_counts.csv family 'table1' contrast 'MSCA: Native - MES' (macros_sig \\sgUnfNSig),
           tables/src/data/fusion_test.json family_all (macros_fusiontest \\ftNSig), recomputed from fusion_cells_9ds_mes.csv.
           bottom: relative gain of the plug-in over its backbone, as reported vs with the released code (test-selected
           combination of the released grid, seed 999), from tables/appendix_official_repro_cells.csv (the generator CSV of
           Table official_repro; Baby; MG R@5 and MDVT R@10 on FREEDOM, DA-MRS R@20 on LightGCN).
Panel (c)  case MES: Recall@20 difference MSCA + MES minus MSCA per dataset with 95% CI of the paired t-test over seeds 1-5:
           without F (4 datasets, sigtests_counts.csv) and after F (6 datasets, recomputed from fusion_cells_9ds_mes.csv,
           identical to fusion_test.json: asserted). Filled = significant after Holm.
revision log 31 (MSCA10, preregistered ten-seed extension): the after-F and after-mix rows of (b) and series of (c) read
           tables/src/data/msca10_margins.csv (build_macros_msca10.py): seeds 1-10, four tuned datasets, Holm over these four; Instruments
           and Clothing (auxiliary family) appear in the tables only.  (b) adds the row of MES against MSCA without F at ten seeds
           (fusion run set, post hoc for the unfused contrast).  The unfused row and series at seeds 1-5 (Table IV) are unchanged.
revision log 29 (items 1 and 4): panel (a) reads the nine-dataset F universe of tables/src/data/rev2_pairs.csv (build_rev2.py: CLASSIC
           cells, 68 pairs incl. MicroLens) and adds the mix-fused mean of every pair as an open marker; panel (b) adds the row of MES
           against MSCA after the mix and panel (c) its per-dataset margin (tables/src/data/rev2_margins.csv, Holm over six datasets).
           The after-F series of (b) and (c) still come from fusion_test.json (item 3 of revision log 29 will update them).
Usage: python -s figures/src/fig_teaser_b.py   (writes figures/fig_teaser_b.pdf, .png preview, .sources.json)
"""
import csv
import json
import re

import numpy as np
from scipy import stats
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from style import C_CTRL, C_GREY, C_OURS, DCOL, FIG, ROOT, plt, save, setup

DATA = ROOT / 'tables' / 'src' / 'data'
TAB = ROOT / 'tables'
SRC_F9 = DATA / 'fusion_summary_9ds_mes.csv'
SRC_PC = DATA / 'classic_plugincount.csv'
SRC_PCM = TAB / 'macros_plugincount.tex'
SRC_SIG = TAB / 'sigtests_counts.csv'
SRC_SIGM = TAB / 'macros_sig.tex'
SRC_FT = DATA / 'fusion_test.json'
SRC_FC = DATA / 'fusion_cells_9ds_mes.csv'
SRC_OR = TAB / 'appendix_official_repro_cells.csv'
SRC_RVP = DATA / 'rev2_pairs.csv'       # revision log 29
SRC_RVM = DATA / 'rev2_margins.csv'     # revision log 29
SRC_RVT = TAB / 'macros_rev2.tex'       # revision log 29
SRC_MX = DATA / 'msca10_margins.csv'    # revision log 31
SRC_MXM = TAB / 'macros_msca10.tex'     # revision log 31
TUNED = ['baby', 'pantry', 'scientific', 'sports']
C_MIX = '#6a3d9a'
FS = 7.0
DSN = {'baby': 'Baby', 'pantry': 'Pantry', 'scientific': 'Scientific', 'sports': 'Sports', 'clothing': 'Clothing',
       'instruments': 'Instruments', 'arts': 'Arts', 'office': 'Office', 'microlens': 'MicroLens'}
WORD = {w: i for i, w in enumerate(['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'])}
WORD['none'] = 0
C_UP = '#2e7d32'
C_DOWN = C_CTRL
C_NS = '#d0d0d0'
TCRIT = stats.t.ppf(0.975, 4)


def macro(path, name):
    m = re.search(r'\\newcommand\{\\' + name + r'\}\{([^}]*)\}', path.read_text(encoding='utf-8'))
    v = m.group(1)
    return WORD[v] if v in WORD else int(v)


def mix_means():
    import sys
    sys.path.insert(0, str(TAB / 'src'))
    import build_rev2 as rv
    out = {}
    for (m, d, s), v in rv.bc.cell.items():
        if 'mix' in v and 'combo' in v and 'base' in v:
            out.setdefault((m, d), []).append(v['mix'])
    return {k: float(np.mean(v)) for k, v in out.items()}


def panel_a(ax):
    rows = list(csv.DictReader(open(SRC_RVP, encoding='utf-8')))
    base = np.array([float(r['base']) for r in rows])
    fused = np.array([float(r['F']) for r in rows])
    mixd = mix_means()
    mixv = np.array([mixd[(r['model'], r['dataset'])] for r in rows])
    mes = np.array([r['model'] == 'MSCA+MES' for r in rows])
    assert len(rows) == macro(SRC_RVT, 'rvFNPairs') and (fused > base).all(), 'F must lift every pair'
    lo, hi = 0.06, 0.16
    ax.plot([lo, hi], [lo, hi], color=C_GREY, lw=0.7, ls='--', zorder=1)
    ax.scatter(base, mixv, s=9, facecolors='none', edgecolors=C_MIX, lw=0.5, alpha=0.9, zorder=2)
    ax.scatter(base[~mes], fused[~mes], s=11, color=C_GREY, alpha=0.8, lw=0, zorder=2)
    ax.scatter(base[mes], fused[mes], s=34, marker='*', color=C_OURS, lw=0, zorder=3)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ticks = [0.06, 0.10, 0.14]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(['.06', '.10', '.14'])
    ax.set_yticklabels(['.06', '.10', '.14'])
    ax.set_xlabel('Recall@20 without F', labelpad=1)
    ax.set_ylabel('Recall@20 with F or mix', labelpad=1)
    ax.spines[['top', 'right']].set_visible(False)
    ax.text(0.97, 0.03, f'F above the dashed line:\n{int((fused > base).sum())} of {len(rows)} pairs', ha='right', va='bottom',
            transform=ax.transAxes, fontsize=FS)
    h = [Line2D([], [], marker='*', color=C_OURS, ls='', ms=7, label='MSCA + MES'),
         Line2D([], [], marker='o', color=C_GREY, ls='', ms=3.2, label='other models'),
         Line2D([], [], marker='o', color=C_MIX, mfc='none', ls='', ms=3.4, mew=0.6, label='mix instead of F')]
    ax.legend(handles=h, loc='upper center', bbox_to_anchor=(0.45, -0.19), ncol=2, frameon=False, fontsize=FS,
              handletextpad=0.2, columnspacing=0.6, labelspacing=0.25)
    ax.set_title('(a) F and the mix lift every model', fontsize=FS + 0.5, pad=3)


def counts():
    """Significant up / not significant / down counts for the rows of panel (b)."""
    pc = list(csv.DictReader(open(SRC_PC, encoding='utf-8')))
    assert len(pc) == macro(SRC_PCM, 'pcNCells') == 60
    up = sum(r['reject'] == '1' and float(r['diff']) > 0 for r in pc)
    dn = sum(r['reject'] == '1' and float(r['diff']) < 0 for r in pc)
    assert (up, dn) == (macro(SRC_PCM, 'pcNUp'), macro(SRC_PCM, 'pcNDown')), (up, dn)
    plug = (up, len(pc) - up - dn, dn)
    unf = unfused_rows()
    n_unf = sum(r['reject'] for r in unf)
    assert n_unf == macro(SRC_SIGM, 'sgUnfNSig') and len(unf) == macro(SRC_SIGM, 'sgUnfNDs')
    mes_unf = (n_unf, len(unf) - n_unf, 0)
    # revision log 31: ten seeds, four tuned datasets
    unf10 = mx_rows('base')
    n10 = sum(r['reject'] and r['diff'] > 0 for r in unf10)
    assert n10 == macro(SRC_MXM, 'xnUnfNSig') and len(unf10) == macro(SRC_MXM, 'xnUnfNDs') == 4
    mes_unf10 = (n10, len(unf10) - n10, 0)
    post = mx_rows('F')
    n_post = sum(r['reject'] for r in post)
    assert n_post == macro(SRC_MXM, 'xnFNSig') == 0 and len(post) == macro(SRC_MXM, 'xnNDs')
    mes_post = (n_post, len(post) - n_post, 0)
    mix = mx_rows('mix')
    mes_mix = (sum(r['reject'] and r['diff'] > 0 for r in mix), sum(not r['reject'] for r in mix), sum(r['reject'] and r['diff'] < 0 for r in mix))
    assert mes_mix[0] == macro(SRC_MXM, 'xnMixNSigPos') == 0 and mes_mix[2] == macro(SRC_MXM, 'xnMixNSigNeg')
    return plug, mes_unf, mes_post, unf, post, mes_mix, mix, mes_unf10


def mx_rows(fus):
    """revision log 31: MSCA + MES minus MSCA over seeds 1-10 on the four tuned datasets (msca10_margins.csv, Holm over these four)"""
    out = []
    for r in csv.DictReader(open(SRC_MX, encoding='utf-8')):
        if r['fusion'] == fus and r['metric'] == 'r20' and r['family'] == 'tuned':
            assert r['n'] == '10' and r['window'] == '1-10'
            out.append(dict(ds=r['dataset'], diff=float(r['diff']), lo=float(r['ci_lo']), hi=float(r['ci_hi']), reject=r['reject'] == '1'))
    assert [o['ds'] for o in out] == TUNED
    return out


def unfused_rows():
    """MSCA + MES minus MSCA without F, four datasets of Table I (sigtests_counts.csv stores Native - MES)."""
    out = []
    for r in csv.DictReader(open(SRC_SIG, encoding='utf-8')):
        if r['family'].startswith('table1') and r['contrast'] == 'MSCA: Native - MES' and r['metric'] == 'r20':
            out.append(dict(ds=r['dataset'], diff=-float(r['diff']), lo=-float(r['ci_hi']), hi=-float(r['ci_lo']),
                            reject=int(r['reject']) == 1 and float(r['diff']) < 0))
    assert [o['ds'] for o in out] == ['baby', 'pantry', 'scientific', 'sports']
    return out


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


def panel_b(fig, spec):
    plug, mes_unf, mes_post, _, _, mes_mix, _, mes_unf10 = counts()
    sub = spec.subgridspec(2, 1, height_ratios=[1.8, 0.9], hspace=1.3)
    ax = fig.add_subplot(sub[0])
    rows = [('Plug-ins vs. own\nbackbone (60 cells)', plug), ('MES, no F, 5 seeds', mes_unf),
            ('MES, no F, 10 seeds', mes_unf10),
            ('MES, after F, 10 seeds', mes_post), ('MES, after mix, 10 seeds', mes_mix)]
    for i, (lab, (u, n, d)) in enumerate(rows):
        y = 4 - i
        tot = u + n + d
        x0 = 0.0
        for v, col in ((d, C_DOWN), (n, C_NS), (u, C_UP)):
            if v:
                ax.barh(y, 100 * v / tot, left=x0, height=0.62, color=col, ec='white', lw=0.5)
                if v / tot >= 0.12:
                    ax.text(x0 + 50 * v / tot, y, str(v), ha='center', va='center', fontsize=FS,
                            color='white' if col != C_NS else 'black')
            x0 += 100 * v / tot
    # small counts that do not fit inside the bar are written to the right of it
    ax.text(101.5, 4, f'{plug[0]} up, {plug[2]} down', va='center', ha='left', fontsize=FS)
    ax.text(101.5, 3, f'{mes_unf[0]} up', va='center', ha='left', fontsize=FS)
    ax.text(101.5, 2, f'{mes_unf10[0]} up', va='center', ha='left', fontsize=FS)
    ax.text(101.5, 1, f'{mes_post[0]} up', va='center', ha='left', fontsize=FS)
    ax.text(101.5, 0, f'{mes_mix[0]} up, {mes_mix[2]} down', va='center', ha='left', fontsize=FS)
    ax.set_yticks([4, 3, 2, 1, 0])
    ax.set_yticklabels([r[0] for r in rows], fontsize=FS)
    ax.set_ylim(-0.55, 4.55)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 50, 100])
    ax.set_xticklabels(['0', '50', '100'])
    ax.set_xlabel('Share of cells', labelpad=1)
    ax.spines[['top', 'right']].set_visible(False)
    h = [Patch(fc=C_UP, label='significantly up'), Patch(fc=C_NS, label='not significant'),
         Patch(fc=C_DOWN, label='significantly down')]
    ax.legend(handles=h, loc='upper center', bbox_to_anchor=(0.30, -0.30), ncol=3, frameon=False, fontsize=FS,
              handletextpad=0.3, columnspacing=0.8, handlelength=1.0)
    ax.set_title('(b) Significant cells under the protocol', fontsize=FS + 0.5, pad=3)

    # reported vs reproduced plug-in gains
    ax2 = fig.add_subplot(sub[1])
    cells = [r for r in csv.DictReader(open(SRC_OR, encoding='utf-8')) if r['seed'] == '999']
    lab = {'MG': 'MG (R@5)', 'MDVT-D': 'MDVT (R@10)', 'DA-MRS': 'DA-MRS (R@20)'}
    assert [c['plugin'] for c in cells] == ['MG', 'MDVT-D', 'DA-MRS']
    names, rep, rer = [], [], []
    for c in cells:
        names.append(lab[c['plugin']])
        rep.append(100 * (float(c['reported_plugin']) / float(c['reported_native']) - 1))
        rer.append(100 * (float(c['testsel_plugin']) / float(c['testsel_native']) - 1))
    for i, (a, b) in enumerate(zip(rep, rer)):
        y = 2 - i
        ax2.plot([b, a], [y, y], color='#bbbbbb', lw=0.8, zorder=1)
        ax2.scatter([a], [y], s=22, color='white', ec=C_OURS, lw=1.0, zorder=3)
        ax2.scatter([b], [y], s=22, color=C_OURS, lw=0, zorder=3)
    ax2.set_title('Plug-in gain, reported vs. released code', fontsize=FS, pad=2)
    ax2.set_yticks([2, 1, 0])
    ax2.set_yticklabels(names, fontsize=FS)
    ax2.set_ylim(-0.55, 2.55)
    ax2.set_xlim(-2, 40)
    ax2.set_xticks([0, 10, 20, 30, 40])
    ax2.set_xlabel('Gain over backbone (%)', labelpad=1)
    ax2.spines[['top', 'right']].set_visible(False)
    h2 = [Line2D([], [], marker='o', color=C_OURS, mfc='white', ls='', ms=4.5, label='reported'),
          Line2D([], [], marker='o', color=C_OURS, ls='', ms=4.5, label='released code')]
    ax2.legend(handles=h2, loc='upper center', bbox_to_anchor=(0.40, -0.55), ncol=2, frameon=False, fontsize=FS,
               handletextpad=0.2, columnspacing=0.8)
    return rep, rer, plug, mes_unf, mes_post


def panel_c(ax):
    _, _, _, unf, post, _, mix, _ = counts()
    dss = [p['ds'] for p in post]
    assert dss == TUNED == [m['ds'] for m in mix] == [u['ds'] for u in unf]
    ud = {u['ds']: u for u in unf}
    off = {'pre': 0.24, 'post': 0.0, 'mix': -0.24}
    for i, ds in enumerate(dss):
        y = len(dss) - 1 - i
        for key, row, col in (('pre', ud.get(ds), C_OURS), ('post', post[i], C_CTRL), ('mix', mix[i], C_MIX)):
            if row is None:
                continue
            yy = y + off[key]
            ax.plot([1e4 * row['lo'], 1e4 * row['hi']], [yy, yy], color=col, lw=1.0, zorder=2)
            ax.scatter([1e4 * row['diff']], [yy], s=20, zorder=3, color=col if row['reject'] else 'white', ec=col, lw=0.9)
    ax.axvline(0, color='black', lw=0.5, ls=':', zorder=0)
    ax.set_yticks(range(len(dss)))
    ax.set_yticklabels([DSN[d] for d in dss][::-1], fontsize=FS)
    ax.set_ylim(-0.6, len(dss) - 0.4)
    ax.set_xlim(-32, 50)
    assert all(-32 < 1e4 * r['lo'] and 1e4 * r['hi'] < 50 for r in mix), 'mix interval outside the axis'
    ax.set_xticks([-20, 0, 20, 40])
    ax.set_xlabel(r'MES $-$ MSCA, Recall@20 ($\times10^{-4}$)', labelpad=1)
    ax.spines[['top', 'right']].set_visible(False)
    n_u, n_p = sum(u['reject'] for u in unf), sum(p['reject'] for p in post)
    n_mu, n_md = sum(m['reject'] and m['diff'] > 0 for m in mix), sum(m['reject'] and m['diff'] < 0 for m in mix)
    h = [Line2D([], [], marker='o', color=C_OURS, ls='-', ms=4.5, label=f'without F, 5 seeds: {n_u} of {len(unf)} significant'),
         Line2D([], [], marker='o', color=C_CTRL, ls='-', ms=4.5, label=f'after F, 10 seeds: {n_p} of {len(post)} significant'),
         Line2D([], [], marker='o', color=C_MIX, ls='-', ms=4.5, label=f'after mix, 10 seeds: {n_mu} up, {n_md} down of {len(mix)}'),
         Line2D([], [], marker='o', color='black', mfc='white', ls='', ms=4.5, label='open: not significant')]
    assert (n_u, n_p) == (2, 0) and n_md == 1
    ax.legend(handles=h, loc='upper center', bbox_to_anchor=(0.30, -0.19), ncol=1, frameon=False, fontsize=FS,
              handletextpad=0.3, labelspacing=0.2)
    ax.set_title('(c) Case MES: margin before and after fusion', fontsize=FS + 0.5, pad=3)
    return unf, post


def main():
    setup()
    plt.rcParams.update({'mathtext.fontset': 'stix', 'font.size': FS, 'axes.labelsize': FS + 0.5,
                         'xtick.labelsize': FS, 'ytick.labelsize': FS, 'legend.fontsize': FS, 'axes.titlesize': FS + 0.5})
    fig = plt.figure(figsize=(DCOL, 3.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.95, 1.2, 1.05], wspace=0.62, left=0.058, right=0.995, top=0.91, bottom=0.30)
    panel_a(fig.add_subplot(gs[0]))
    rep, rer, plug, mu, mp = panel_b(fig, gs[1])
    unf, post = panel_c(fig.add_subplot(gs[2]))
    print('plug-ins up/ns/down:', plug, '; MES unfused:', mu, '; MES fused:', mp)
    print('reported vs released gain %:', [round(x, 1) for x in rep], [round(x, 1) for x in rer])
    print('unfused:', [(u['ds'], round(1e4 * u['diff'], 1), u['reject']) for u in unf])
    print('fused  :', [(p['ds'], round(1e4 * p['diff'], 1), round(1e4 * p['lo'], 1), round(1e4 * p['hi'], 1), p['reject']) for p in post])
    save(fig, 'fig_teaser_b', [SRC_RVP, SRC_RVT, SRC_PC, SRC_PCM, SRC_SIG, SRC_SIGM, SRC_MX, SRC_MXM, SRC_OR])
    fig.savefig(FIG / 'fig_teaser_b.png', dpi=450)


if __name__ == '__main__':
    main()
