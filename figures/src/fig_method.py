"""Figure 2 (method overview of MES, masked entry supervision). Schematic only, it reads no result data.

Rewritten 2026-10-08 for the method change XRE -> MES (old script: figures/src/_old/fig_method_pre1009.py).
Top branch (full view): full graph G -> backbone F_theta -> backbone objective L_B (BPR on (u,p,n) + R_b); inference
also ranks on G. Bottom branch (masked view at step t): masked graph G_f without the training edges of fold
f = t mod 2 -> the same backbone (shared theta) -> masked ranking of the items outside the visible history V_u, whose
top K rows (bracket) form the masked list L_u. Entry pair: held-out positive p outside L_u versus the displaced item
D_u (lowest-ranked item of L_u that is not held out), L_E = softplus(s_check(u,D_u) - s_check(u,p)).
Middle glyph: on the full graph the score s(u,p) of a positive contains its own message (self-message, hatched), the
masked score s_check(u,p) does not (dashed empty slot). Both branches feed L = L_B + lambda L_E.
No anchor term, no retention pairs (ablations only). Notation follows text/3_method/3b_model.tex and the rev1009 brief.
Usage: python -s figures/src/fig_method.py
"""
import matplotlib._mathtext as mathtext
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from style import C_CTRL, C_GREY, C_OURS, COL, plt, save, setup

# matplotlib 3.8 mathtext lacks \check as an accent (it prints a spacing caron before the letter); map it to the
# combining caron U+030C so that $\check{s}$ renders as in the manuscript.
mathtext.Parser._accent_map.setdefault('check', r'\check')

H, Y0 = 2.21, 0.10            # top and bottom of the canvas (in); data units are inches
FS, FS2 = 7, 6.5              # main and smallest font size (pt)
INK = '#333333'
POS_FC = '#d6e2f2'            # light fill of held-out positives and of L_E (C_OURS family)
NEG_FC = '#f4dad9'            # light fill of D_u (C_CTRL family)
BOX_FC = '#f2f2f2'


def box(ax, x, y, w, h, text, fc=BOX_FC, ec=INK, lw=0.6, fs=FS, **kw):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.04', fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, **kw)


def arrow(ax, p0, p1, color=INK, lw=0.7, ms=6):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=ms, color=color, lw=lw,
                                 shrinkA=0, shrinkB=0, zorder=5))


def push(ax, x, y, up, color):
    """Short vertical arrow inside a list row: the order that the entry pair favors."""
    d = 0.034
    arrow(ax, (x, y - d if up else y + d), (x, y + d if up else y - d), color=color, lw=0.9, ms=4.5)


def graph_icon(ax, x, y, masked):
    """Small bipartite user-item graph; in the masked view the edges of the held-out fold are dashed."""
    us = [(x, y + 0.27), (x, y + 0.13), (x, y - 0.01)]
    it = [(x + 0.42, y + 0.30), (x + 0.42, y + 0.18), (x + 0.42, y + 0.06), (x + 0.42, y - 0.06)]
    edges = [(0, 0, 0), (0, 2, 1), (1, 1, 0), (1, 3, 1), (2, 0, 1), (2, 2, 0), (2, 3, 0)]   # (user, item, fold)
    for u, i, fold in edges:
        (x0, y0), (x1, y1) = us[u], it[i]
        if masked and fold == 1:
            ax.plot([x0, x1], [y0, y1], color=C_CTRL, lw=0.6, ls=(0, (1.5, 1.2)))
        else:
            ax.plot([x0, x1], [y0, y1], color=INK, lw=0.6)
    for px, py in us:
        ax.plot(px, py, 'o', ms=3.4, mfc='white', mec=INK, mew=0.6)
    for px, py in it:
        ax.plot(px, py, 's', ms=3.0, mfc=C_GREY, mec=INK, mew=0.4)


def score_glyph(ax, x0, y1, y2):
    """Two horizontal score bars: s(u,p) with the hatched self-message, s_check(u,p) with an empty dashed slot."""
    bh, a, b = 0.10, 0.40, 0.26                  # bar height, other messages, self-message (in)
    for y, lab in ((y1, '$s(u,p)$'), (y2, r'$\check{s}(u,p)$')):
        ax.text(x0 - 0.04, y, lab, ha='right', va='center', fontsize=FS)
        ax.add_patch(Rectangle((x0, y - bh / 2), a, bh, fc='#e6e6e6', ec=C_GREY, lw=0.5, zorder=3))
    ax.add_patch(Rectangle((x0 + a, y1 - bh / 2), b, bh, fc=POS_FC, ec=C_OURS, lw=0.6, hatch='//////', zorder=3))
    ax.add_patch(Rectangle((x0 + a, y2 - bh / 2), b, bh, fill=False, ec=C_GREY, lw=0.5, ls=(0, (1.5, 1.2)), zorder=3))
    ax.text(x0 + a + b / 2, y1 + bh / 2 + 0.025, 'self-message of $p$', ha='center', va='bottom', fontsize=FS2, color=C_OURS)
    ax.text(x0 + a + b / 2, y2, 'none', ha='center', va='center', fontsize=FS2, color=C_GREY, style='italic')


def main():
    setup()
    plt.rcParams['mathtext.fontset'] = 'stix'          # Times-like math, matching the serif text of the manuscript
    plt.rcParams['hatch.linewidth'] = 0.5
    w = COL - 2 * plt.rcParams['savefig.pad_inches']   # the saved PDF is then one column wide
    fig = plt.figure(figsize=(w, H - Y0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(Y0, H)
    ax.axis('off')

    yt, ym = 1.96, 0.78           # centre lines of the full (top) and masked (bottom) branches
    gx, cx = 0.21, 0.42           # graph icon left edge and label centre
    # ---- inputs -------------------------------------------------------------------------------------------------------
    graph_icon(ax, gx, yt - 0.10, masked=False)
    ax.text(cx, yt - 0.25, 'Full graph $G$', ha='center', va='top', fontsize=FS)
    ax.text(cx, yt - 0.39, '(also at inference)', ha='center', va='top', fontsize=FS2, color=C_GREY)
    graph_icon(ax, gx, ym - 0.10, masked=True)
    ax.text(cx, ym - 0.25, 'Masked graph $G_f$', ha='center', va='top', fontsize=FS)
    ax.text(cx, ym - 0.39, 'fold $f=t$ mod 2', ha='center', va='top', fontsize=FS, color=C_CTRL)
    ax.text(cx, ym - 0.52, 'edges removed', ha='center', va='top', fontsize=FS, color=C_CTRL)

    # ---- shared backbone ----------------------------------------------------------------------------------------------
    bx, bw, bh = 0.76, 0.44, 0.36
    for yc in (yt, ym):
        box(ax, bx, yc - bh / 2, bw, bh, 'Backbone\n$F_\\theta$', fc='#e6e6e6', linespacing=1.15)
        arrow(ax, (gx + 0.48, yc), (bx, yc))
    ax.plot([bx + bw / 2] * 2, [ym + bh / 2, yt - bh / 2], color=INK, lw=0.7, ls=(0, (2.5, 1.5)))
    ax.text(bx + bw / 2 - 0.05, (yt + ym) / 2, 'shared $\\theta$', ha='right', va='center', fontsize=FS)

    # ---- right column: loss terms -------------------------------------------------------------------------------------
    rx, rw = 2.64, 0.80
    ytot = 1.34
    box(ax, rx, yt - 0.15, rw, 0.30, 'Backbone\nobjective $\\mathcal{L}_B$', linespacing=1.05)
    box(ax, rx, ytot - 0.20, rw, 0.40, 'MES objective\n$\\mathcal{L}_B+\\lambda\\,\\mathcal{L}_E$',
        fc='white', lw=0.9, linespacing=1.35)
    arrow(ax, (rx + rw / 2, yt - 0.15), (rx + rw / 2, ytot + 0.20))      # L_B -> total

    # full-view backbone -> L_B
    arrow(ax, (bx + bw, yt), (rx, yt))
    ax.text((bx + bw + rx) / 2, yt + 0.03, 'BPR on $(u,p,n)$ $+\\,\\mathcal{R}_b$', ha='center', va='bottom', fontsize=FS)

    # ---- self-message glyph between the branches ----------------------------------------------------------------------
    score_glyph(ax, 1.62, 1.57, 1.36)

    # ---- masked list glyph (rank 1 on top) ----------------------------------------------------------------------------
    lx, lw_, rh, K = 1.66, 0.34, 0.105, 4
    top = ym + 0.24                                          # top edge of rank 1
    gap = 0.06                                               # gap at the cutoff
    rows = [('', None), ('', None), ('', None), ('$D_u$', 'neg'), ('', None), ('$p$', 'pos')]
    yc = {}
    for k, (lab, kind) in enumerate(rows):
        y = top - (k + 1) * rh - (gap if k >= K else 0)
        fc = POS_FC if kind == 'pos' else (NEG_FC if kind == 'neg' else 'white')
        ec = C_OURS if kind == 'pos' else (C_CTRL if kind == 'neg' else C_GREY)
        ax.add_patch(Rectangle((lx, y), lw_, rh, fc=fc, ec=ec, lw=0.6, zorder=2 if kind else 1))
        if lab:
            yc[lab] = y + rh / 2
            ax.text(lx + 0.13, y + rh / 2, lab, ha='center', va='center', fontsize=FS, zorder=3)
            push(ax, lx + 0.26, y + rh / 2, kind == 'pos', C_OURS if kind == 'pos' else C_CTRL)
    ycut = top - K * rh - gap / 2
    ax.plot([lx - 0.06, lx + lw_ + 0.03], [ycut, ycut], color=INK, lw=0.9, ls=(0, (3, 1.5)), zorder=4)
    ax.text(lx - 0.09, ycut, 'cutoff', ha='right', va='center', fontsize=FS)
    ax.text(lx + lw_ / 2, top + 0.03, r'Masked ranking of $\mathcal{I}\setminus\mathcal{V}_u$', ha='center', va='bottom',
            fontsize=FS)
    xb = lx - 0.05                                           # bracket of the top-K rows: the masked list L_u
    ax.plot([xb + 0.035, xb, xb, xb + 0.035], [top, top, ycut + 0.022, ycut + 0.022], color=INK, lw=0.6)
    ax.text(xb - 0.03, top - 0.05, '$L_u$', ha='right', va='center', fontsize=FS)
    ax.text(xb - 0.03, top - 0.15, 'top $K$', ha='right', va='center', fontsize=FS2)
    arrow(ax, (bx + bw, ym), (xb - 0.01, ym))

    # entry pair p vs D_u (bracket on the right of the list) -> L_E
    xe = lx + lw_ + 0.05
    ax.plot([xe, xe + 0.05, xe + 0.05, xe], [yc['$D_u$'], yc['$D_u$'], yc['$p$'], yc['$p$']], color=INK, lw=0.6)
    ye = (yc['$D_u$'] + yc['$p$']) / 2
    ax.plot([xe + 0.05, xe + 0.08], [ye, ye], color=INK, lw=0.6)
    ax.text(xe + 0.10, ye, 'entry\npair', ha='left', va='center', fontsize=FS, linespacing=1.05)
    arrow(ax, (xe + 0.36, ye), (rx, ye))
    box(ax, rx, ye - 0.15, rw, 0.30, 'Entry loss\n$\\mathcal{L}_E$', fc=POS_FC, ec=C_OURS, linespacing=1.05)
    arrow(ax, (rx + rw / 2, ye + 0.15), (rx + rw / 2, ytot - 0.20))      # L_E -> total
    ax.text(rx + rw, ye - 0.20, r'$\mathcal{L}_E$: softplus$(\check{s}(u,D_u)-\check{s}(u,p))$', ha='right',
            va='top', fontsize=FS2)

    save(fig, 'fig_method', [])


if __name__ == '__main__':
    main()
