"""Figure 3 (worked example of the entry pair of MES, complementary to Fig. 2). Schematic only, it reads no data.

Rewritten 2026-10-08 for the method change XRE -> MES (old script: figures/src/_old/fig_method_v2_pre1009.py); the
retention pair and the challenger C_u are gone.
One user u with history H_u = {i1, i2, p, r}. The fold f holds out p and r, so the visible history is V_u = {i1, i2}.
Four ranked lists of the same user over the items outside V_u (rank 1 on top, cutoff after rank K = 4), left to right:
 1 full graph G: p and r send their own messages into e_u and sit inside the top-K, the test target y sits below.
 2 masked graph G_f: the messages of p and r are removed, p drops below the cutoff (scored like y), r stays at rank K.
 3 entry pair: p (rank 5) and the displaced item D_u = i5 (rank 3) trade places across the cutoff (target order).
 4 search for D_u: upward from rank K, skipping the held-out r, so D_u is the lowest-ranked item of L_u not held out.
It does not repeat Fig. 2 (pipeline, backbone, loss sum). Notation follows text/3_method/3b_model.tex.
Usage: python -s figures/src/fig_method_v2.py
"""
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from style import C_CTRL, C_GREY, C_OURS, COL, plt, save, setup

FS, FS2 = 7, 6.5             # main and smallest font size (pt)
INK = '#333333'
POS_FC = '#d6e2f2'           # held-out / supervised positives (C_OURS family)
NEG_FC = '#f4dad9'           # D_u (C_CTRL family)
TGT_FC = '#e6e6e6'           # test target y
H, Y0 = 1.98, 0.11           # canvas top and bottom (in)
K = 4
RH, GAP = 0.135, 0.05        # row height and gap at the cutoff (in)
TOP = 1.56                   # top edge of rank 1
CW = 0.825                   # column pitch
X0 = 0.17                    # left edge of column 1 (rank numbers to its left)
CL, CR = 0.15, 0.55          # cell left / right offset inside a column

FULL = [('p', 'pos'), ('r', 'pos'), ('i_3', ''), ('i_4', ''),
        ('i_5', ''), ('y', 'tgt'), ('i_6', ''), ('i_7', '')]
MASK = [('i_3', ''), ('i_4', ''), ('i_5', ''), ('r', 'pos'),
        ('p', 'pos'), ('i_6', ''), ('y', 'tgt'), ('i_7', '')]


def rc(k):
    """Vertical centre of rank k (1-based)."""
    return TOP - (k - 0.5) * RH - (GAP if k > K else 0)


def arrow(ax, p0, p1, color=INK, lw=0.7, ms=5, ls='-', rad=0.0, z=5):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=ms, color=color, lw=lw, ls=ls,
                                 connectionstyle=f'arc3,rad={rad}', shrinkA=0, shrinkB=0, zorder=z))


def circled(ax, x, y, n):
    ax.plot(x, y, 'o', ms=8.2, mfc=INK, mec=INK, zorder=6)
    ax.text(x, y - 0.004, str(n), ha='center', va='center', fontsize=FS2, color='white', fontweight='bold', zorder=7)


def chip(ax, x, y, lab, kind, dashed=False):
    w, h = 0.17, 0.125
    fc = POS_FC if kind == 'pos' else '#f2f2f2'
    ec = C_CTRL if dashed else (C_OURS if kind == 'pos' else INK)
    ax.add_patch(FancyBboxPatch((x, y - h / 2), w, h, boxstyle='round,pad=0,rounding_size=0.025',
                                fc='white' if dashed else fc, ec=ec, lw=0.7, ls=(0, (2, 1.2)) if dashed else '-'))
    ax.text(x + w / 2, y, f'${lab}$', ha='center', va='center', fontsize=FS, color=C_GREY if dashed else INK)


def column(ax, c, items, roles=None, ext=0.03):
    """Ranked list of column c; roles maps a rank to 'neg' (D_u) to override the fill."""
    roles = roles or {}
    x = X0 + c * CW
    for k, (lab, kind) in enumerate(items, start=1):
        kind = roles.get(k, kind)
        fc = {'pos': POS_FC, 'neg': NEG_FC, 'tgt': TGT_FC}.get(kind, 'white')
        ec = {'pos': C_OURS, 'neg': C_CTRL}.get(kind, C_GREY)
        ax.add_patch(Rectangle((x + CL, rc(k) - RH / 2), CR - CL, RH, fc=fc, ec=ec, lw=0.6 if kind else 0.45,
                               zorder=3 if kind else 2))
        box = None
        if kind == 'neg':        # light hatch keeps D_u apart from the positives in grayscale
            ax.add_patch(Rectangle((x + CL, rc(k) - RH / 2), CR - CL, RH, fill=False, ec='#dfa3a1', lw=0,
                                   hatch='//////', zorder=3))
            box = dict(boxstyle='square,pad=0.08', fc=NEG_FC, ec='none')
        ax.text(x + (CL + CR) / 2, rc(k), f'${lab}$', ha='center', va='center', fontsize=FS, zorder=4, bbox=box)
    ycut = TOP - K * RH - GAP / 2
    ax.plot([x + CL - ext, x + CR + 0.03], [ycut, ycut], color=INK, lw=0.9, ls=(0, (3, 1.6)), zorder=4)
    return x


def notes(ax, c, l1, l2):
    xc = X0 + c * CW + (CL + CR) / 2
    yb = rc(8) - RH / 2 - 0.05
    ax.text(xc, yb, l1, ha='center', va='top', fontsize=FS)
    ax.text(xc, yb - 0.125, l2, ha='center', va='top', fontsize=FS)


def title(ax, c, n, text):
    x = X0 + c * CW + CL - 0.06
    circled(ax, x, 1.90, n)
    ax.text(x + 0.08, 1.90, text, ha='left', va='center', fontsize=FS)


def main():
    setup()
    plt.rcParams['hatch.linewidth'] = 0.5
    plt.rcParams['mathtext.fontset'] = 'stix'
    w = COL - 2 * plt.rcParams['savefig.pad_inches']
    fig = plt.figure(figsize=(w, H - Y0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(Y0, H)
    ax.axis('off')

    # rank numbers, shared by the four lists
    ax.text(0.085, TOP + 0.06, 'rank', ha='center', va='bottom', fontsize=FS2, color=C_GREY)
    for k in range(1, 9):
        ax.text(0.085, rc(k), str(k), ha='center', va='center', fontsize=FS2, color=C_GREY)

    ych = 1.725                                          # row of history chips / subtitles
    # ---- 1 full view ------------------------------------------------------------------------------------------------
    title(ax, 0, 1, 'Full graph $G$')
    x = column(ax, 0, FULL)
    for m, (lab, kind) in enumerate([('i_1', ''), ('i_2', ''), ('p', 'pos'), ('r', 'pos')]):
        chip(ax, x + 0.03 + m * 0.19, ych, lab, kind)
    notes(ax, 0, '$p$, $r$ lifted by', 'own messages')

    # ---- 2 masked view ----------------------------------------------------------------------------------------------
    title(ax, 1, 2, 'Masked $G_f$')
    x1 = column(ax, 1, MASK)
    for m, (lab, kind) in enumerate([('i_1', ''), ('i_2', ''), ('p', 'pos'), ('r', 'pos')]):
        chip(ax, x1 + 0.03 + m * 0.19, ych, lab, kind, dashed=m >= 2)
    arrow(ax, (x + CR + 0.02, rc(1)), (x1 + CL - 0.02, rc(5)), color=C_OURS, lw=0.7, ms=4.5)
    notes(ax, 1, '$p$ drops below', 'the cutoff, like $y$')

    # ---- 3 entry pair -----------------------------------------------------------------------------------------------
    title(ax, 2, 3, 'Entry pair')
    ax.text(X0 + 2 * CW + (CL + CR) / 2 + 0.03, ych, 'held out: $p$, $r$', ha='center', va='center', fontsize=FS2,
            color=C_GREY)
    x2 = column(ax, 2, MASK, roles={3: 'neg'}, ext=0.20)   # cutoff drawn under the swap arcs
    xl = x2 + CL - 0.05
    # entry pair: p up into the place of D_u (outer, solid), D_u down into the place of p (inner, dashed)
    arrow(ax, (xl, rc(5) - 0.03), (xl, rc(3) + 0.03), color=C_OURS, lw=0.9, ms=5, rad=-0.8)
    arrow(ax, (xl, rc(3) - 0.025), (xl, rc(5) + 0.025), color=C_CTRL, lw=0.9, ms=5, rad=-0.5, ls=(0, (2.2, 1.2)))
    xr = x2 + CR + 0.025
    ax.text(xr, rc(3), '$D_u$', ha='left', va='center', fontsize=FS, color=C_CTRL)
    notes(ax, 2, 'target order:', '$p$ above $D_u$')

    # ---- 4 search for the displaced item ----------------------------------------------------------------------------
    title(ax, 3, 4, 'Search for $D_u$')
    ax.text(X0 + 3 * CW + (CL + CR) / 2 + 0.03, ych, 'upward from rank $K$', ha='center', va='center',
            fontsize=FS2, color=C_GREY)
    x3 = column(ax, 3, MASK, roles={3: 'neg'})
    xl = x3 + CL - 0.015
    arrow(ax, (xl, rc(4)), (xl, rc(3)), color=INK, lw=0.6, ms=4, rad=-0.9)     # search for D_u: up from rank K
    xr = x3 + CR + 0.025
    ax.text(xr, rc(3), '$D_u$', ha='left', va='center', fontsize=FS, color=C_CTRL)
    ax.text(xr, rc(4), 'skip', ha='left', va='center', fontsize=FS2, color=C_GREY, style='italic')
    notes(ax, 3, '$D_u$ skips the', 'held-out $r$')

    save(fig, 'fig_method_v2', [])


if __name__ == '__main__':
    main()
