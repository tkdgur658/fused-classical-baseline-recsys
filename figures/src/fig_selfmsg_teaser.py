"""Figure 2(b) (former Figure 1 teaser; copied from fig_teaser.py at ⑫): the self-message of a supervised positive in a graph-based multimodal recommender, how large it is
compared with an ID-only graph recommender, and what remains of it under existing remedies.

Left (schematic, no data): Proposition prop:history. The user representation sums the messages of the history items. The
message of p reaches the training score through the collaborative channel and through the item-item views of p
(co-occurrence, image, text). A collaborative-only correction (EFLEC-style) removes the first channel only.

Right top (data): share of the training score of a positive that is its own message (sum S / sum s_up over the
training-positive sample, Proposition prop:history), for LightGCN (ID-only; its user ID term is no message), MSCA trained
with BPR (Native) and MSCA trained with a top-K loss (mean over PSL, SL@K and SSM). The MSCA bars are split into the
collaborative channel and the item-item view channels (co-occurrence, image, text) of the message.
Right bottom (data): share of training positives inside the top-40 list of their user (candidates = non-history items + p):
MSCA Native on the full graph, MSCA with a top-K loss on the full graph (mean of the three losses), MSCA Native after the
collaborative channel of the own message is removed (EFLEC-style), after the whole own message is removed (closed-form
leave-one-edge-out), and held-out test targets of the same users (Native).
Values = mean over seeds 1-5 (MSCA) or the available LightGCN seeds (Baby 1, Pantry 3, Scientific 1, Sports 5:
seeds 1-5 trained with MMRec_clean, added by mmchan_sports/add_lgcn_sports.sh). A dataset without LightGCN cells gets no LightGCN bar and an
'n/a' mark in its place instead of an error.
Sources: tables/mmchan_perseed.csv (summ.py: MSCA Native and LightGCN; LightGCN Sports from summ_sports.py) and
tables/mmchan_topk_perseed.csv (summ3.py: MSCA Native and the top-K loss checkpoints with the EFLEC-style removal; Sports
rows = summ_sports.py on the Table I Sports Native and top-K runs), collected by tables/src/collect_mmchan.py.
Usage: python -s figures/src/fig_selfmsg_teaser.py
"""
import csv
import statistics as st

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from style import C_CTRL, C_GREY, C_OURS, COL, TABLES, plt, save, setup

DATASETS = [('baby', 'Baby'), ('pantry', 'Pantry'), ('scientific', 'Scientific'), ('sports', 'Sports')]
SEEDS = [1, 2, 3, 4, 5]
TK = ['psl', 'slk', 'ssm']
INK = '#333333'
FS = 6.8
POS_FC = '#d6e2f2'
NEG_FC = '#f4dad9'
BOX_FC = '#f2f2f2'
C_VIEW = '#e8a09e'      # item-item view channels (lighter shade of C_CTRL)
C_TK = '#8c3a37'        # top-K loss bars
C_TKV = '#c99492'
C_EF = '#e39b3b'        # EFLEC-style removal
SRC1 = TABLES / 'mmchan_perseed.csv'
SRC3 = TABLES / 'mmchan_topk_perseed.csv'


def load():
    V, V3 = {}, {}
    with open(SRC1, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            V[(r['dataset'], r['model'], r['arm'], int(r['seed']), r['metric'])] = float(r['value'])
    with open(SRC3, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            V3[(r['dataset'], r['arm'], int(r['seed']), r['metric'])] = float(r['value'])
    return V, V3


def rbox(ax, x, y, w, h, text, fc=BOX_FC, ec=INK, lw=0.6, fs=FS, **kw):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.03', fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, **kw)


def arrow(ax, p0, p1, color=INK, lw=0.7, ls='-'):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=5.5, color=color, lw=lw, ls=ls,
                                 shrinkA=0, shrinkB=0))


def schematic(ax):
    """Training score of p (own message through two channels) and test score of y. Data units [0, 1] x [0, 1]."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    # training pair
    y0 = 0.60
    ax.text(0.0, y0 + 0.335, 'Training pair $(u,p)$', fontsize=FS, ha='left', va='bottom', fontweight='bold')
    for k, lab in enumerate(['$i_1$', '$i_2$']):
        rbox(ax, 0.0, y0 + 0.235 - 0.085 * k, 0.15, 0.065, lab)
        arrow(ax, (0.155, y0 + 0.267 - 0.085 * k), (0.375, y0 + 0.12))
    rbox(ax, 0.0, y0 + 0.02, 0.15, 0.08, '$p$', fc=POS_FC)
    arrow(ax, (0.155, y0 + 0.075), (0.375, y0 + 0.11), color=C_CTRL)
    arrow(ax, (0.155, y0 + 0.045), (0.375, y0 + 0.095), color=C_VIEW, lw=1.1)
    rbox(ax, 0.38, y0 + 0.06, 0.11, 0.09, '$\\mathbf{e}_u$')
    arrow(ax, (0.495, y0 + 0.105), (0.555, y0 + 0.105))
    rbox(ax, 0.56, y0 + 0.06, 0.22, 0.09, '$s(u,p)$', fc=POS_FC)
    ax.text(0.0, y0 - 0.04, 'self-message of $p$ through', fontsize=FS - 0.8, ha='left', va='center', color=INK)
    ax.text(0.0, y0 - 0.10, 'collaborative', fontsize=FS - 0.8, ha='left', va='center', color=C_CTRL)
    ax.text(0.47, y0 - 0.10, '+ item–item views', fontsize=FS - 0.8, ha='left', va='center', color='#b5605d')
    # test target
    y1 = 0.10
    ax.text(0.0, y1 + 0.235, 'Test target $y$', fontsize=FS, ha='left', va='bottom', fontweight='bold')
    for k, lab in enumerate(['$i_1$', '$i_2$', '$i_3$']):
        rbox(ax, 0.0, y1 + 0.15 - 0.085 * k, 0.15, 0.065, lab)
        arrow(ax, (0.155, y1 + 0.182 - 0.085 * k), (0.375, y1 + 0.09))
    rbox(ax, 0.38, y1 + 0.045, 0.11, 0.09, '$\\mathbf{e}_u$')
    arrow(ax, (0.495, y1 + 0.09), (0.555, y1 + 0.09))
    rbox(ax, 0.56, y1 + 0.045, 0.22, 0.09, '$s(u,y)$', fc=NEG_FC)
    ax.text(0.38, y1 - 0.03, 'other history only', fontsize=FS - 0.6, ha='left', va='center', color=C_GREY)


def share_panel(ax, V, V3):
    w = 0.26
    for j, (ds, _) in enumerate(DATASETS):
        lg = [v for k, v in V.items() if k[0] == ds and k[1] == 'LightGCN' and k[4] == 'S_share_of_s']
        if lg:
            ax.bar(j - w, st.mean(lg), w * 0.92, color=C_GREY, label='LightGCN (ID-only)' if j == 0 else None)
        else:   # no LightGCN checkpoint for this dataset yet: no bar, marked n/a
            ax.text(j - w, 0.01, 'n/a', ha='center', va='bottom', rotation=90, fontsize=FS - 2.6, color=C_GREY)
        for x, arms, c1, c2, lab in ((j, ['native'], C_CTRL, C_VIEW, 'MSCA'), (j + w, TK, C_TK, C_TKV, 'MSCA top-$K$')):
            tot = st.mean(V3[(ds, a, s, 'S_share_of_s')] for a in arms for s in SEEDS)
            col = st.mean(V3[(ds, a, s, 'S_share_of_s')] * V3[(ds, a, s, 'chS_share_collab')] for a in arms for s in SEEDS)
            # legend 2026-10-07: one entry per fill (five), so the hatched top-K item-item part has its own entry
            ax.bar(x, col, w * 0.92, color=c1, label=f'{lab}: collab.' if j == 0 else None)
            ax.bar(x, tot - col, w * 0.92, bottom=col, color=c2, hatch='////' if c2 == C_TKV else None, lw=0,
                   label=f'{lab}: item–item' if j == 0 else None)
    ax.set_xticks(range(len(DATASETS)))
    ax.set_xticklabels([])
    ax.set_ylim(0, 0.6)
    ax.set_yticks([0, 0.25, 0.5])
    ax.set_yticklabels(['0', '.25', '.50'])
    ax.set_ylabel('Self-message\n/ training score', labelpad=1, fontsize=FS - 0.6)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(loc='upper left', bbox_to_anchor=(-0.02, 1.32), ncol=2, frameon=False, handlelength=0.9,
              columnspacing=0.6, fontsize=FS - 1.4, handletextpad=0.3, labelspacing=0.15)


def top_panel(ax, V3):
    groups = [(['native'], 'full:train_top40', C_CTRL, 'MSCA, full graph'),
              (TK, 'full:train_top40', C_TK, 'MSCA top-$K$, full graph'),
              (['native'], 'rank_looEflec:train_top40', C_EF, 'MSCA, $-$collab. self-msg.'),
              (['native'], 'rank_loo:train_top40', C_OURS, 'MSCA, $-$all self-msg.'),
              (['native'], 'full:tgt_top40', C_GREY, 'Test targets')]
    w = 0.165
    for j, (ds, _) in enumerate(DATASETS):
        for k, (arms, metric, col, lab) in enumerate(groups):
            v = st.mean(V3[(ds, a, s, metric)] for a in arms for s in SEEDS)
            ax.bar(j + (k - 2) * w, v, w * 0.92, color=col, label=lab if j == 0 else None)
    ax.set_xticks(range(len(DATASETS)))
    ax.set_xticklabels([n for _, n in DATASETS], fontsize=FS - 1.8)   # four datasets in a narrow panel
    ax.tick_params(axis='x', pad=4.875)          # keeps the bottom of the figure (printed height) as with 6.5 pt labels
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(['0', '.50', '1.0'])
    ax.set_ylabel('Share inside\nthe top-40 list', labelpad=1, fontsize=FS - 0.6)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(loc='upper left', bbox_to_anchor=(-0.02, 1.36), ncol=2, frameon=False, handlelength=0.9,
              columnspacing=0.6, fontsize=FS - 1.4, handletextpad=0.3, labelspacing=0.15)


def main():
    setup()
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['hatch.linewidth'] = 0.4
    plt.rcParams['hatch.color'] = 'white'
    V, V3 = load()
    fig = plt.figure(figsize=(COL, 2.55))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.15], height_ratios=[1, 1], wspace=0.42, hspace=0.62)
    schematic(fig.add_subplot(gs[:, 0]))
    share_panel(fig.add_subplot(gs[0, 1]), V, V3)
    top_panel(fig.add_subplot(gs[1, 1]), V3)
    save(fig, 'fig_selfmsg_teaser', [SRC1, SRC3])


if __name__ == '__main__':
    main()
