"""Analysis figure: crowding at the top of the list and the effect of XRE on held-out positives (fig:crowding).

Supports Proposition prop:history (b) and Lemma lem:masked. Data: one seed-1 checkpoint per dataset and arm of MSCA,
restored best-valid state whose re-evaluated test Recall@20 matches its result.json to four decimals (HOST_A
<RUN_ROOT_B>/P04_listana/tools/list_stats_full.py). Baby / Pantry / Scientific: P04_listana reruns of the
Table I configuration (no table cell uses them; source 'listana'). Sports: the seed-1 Table I Native and XRE runs of
P04_sports/runs_xres (source 'tableI'). Both CSV files are written by tables/src/collect_listana.py (remote stats/ and
stats_sports/ summaries of list_stats_full.py).
  (a) tables/listana_stats.csv, arm native: share of items that are neighbors of at least one history item in an item-item
      graph of MSCA (co-occurrence, visual kNN or textual kNN), for the top-20 list (link_any), the held-out test targets
      (tgt_link_any) and 20 uniformly drawn items (rnd_link_any). Mean over test users.
  (b) tables/listana_rank.csv, metric share_top40: training positives scored on the masked view without their fold
      (train_masked) and held-out test targets (test), Native vs XRE.
When a (dataset, arm, condition, metric) appears twice, the 'listana' source wins.
Usage: python -s figures/src/fig_crowding.py
"""
import csv

from style import C_CTRL, C_GREY, C_OURS, COL, TABLES, plt, save, setup

DATASETS = [('baby', 'Baby'), ('pantry', 'Pantry'), ('scientific', 'Scientific'), ('sports', 'Sports')]
STATS = TABLES / 'listana_stats.csv'
RANK = TABLES / 'listana_rank.csv'
C_LIGHT_OURS = '#9db7da'
C_LIGHT_CTRL = '#e3a9a7'


def load(path, key_cols):
    out = {}
    with open(path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            k = tuple(r[c] for c in key_cols)
            if k not in out or r['source'] == 'listana':
                out[k] = float(r['value'])
    return out


def label(ax, x, v, fs=5.2, dy=0.015):
    ax.text(x, v + dy, f'{v:.2f}'.lstrip('0'), ha='center', va='bottom', fontsize=fs)


def panel_a(ax):
    st = load(STATS, ('dataset', 'arm', 'metric'))
    groups = [('link_any', C_CTRL, 'Top-20 list'), ('tgt_link_any', C_GREY, 'Test target'),
              ('rnd_link_any', '#cccccc', 'Random item')]
    w = 0.27
    for j, (ds, _) in enumerate(DATASETS):
        for k, (m, col, lab) in enumerate(groups):
            v = st[(ds, 'native', m)]
            ax.bar(j + (k - 1) * w, v, w * 0.92, color=col, label=lab if j == 0 else None)
            label(ax, j + (k - 1) * w, v)
    ax.set_title('(a) Neighbors of the history (MSCA)', fontsize=6.8)
    ax.set_ylabel('Share of items')
    ax.legend(loc='upper center', ncol=3, frameon=False, handlelength=0.8, fontsize=5.2, borderaxespad=0.1,
              columnspacing=0.6)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 0.5, 1])


def panel_b(ax):
    rk = load(RANK, ('dataset', 'arm', 'condition', 'metric'))
    groups = [('native', 'train_masked', C_LIGHT_CTRL, 'Held-out positive, MSCA'),
              ('xre', 'train_masked', C_LIGHT_OURS, 'Held-out positive, MSCA + XRE'),
              ('native', 'test', C_CTRL, 'Test target, MSCA'), ('xre', 'test', C_OURS, 'Test target, MSCA + XRE')]
    w = 0.2
    for j, (ds, _) in enumerate(DATASETS):
        for k, (arm, cond, col, lab) in enumerate(groups):
            v = rk[(ds, arm, cond, 'share_top40')]
            ax.bar(j + (k - 1.5) * w, v, w * 0.92, color=col, label=lab if j == 0 else None)
            label(ax, j + (k - 1.5) * w, v, fs=4.6, dy=0.015 + 0.045 * (k % 2))   # staggered: four datasets
    ax.set_title('(b) Share in the top-40 list', fontsize=6.8)
    ax.legend(loc='upper left', frameon=False, handlelength=0.8, fontsize=5.2, borderaxespad=0.1)
    ax.set_ylim(0, 0.96)   # 0.96 (was 0.8): the 'MSCA + XRE' legend clears the bar labels
    ax.set_yticks([0, 0.25, 0.5, 0.75])


def main():
    setup()
    fig, axes = plt.subplots(1, 2, figsize=(COL, 1.75))
    panel_a(axes[0])
    panel_b(axes[1])
    for ax in axes:
        ax.set_xticks(range(len(DATASETS)))
        ax.set_xticklabels([n for _, n in DATASETS], fontsize=6)
        ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(w_pad=0.6)
    save(fig, 'fig_crowding', [STATS, RANK])


if __name__ == '__main__':
    main()
