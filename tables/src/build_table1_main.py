"""Build tables/table1_main.tex (Table I, tab:main: MSCA x multimodal training objective / plug-in / top-K loss) from
tables/table1_cells.csv, tables/topk_cells.csv and tables/xre_cells.csv. Numbers are never typed by hand.

table1_cells.csv is written by tables/src/collect_table1.py (Native and the six plug-in rows; sources, configurations and
rules in its docstring: seeds 1-5, one run per seed). The top-K loss rows (SL@K, PSL, SSM; each replaces the BPR
term of MSCA, with one score function (cosine or the dot product of MSCA) and one temperature per dataset chosen by the
seed-1 validation Recall@20, collect_topk.py RULE 3; package objx_s under
<RUN_ROOT_B>/P04_topk) come from tables/topk_cells.csv (tables/src/collect_topk.py; seeds 1-5, one run per
seed), which is merged here; table1_cells.csv is not changed by them. The proposed row of every block, "Proposed (MES)", comes from
tables/src/mes_source.py (revision 2026-10-08: MES replaces XRE): tables/mes_perseed.csv once it holds seeds 1-5 of a cell,
otherwise a PROVISIONAL stand-in (MSCA: xre_perseed.csv row wo_A1 = E + R; COHESION / SMORE: cohsmo_perseed.csv row XRE =
E + R + A). Provisional cells print every number of the row as \tbd{...} (red) and the caption names the stand-ins; bold
marks and table1_marks.csv are computed with the values that are printed, provisional ones included. Every printed cell is the mean over
seeds 1-5 with the sample standard deviation over the seeds, value{\\scriptsize$\\pm$.0011} as in Table tab:sota
(build_table2_main.py; user decision 2026-10-01). A cell with fewer than five collected seeds prints a red \\todo{pending}
and the caption lists the pending cells, so re-running the collectors and this builder fills the table once the runs are
synced to HOST_A.
Within each backbone block and column, the best printed mean (among Native, the plug-ins, the top-K losses and Ours) is set
in bold (the mean only, not the standard deviation); ours is not assumed to be best. Bold is computed over complete cells
only, so the bold mark of a column with a pending cell can still move when that cell is filled.
Table I prints the MSCA block only (user decision 2026-09-30) on Baby, Pantry, Scientific and Instruments (Instruments
column: user decision 2026-10-01, runs under <RUN_ROOT_B>/P04_instr). With four datasets and mean +- SD cells the
table no longer fits one column, so it is a table* (two-column float). The FREEDOM and MMGCN cells are printed in the
appendix transfer table (build_table_xre_appendix.py) and summarized in figures/src/fig_transfer.py.
Also writes tables/table1_marks.csv (printed mean, SD and bold mark per cell) for the prose macros (build_macros.py).
Usage: python tables/src/build_table1_main.py
"""
import csv
import hashlib
from pathlib import Path

from collect_xre import check_setting
import mes_source
import sigtest   # revision log 18-3: significance marks under the single test family

ROOT = Path(__file__).resolve().parents[1]
CELLS = ROOT / 'table1_cells.csv'
XRE_CELLS = ROOT / 'xre_cells.csv'
TOPK_CELLS = ROOT / 'topk_cells.csv'   # top-K loss rows (collect_topk.py), merged here
OUT = ROOT / 'table1_main.tex'
MARKS = ROOT / 'table1_marks.csv'
N_SEEDS = 5

BACKBONES = [('MSCA', 'MSCA~\\cite{msca2026}'), ('COHESION', 'COHESION~\\cite{xu2025cohesion}'),
             ('SMORE', 'SMORE~\\cite{ong2025smore}')]   # COHESION / SMORE blocks added 2026-10-06 (tables/cohsmo_*.csv)
COHSMO = ('COHESION', 'SMORE')
COHSMO_FILES = ['cohsmo_perseed.csv', 'cohsmo_topk_perseed.csv']   # collect_cohsmo.py (+ top-K rows when collected)
COHSMO_ROW = {'INA': 'INA', 'MDVT': 'MDVT', 'MG': 'MG', 'DA-MRS': 'DA-MRS', 'UltraGCN-LI': 'UltraGCN-LI',
              'Native': 'Native', 'SL@K': 'SL@K', 'PSL': 'PSL', 'SSM': 'SSM'}
  # FREEDOM / MMGCN blocks: appendix transfer table and fig_transfer
# (backbone, row) pairs that do not apply (none among the reported graph backbones).
NOT_APPLICABLE = set()
PANEL = 4   # datasets per stacked panel of the table (the table never uses \resizebox)
ROWS = [  # (key in csv, printed label, group); ours first, then the top-K losses and the plug-ins, each group from the
    # newest publication, Native last. The top-K losses replace the BPR term of MSCA (collect_topk.py); the plug-ins add a term.
    # Printed labels (user decision 2026-10-06): the backbone row prints the backbone name of its block (key 'Native',
    # label None = filled per block), XRE and the plug-ins add a term and print '+ ', the top-K losses replace BPR (no '+').
    # printed rows (2026-10-06): plain names, no '+'/'w/' markers; order and indentation follow LAYOUT below
    ('Ours', 'Proposed (MES)', None),
    ('SL@K', 'SL@$K$~\\cite{yang2025slk}', 'Top-$K$ loss'),
    ('PSL', 'PSL~\\cite{yang2024psl}', 'Top-$K$ loss'),
    ('SSM', 'SSM~\\cite{wu2024ssm}', 'Top-$K$ loss'),
    ('INA', 'INA-style~\\cite{iimrec}', 'Neighbor supervision'),
    ('MDVT', 'MDVT~\\cite{mdvt}', 'Augmentation'),
    ('MG', 'MG~\\cite{zhong2024mg}', 'Optimization'),
    ('DA-MRS', 'DA-MRS~\\cite{xu2024damrs}', 'Reweighting'),
    ('UltraGCN-LI', 'UltraGCN $\\mathcal{L}_I$~\\cite{mao2021ultragcn}', 'Neighbor supervision'),
    ('Native', None, None),
]
import os as _os
LAYOUT = _os.environ.get('TABLE1_LAYOUT', 'B')   # user choice 2026-10-06: B   # A / B / C, see the docstring of patch_t1_variants (2026-10-06)
_ROWD = {r[0]: r for r in ROWS}
_ORDER = {'A': ['Native', 'Ours', 'SL@K', 'PSL', 'SSM', 'INA', 'MDVT', 'MG', 'DA-MRS', 'UltraGCN-LI'],
          'B': ['Ours', 'SL@K', 'PSL', 'SSM', 'INA', 'MDVT', 'MG', 'DA-MRS', 'UltraGCN-LI', 'Native'],
          'C': ['Native', 'Ours', 'SL@K', 'PSL', 'SSM', 'INA', 'MDVT', 'MG', 'DA-MRS', 'UltraGCN-LI']}[LAYOUT]
ROWS = [_ROWD[k] for k in _ORDER]
INDENT = '\\hspace{0.6em}' if LAYOUT == 'A' else ''
DATASETS = [('baby', 'Baby'), ('pantry', 'Pantry'), ('scientific', 'Scientific'), ('sports', 'Sports')]   # Instruments and Clothing removed (user decision 2026-10-05)
METRICS = ('r20_mean', 'n20_mean')
BLOCK_END = ()   # no rule inside a block (user 2026-10-06); \midrule between blocks only


def load():
    """complete cells (n = 5) of table1_cells.csv"""
    with open(CELLS, newline='') as f:
        return {(r['backbone'], r['row'], r['dataset']): r for r in csv.DictReader(f) if int(r['n']) == N_SEEDS}


def load_topk():
    """complete cells (n = 5) of topk_cells.csv (rows SL@K, PSL, SSM); empty while the file is absent"""
    if not TOPK_CELLS.exists():
        return {}
    with open(TOPK_CELLS, newline='', encoding='utf-8') as f:
        return {(r['backbone'], r['row'], r['dataset']): r for r in csv.DictReader(f) if int(r['n']) == N_SEEDS}


def load_mes():
    """Ours cells: (backbone, 'Ours', dataset) -> mean / SD over seeds 1-5 of MES (tables/src/mes_source.py, one run per
    seed), complete cells only; 'provisional' marks a stand-in cell"""
    out = {}
    for b_key, _ in BACKBONES:
        for d_key, _ in DATASETS:
            c = mes_source.cell(b_key, d_key)
            if c:
                out[(b_key, 'Ours', d_key)] = dict(n=c['n'], r20_mean=c['r20_mean'], n20_mean=c['n20_mean'],
                                                   r20_std=c['r20_std'], n20_std=c['n20_std'], provisional=c['provisional'])
    return out, {'1'}


def load_cohsmo():
    """COHESION / SMORE cells (mean and SD over seeds 1-5) from the per-seed files; complete cells only"""
    import statistics as st
    per = {}
    for name in COHSMO_FILES:
        f = ROOT / name
        if not f.exists():
            continue
        for r in csv.DictReader(open(f, newline='', encoding='utf-8')):
            row = COHSMO_ROW.get(r['row'])
            if row:
                per.setdefault((r['backbone'], row, r['dataset']), {})[int(r['seed'])] = (float(r['r20']), float(r['n20']))
    out = {}
    for k, v in per.items():
        if sorted(v) == [1, 2, 3, 4, 5]:
            a = [x[0] for x in v.values()]; b = [x[1] for x in v.values()]
            out[k] = dict(n='5', r20_mean=st.mean(a), n20_mean=st.mean(b), r20_std=st.stdev(a), n20_std=st.stdev(b))
    return out


def load_perseed():
    """revision log 18-3: {(backbone, row, dataset, metric): {seed: value}} of every printed row, for the significance marks.
    MSCA rows: table1_perseed.csv (Native and the plug-ins) and topk_perseed.csv; COHESION / SMORE: COHSMO_FILES; the
    proposed rows: mes_source.perseed (one run per seed)."""
    per = {}
    for name in ('table1_perseed.csv', 'topk_perseed.csv'):
        for r in csv.DictReader(open(ROOT / name, newline='', encoding='utf-8')):
            if r['backbone'] == 'MSCA':
                for m in ('r20', 'n20'):
                    d = per.setdefault(('MSCA', r['row'], r['dataset'], m), {})
                    assert int(r['seed']) not in d, ('duplicate seed', name, r['row'], r['dataset'], r['seed'])
                    d[int(r['seed'])] = float(r[m])
    for name in COHSMO_FILES:
        if (ROOT / name).exists():
            for r in csv.DictReader(open(ROOT / name, newline='', encoding='utf-8')):
                row = COHSMO_ROW.get(r['row'])
                if row:
                    for m in ('r20', 'n20'):
                        per.setdefault((r['backbone'], row, r['dataset'], m), {})[int(r['seed'])] = float(r[m])
    for b_key, _ in BACKBONES:
        for (d, m), sv in mes_source.perseed(b_key).items():
            if m in ('r20', 'n20'):
                per[(b_key, 'Ours', d, m)] = dict(sv)
    return per


def sig_marks(cells, per):
    """revision log 18-3 (tables/src/sigtest.py): each row of a block against Proposed (MES) of the same block, one metric,
    two-sided paired t-test over seeds 1-5, Holm over the datasets of the table. Returns {(b, row, d, metric): mark} and
    the test rows (tables/sigtests_table1.csv). Per-seed means must reproduce the printed means."""
    marks, out = {}, []
    for b_key, _ in BACKBONES:
        for r_key, _, _ in ROWS:
            if r_key == 'Ours':
                continue
            for k in METRICS:
                m = k.split('_')[0]
                ds_ok = []
                for d_key, _ in DATASETS:
                    a, o = per.get((b_key, r_key, d_key, m), {}), per.get((b_key, 'Ours', d_key, m), {})
                    if sorted(a) == list(range(1, N_SEEDS + 1)) and sorted(o) == list(range(1, N_SEEDS + 1)):
                        for key, v in ((r_key, a), ('Ours', o)):
                            c = cells.get((b_key, key, d_key))
                            assert c and abs(float(c[k]) - sum(v.values()) / N_SEEDS) < 5e-5, (b_key, key, d_key, k)
                        ds_ok.append(d_key)
                fam = sigtest.contrast({d: per[(b_key, r_key, d, m)] for d in ds_ok},
                                       {d: per[(b_key, 'Ours', d, m)] for d in ds_ok}, ds_ok, fam=f'table1 {b_key}',
                                       name=f'{r_key} - MES', a=r_key, b='MES', metric=m)
                for r in fam:
                    r['contrast'] = f'{b_key}: {r_key} - MES'
                    if r['reject']:
                        marks[(b_key, r_key, r['dataset'], k)] = '^{\\blacktriangle}' if r['diff'] > 0 else '^{\\blacktriangledown}'
                out += fam
    return marks, out


def fmt(cell, key, bold=False):
    """mean (bold if best) followed by the SD over seeds in the format of build_table2_main.py"""
    v = f"{float(cell[key]):.4f}"
    sd = cell.get(key.replace('_mean', '_std'))
    pm = f'{{\\scriptsize$\\pm${float(sd):.4f}}}'.replace('0.', '.', 1) if sd not in (None, '') else ''
    return (f'\\textbf{{{v}}}' if bold else v) + pm


def main():
    check_setting()
    cells = load()
    topk = load_topk()
    assert not set(topk) & set(cells), 'top-K rows must not appear in table1_cells.csv'
    cells.update(topk)
    ours, ours_runs = load_mes()
    cells.update(ours)
    cells.update(load_cohsmo())
    sha = hashlib.sha256(CELLS.read_bytes()).hexdigest()[:16]
    tsha = hashlib.sha256(TOPK_CELLS.read_bytes()).hexdigest()[:16] if TOPK_CELLS.exists() else 'absent'
    pending = []   # caption note: per dataset, the rows whose cell is not complete yet
    for d_key, d_name in DATASETS:
        miss = [r for b, _ in BACKBONES if b not in COHSMO for r, _, _ in ROWS
                if (b, r) not in NOT_APPLICABLE and (b, r, d_key) not in cells]
        if miss:
            pending.append(f'{d_name} ' + ('(all rows)' if len(miss) == len(ROWS) * len(BACKBONES) else ', '.join(miss)))
    best = {}
    for b_key, _ in BACKBONES:
        for d_key, _ in DATASETS:
            for key in METRICS:
                vals = [f"{float(c[key]):.4f}" for (bb, rr, dd), c in cells.items()   # printed rows only
                        if bb == b_key and dd == d_key and c.get(key) and (bb, rr) not in NOT_APPLICABLE
                        and rr in {r for r, _, _ in ROWS}]
                best[(b_key, d_key, key)] = max(vals, key=float) if vals else None
    sig, sig_rows = sig_marks(cells, load_perseed())
    sigtest.write_csv(ROOT / 'sigtests_table1.csv', sig_rows)
    runs_note = ''   # MES: one run per seed (mes_source asserts n_runs == 1)
    mes_note = mes_source.caption_note([b for b, _ in BACKBONES], [d for d, _ in DATASETS])
    ncol = 1 + 2 * len(DATASETS)
    dash_any = any((b, r, d) not in cells for b in COHSMO for r, _, _ in ROWS for d, _ in DATASETS)
    lines = [
        f'% Generated by tables/src/build_table1_main.py from tables/table1_cells.csv (sha256 {sha}), '
        f'tables/topk_cells.csv (sha256 {tsha}) and tables/src/mes_source.py (proposed rows). Do not edit numbers by hand.',
        mes_source.comment([b for b, _ in BACKBONES], [d for d, _ in DATASETS]),
        '\\begin{table*}[t]',
        '\\caption{Recall@20 and NDCG@20 of multimodal training objectives on MSCA, COHESION, and SMORE. Each cell '
        'reports the mean over seeds 1--5 with the standard deviation over seeds. All rows of a block share the author '
        'configuration of its backbone and the common training protocol. '
        + ('The last row of each block, Backbone, is the backbone trained with its own objective. ' if LAYOUT == 'B' else
           'Each block starts with the backbone trained with its own objective. ') +
        # 2026-10-07 (page budget): the definitions of UltraGCN L_I and of the other rows live in Section IV (settings).
        # 2026-10-10 (user decision): lambda / K of MES are valid-selected on MSCA and fixed on COHESION and SMORE
        'Proposed (MES) adds MES to this objective, with $\\lambda$ and $K$ chosen per dataset by seed-1 validation '
        'Recall@20 on MSCA and fixed to $\\lambda=0.1$ and $K=40$ on COHESION and SMORE. ' +
        # R0415 (E3): the SSM grid also holds cosine training with inner-product ranking (tables/src/build_cosdot.py).
        'For SL@$K$, PSL, and SSM, the score function (cosine similarity as in their '
        'official code or the inner product of the backbone, also mixed for SSM on MSCA) and their temperature are chosen per '
        'dataset by seed-1 validation Recall@20. ' + runs_note +
        'Other rows train the backbone with the named objective. '   # 2026-10-07: last caption line >= 0.8
        'In each block and column, the best mean is in bold. '
        # revision log 18-3: marks of the single test family (tables/src/sigtest.py)
        '$^{\\blacktriangledown}$/$^{\\blacktriangle}$: significantly lower/higher than Proposed (MES) of the same block '
        f'({sigtest.RULE_TEXT}).' +
        # revision log 26 (decision 4 ii): the Baby cell of the COHESION backbone row keeps the run that tab:sota used before CACHEFIX
        # replaced it; this table is outside the CACHEFIX range (PREREG_CACHEFIX), thus the run was not repeated.
        ' The Baby cell of the COHESION backbone row repeats the run of Table~\\ref{tab:sota} from before its build-correction re-run, '
        'which this table did not repeat (supplementary Table~\\ref{tab:babybuild}).' +
        # revision log 29 (item 5): Baby cache build of each block (appendix_babybuild.tex, row of Table tab:main)
        ' On Baby, the MSCA block reads the canonical build of the item kNN cache and the COHESION and SMORE blocks its second build.' +
        (' A dash marks a cell whose runs are not finished.' if dash_any else '') + mes_note
        +(f"\\todo{{interim: pending {'; '.join(pending).replace('_', ' ')}}}" if pending else '') + '}',
        '\\label{tab:main}',
        '\\centering',
        '\\footnotesize',
        '\\setlength{\\tabcolsep}{3.5pt}',
        '\\begin{tabular}{l' + 'cc' * PANEL + '}',   # no \resizebox (lab rule): datasets stacked in panels of PANEL
    ]
    marks = []
    dash = [False]
    for pi in range(0, len(DATASETS), PANEL):
      panel = DATASETS[pi:pi + PANEL]
      pad = ' &' * (2 * (PANEL - len(panel)))
      lines += [
          '\\toprule' if pi == 0 else '\\midrule[\\heavyrulewidth]',
          ' & ' + ' & '.join(f'\\multicolumn{{2}}{{c}}{{{name}}}' for _, name in panel) + pad + ' \\\\',
          ''.join(f'\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}' for i in range(len(panel))),
          'Objective & ' + ' & '.join('R@20 & N@20' for _ in panel) + pad + ' \\\\',
      ]
      for b_key, b_label in BACKBONES:
        lines.append('\\midrule')
        if LAYOUT in ('B', 'C'):
            lines.append(f'\\multicolumn{{{1 + 2 * len(panel)}}}{{l}}{{\\textit{{Backbone: {b_label}}}}} \\\\')
        for r_key, r_label, _ in ROWS:
            vals = []
            for d_key, _ in panel:
                c = cells.get((b_key, r_key, d_key))
                if (b_key, r_key) in NOT_APPLICABLE:
                    vals += ['n/a', 'n/a']
                    continue
                for k in METRICS:
                    is_best = bool(c) and f"{float(c[k]):.4f}" == best[(b_key, d_key, k)]
                    marks.append([b_key, r_key, d_key, k.split('_')[0], f"{float(c[k]):.4f}" if c else '',
                                  c.get(k.replace('_mean', '_std'), '') if c else '', 'best' if is_best else '',
                                  int(bool(c and c.get('provisional')))])
                if not c:
                    vals += ['--', '--'] if b_key in COHSMO else ['\\todo{pending}', '\\todo{pending}']
                    dash[0] |= b_key in COHSMO
                    continue
                vals += [mes_source.tbd(fmt(c, k, f"{float(c[k]):.4f}" == best[(b_key, d_key, k)]), c.get('provisional'))
                         + (f"\\rlap{{${sig[(b_key, r_key, d_key, k)]}$}}" if (b_key, r_key, d_key, k) in sig else '')
                         for k in METRICS]
            name = (INDENT + r_label) if r_label else (b_label if LAYOUT == 'A' else
                                                       f"Backbone ({b_label.split('~')[0]})")
            lines.append(f'{name} & ' + ' & '.join(vals) + ' \\\\')
            if r_key in BLOCK_END:
                lines.append(f'\\cmidrule(lr){{1-{1 + 2 * len(panel)}}}')
    lines += ['\\bottomrule', '\\end{tabular}', '\\end{table*}', '']
    OUT.write_text('\n'.join(lines), encoding='utf-8', newline='\n')
    with open(MARKS, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['backbone', 'row', 'dataset', 'metric', 'printed', 'std', 'mark', 'provisional'])
        w.writerows(marks)
    print('wrote', OUT, '| pending:', ', '.join(pending) or 'none')


if __name__ == '__main__':
    main()
