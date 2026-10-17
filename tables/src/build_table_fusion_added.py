"""Build tables/table_fusion_added.tex (tab:fuse_added): the fused baseline F and the validation mix on the five datasets added to the
analyses (Instruments, Clothing, Arts, Office, MicroLens).  Added 2026-10-16 (revision log 13); revision log 29 (items 1 and 4): rebuilt from the
CLASSIC fused cells (build_table_classic.cell via build_rev2.py), which hold the TABLE_FUSION/MES9 F of every gated checkpoint and the F
recomputed by tf_fuse_ce.py --combo on Arts, Office, and MicroLens, so that MicroLens and the other models on Arts and Office enter F,
and every dataset gets a mix column.  The previous version (fusion_summary_9ds_mes.csv, four datasets, F only) is
tables/src/_old/build_table_fusion_added_pre29.py.  Numbers are never typed by hand.
Columns per dataset: Recall@20 unfused, with F, with the mix (mean over the gated seeds; the seed count in parentheses when below five).
A missing (model, dataset) pair prints '--'.  Last row (revision log 31): MSCA + MES minus MSCA over seeds 1-10 (msca10_margins.csv), Instruments and Clothing, with the mark of the paired t-test
(Holm over the six datasets with ten seeds, auxiliary family).  No other row carries bold or significance marks.
Usage: python -s tables/src/build_table_fusion_added.py
"""
import csv
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_rev2 as rv  # noqa: E402

ROOT = HERE.parent
OUT = ROOT / 'table_fusion_added.tex'
DATASETS = [('instruments', 'Instruments'), ('clothing', 'Clothing'), ('arts', 'Arts'), ('office', 'Office'), ('microlens', 'MicroLens')]
VARS = [('base', 'Unf.'), ('combo', 'F'), ('mix', 'Mix')]
BS = chr(92)   # one backslash


def sgn(x):
    return f'{x:+.4f}'.replace('-', '$-$').replace('+', '$+$')


def main():
    bc = rv.bc

    def cell(m, d, v):
        s = [i for i in bc.SEEDS if (m, d, i) in bc.cell and v in bc.cell[(m, d, i)] and 'base' in bc.cell[(m, d, i)]]
        if not s:
            return '--'
        x = st.mean(bc.cell[(m, d, i)][v] for i in s)
        return f'{x:.4f}' + ('' if len(s) == 5 else BS + f',({len(s)})')

    # revision log 31 (MSCA10): the last row reads the ten-seed margins of the preregistered extension (msca10_margins.csv, written by
    # build_macros_msca10.py), Instruments and Clothing only, Holm over the six datasets with ten seeds (auxiliary family)
    FUSV = {'base': 'base', 'combo': 'F', 'mix': 'mix'}
    marg = {(r['fusion'], r['dataset']): r for r in csv.DictReader(open(HERE / 'data' / 'msca10_margins.csv', newline='', encoding='utf-8'))
            if r['metric'] == 'r20'}
    for r in marg.values():
        r['diff'] = float(r['diff'])
        r['reject'] = r['reject'] == '1'

    def diff(d, v):
        r = marg.get((FUSV[v], d))
        if not r or r['family'] != 'aux':
            return '--'
        body = sgn(r['diff'])
        if r['reject']:
            body += BS + 'rlap{$^{' + BS + ('blacktriangle' if r['diff'] > 0 else 'blacktriangledown') + '}$}'
        return body

    def aux_sig(v):
        names = [dict(DATASETS)[d] for d in ('instruments', 'clothing') if marg[(FUSV[v], d)]['reject']]
        return 'neither dataset' if not names else (' and '.join(names) + ' alone')

    pairs = rv.f_universe()
    added = [p for p in pairs if p['dataset'] in dict(DATASETS)]
    assert all(p['up'] for p in added)
    four = ['baby', 'pantry', 'scientific', 'sports']
    pre4 = [marg[('base', d)]['diff'] for d in four]
    assert all(marg[('base', d)]['n'] == '10' for d in four)
    cols = [(d, v) for d, _ in DATASETS for v, _ in VARS]
    caption = (
        'Recall@20 without fusion (Unf.), with the fused baseline F, and with the validation mix of RP3$' + BS + 'beta$, ItemKNN-CBF, ItemKNN-CF, and EASE (Mix) '
        'on the five datasets added to the analyses, as the mean over the seeds whose checkpoint passed the re-evaluation gate, '
        'with the seed count in parentheses when it is below five. '
        'On these datasets MES uses $' + BS + 'lambda=0.1$ and $K=40$ without tuning, whereas Table~' + BS + 'ref{tab:sota} uses values selected on validation. '
        'A dash marks a model without stored runs, namely MSCA on Arts, Office, and MicroLens and MSCA + MES on MicroLens. '
        'The last row gives the difference between MSCA + MES and MSCA in the same column over seeds 1--10 of the preregistered ten-seed extension, '
        'whereas the other rows average seeds 1--5, and it is marked when significant with Holm correction over the six datasets with ten seeds, '
        'an auxiliary family beside the four tuned datasets of Table~' + BS + 'ref{tab:sota}. No other row carries bold or significance marks. '
        f'In this family the difference after F is significant on {aux_sig("combo")} and after the mix on {aux_sig("mix")}. '
        f'Without fusion, this difference is {sgn(marg[("base", "instruments")]["diff"])} on Instruments and {sgn(marg[("base", "clothing")]["diff"])} on Clothing, '
        f'against {sgn(min(pre4))} to {sgn(max(pre4))} on the four tuned datasets in the same fusion runs, '
        'whose MSCA shares the checkpoints of Table~' + BS + 'ref{tab:sota} on Sports alone. '
        'The tuning asymmetry makes the two ranges not comparable, and we do not read the larger untuned difference as a stronger effect.')
    mc = BS + 'multicolumn{3}{c}{'
    lines = [
        '% Generated by tables/src/build_table_fusion_added.py (revision log 29) from the CLASSIC fused cells (build_rev2.py). Do not edit numbers by hand.',
        BS + 'begin{table*}[t]',
        BS + 'caption{' + caption + '}',
        BS + 'label{tab:fuse_added}',
        BS + 'centering',
        BS + 'footnotesize',
        BS + 'setlength{' + BS + 'tabcolsep}{1.9pt}',
        BS + 'begin{tabular}{l' + 'ccc' * len(DATASETS) + '}',
        BS + 'toprule',
        ' & ' + ' & '.join(mc + n + '}' for _, n in DATASETS) + ' ' + BS + BS,
        ''.join(BS + f'cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}' for i in range(len(DATASETS))),
        'Model & ' + ' & '.join(' & '.join(lab for _, lab in VARS) for _ in DATASETS) + ' ' + BS + BS,
        BS + 'midrule',
    ]
    for m in bc.MODELS:
        lines.append(f'{bc.MLAB[m]} & ' + ' & '.join(cell(m, d, v) for d, v in cols) + ' ' + BS + BS)
    lines += [BS + 'midrule',
              'MES $-$ MSCA & ' + ' & '.join(diff(d, v) for d, v in cols) + ' ' + BS + BS,
              BS + 'bottomrule', BS + 'end{tabular}', BS + 'end{table*}', '']
    OUT.write_text('\n'.join(lines), encoding='utf-8', newline='\n')
    print('wrote', OUT.name, len(added), 'pairs, all raised by F')


if __name__ == '__main__':
    main()
