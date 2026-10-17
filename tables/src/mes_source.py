"""Per-seed values of the proposed method MES (Masked Entry Supervision) for every table and macro builder.

One function answers "which numbers stand for MES on (backbone, dataset, seed)", so the switch from provisional stand-in
values to the real MES runs is automatic (revision rev-selfmsg-1009, 2026-10-08):

  real        tables/mes_perseed.csv (columns backbone,row,dataset,seed,r20,n20[,r10,n10][,n_runs]; row = 'MES'), or the
              file named by the environment variable MES_PERSEED (used to test the non-provisional path). A
              (backbone, dataset) cell uses this file once it holds seeds 1-5 there; provisional = False.
  provisional otherwise, per (backbone, dataset):
              MSCA                     tables/xre_perseed.csv row wo_A1 (E + R: the masked exchange without the anchor
                                       term; replicate-1 runs only, n_runs == 1 is asserted as in build_paired.py and
                                       collect_hyp_forest.py)
              COHESION, SMORE          tables/cohsmo_perseed.csv row XRE (the earlier anchor-plus-exchange design E + R + A)
              MENTOR, FREEDOM, MMGCN   tables/xre_perseed.csv row XRE (E + R + A)
              provisional = True; the builders print such numbers as \\tbd{...} (red, defined in main.tex/supplement.tex).

Switching to real numbers: put mes_perseed.csv into tables/ (or set MES_PERSEED) and re-run
`python -s tables/src/make_all.py --no-collect`; no \\tbd remains for the cells that the file covers.
"""
import csv
import os
import statistics as st
from pathlib import Path

TABLES = Path(__file__).resolve().parents[1]
DEFAULT = TABLES / 'mes_perseed.csv'
XRE_PERSEED = TABLES / 'xre_perseed.csv'
COHSMO_PERSEED = TABLES / 'cohsmo_perseed.csv'
SEEDS = (1, 2, 3, 4, 5)
METRICS = ('r20', 'n20', 'r10', 'n10')
NAME = 'MES'
# provisional stand-in per backbone: (file, row)
STANDIN = {'MSCA': (XRE_PERSEED, 'wo_A1'), 'COHESION': (COHSMO_PERSEED, 'XRE'), 'SMORE': (COHSMO_PERSEED, 'XRE'),
           'MENTOR': (XRE_PERSEED, 'XRE'), 'FREEDOM': (XRE_PERSEED, 'XRE'), 'MMGCN': (XRE_PERSEED, 'XRE')}
CAPTION_NOTE = ('\\tbd{Red values are provisional: the masked exchange without the anchor term (MSCA) and the earlier '
                'anchor-plus-exchange design (COHESION, SMORE) stand in until the MES runs finish.}')


def path():
    """the real MES file (MES_PERSEED overrides tables/mes_perseed.csv)"""
    return Path(os.environ.get('MES_PERSEED') or DEFAULT)


def _read(f, row, assert_one_run):
    """{(backbone, dataset): {seed: {metric: value}}} of one row of a per-seed file"""
    out = {}
    if not f.exists():
        return out
    with open(f, newline='', encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            if r['row'] != row:
                continue
            if assert_one_run and r.get('n_runs') not in (None, ''):
                assert int(r['n_runs']) == 1, r   # one run per seed (rep1)
            out.setdefault((r['backbone'], r['dataset']), {})[int(r['seed'])] = {
                m: float(r[m]) for m in METRICS if r.get(m) not in (None, '')}
    return out


_CACHE = {}
K40_STANDIN = set()   # MSCA datasets whose stand-in is MES lambda 0.1, K 40 (row MES_k40)


def _load():
    key = str(path())
    if key not in _CACHE:
        real = _read(path(), NAME, True)
        stand = {}
        for bb, (f, row) in STANDIN.items():
            for (b, d), v in _read(f, row, f == XRE_PERSEED).items():
                if b == bb:
                    stand[(b, d)] = v
        # 2026-10-10: the main MSCA row is MES at the valid-selected lambda / K per dataset; until that cell has seeds 1-5,
        # MES at lambda 0.1, K 40 (row MES_k40, P04_xen0) stands in (still printed red), not E + R
        for (b, d), v in _read(path(), 'MES_k40', True).items():
            if b == 'MSCA' and _complete(v):
                stand[(b, d)] = v
                K40_STANDIN.add(d)
        _CACHE[key] = (real, stand)
    return _CACHE[key]


def _complete(v):
    return v is not None and all(s in v and 'r20' in v[s] and 'n20' in v[s] for s in SEEDS)


def is_provisional(backbone, dataset):
    """True while the real MES file lacks seeds 1-5 of this cell (the stand-in values are used then)"""
    real, _ = _load()
    return not _complete(real.get((backbone, dataset)))


def seeds(backbone, dataset):
    """{seed: {'r20', 'n20'[, 'r10', 'n10']}} of MES on (backbone, dataset): real values when complete, else the stand-in
    (possibly incomplete or empty)."""
    real, stand = _load()
    v = real.get((backbone, dataset))
    return dict(v) if _complete(v) else dict(stand.get((backbone, dataset), {}))


def row(backbone, dataset, seed):
    """(values, provisional) of one seed: values = {'r20', 'n20', ...} or None"""
    return seeds(backbone, dataset).get(seed), is_provisional(backbone, dataset)


def perseed(backbone, datasets=None):
    """{(dataset, metric): {seed: value}} (the format of native_rows.perseed)"""
    out = {}
    real, stand = _load()
    for (b, d) in set(real) | set(stand):
        if b != backbone or (datasets and d not in datasets):
            continue
        for s, vals in seeds(b, d).items():
            for m, x in vals.items():
                out.setdefault((d, m), {})[s] = x
    return out


def cell(backbone, dataset):
    """mean and sample SD over seeds 1-5 (the format of xre_cells.csv), or None while a seed is missing"""
    v = seeds(backbone, dataset)
    if not _complete(v):
        return None
    out = dict(n='5', seeds='1/2/3/4/5', provisional=is_provisional(backbone, dataset))
    for m in METRICS:
        x = [v[s][m] for s in SEEDS if m in v[s]]
        if len(x) == len(SEEDS):
            out[f'{m}_mean'] = st.mean(x)
            out[f'{m}_std'] = st.stdev(x)
    return out


def any_provisional(backbones, datasets):
    return any(is_provisional(b, d) for b in backbones for d in datasets)


def tbd(s, provisional=True):
    """wrap a printed value in \\tbd{} (red) while it is provisional"""
    return f'\\tbd{{{s}}}' if provisional and s not in (None, '') else s


def caption_note(backbones, datasets):
    """' \\tbd{Red values are provisional: ...}' naming the stand-ins of the provisional backbones, or '' when none is"""
    msca = 'MSCA' in backbones and any_provisional(['MSCA'], datasets)
    msca_k40 = msca and all(d in K40_STANDIN for d in datasets if is_provisional('MSCA', d))
    cs = [b for b in ('COHESION', 'SMORE') if b in backbones and any_provisional([b], datasets)]
    if msca and not msca_k40 and len(cs) == 2:
        return ' ' + CAPTION_NOTE
    msca_txt = ('MES with $\\lambda=0.1$ and $K=40$ (MSCA)' if msca_k40 else
                'the masked exchange without the anchor term (MSCA)')
    parts = ([msca_txt] if msca else []) + (
        [f"the earlier anchor-plus-exchange design ({', '.join(cs)})"] if cs else [])
    if not parts:
        return ''
    verb = 'stand' if len(parts) == 2 else 'stands'
    return f" \\tbd{{Red values are provisional: {' and '.join(parts)} {verb} in until the MES runs finish.}}"


def source_text(backbones, datasets):
    """one line for the comment header of every generated file"""
    parts = []
    for b in backbones:
        prov = [d for d in datasets if is_provisional(b, d)]
        real = [d for d in datasets if d not in prov]
        f, r = STANDIN[b]
        if real:
            parts.append(f'{b} {"/".join(real)}: {path().name} row MES')
        k40 = [d for d in prov if b == 'MSCA' and d in K40_STANDIN]
        other = [d for d in prov if d not in k40]
        if k40:
            parts.append(f'{b} {"/".join(k40)}: PROVISIONAL stand-in {path().name} row MES_k40 (lambda 0.1, K 40)')
        if other:
            parts.append(f'{b} {"/".join(other)}: PROVISIONAL stand-in {f.name} row {r}')
    return 'MES source: ' + '; '.join(parts)


def comment(backbones, datasets):
    return '% ' + source_text(backbones, datasets)
