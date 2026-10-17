"""Build the supplementary hyperparameter table (label tab:hparams). Numbers are never typed by hand.

Inputs:
  tables/hparams_configs.tsv   (package, config, key, value) of every reported configuration and of configs/overall.yaml,
                               produced by tables/src/collect_hparams.py from the YAML files behind the chosen runs
  tables/hparams_sources.tsv   rule and source location of every key of the author-configuration tables
  tables/topk_cells.csv        chosen configuration, score function and temperature of each top-K loss cell
                               (tables/src/collect_topk.py)
  tables/cohsmo_mg_astar.tsv   MG alpha2 chosen on COHESION and SMORE (tables/src/collect_cohsmo.py --astar-only); the
                               other plug-in values on these two backbones equal the MSCA values (checked on the package
                               YAML files of P04_cohsmo_wt 77bb243, 2026-10-06), as the caption states
Output:
  tables/appendix_hparams.tex
Rows: XRE and the five plug-ins of Table I (values over every backbone and dataset they run on), the three top-K losses on
MSCA, then every backbone and compared recommender. A value that is the same in every cell of its row is printed once.
Otherwise the cell lists the per-dataset values in the order Baby/Pantry/Scientific/Instruments/Clothing/Sports (the
datasets of that backbone or model, in this order).
MSCA row: the MSCA block of objx_s <ds>_native (the configuration of the MSCA row of Table II and of every Native row on
MSCA, tables/src/native_rows.py), restricted to the keys of msca MSCA_A5_baby and checked against msca MSCA_A5_<ds> where
that configuration is collected.
Source labels:
  compared recommenders with an author-configuration table (FREEDOM, LGMRec, PGL, DiffMM, MMGCN): derived from the
    source column of that table for the printed keys (paper text, repository file or log, common L2 rule);
  MSCA, SMORE, COHESION, MENTOR: from their package documents (P04_t2b2048L_wt git 154df16
    results/newbl/msca_smore_CONFIG.md, models/cohesion/README.md and the inline comments of COHESION_A5_<ds>.yaml,
    <TOOLS>/hp_audit/hp_audit.csv rows MENTOR);
  plug-ins: from their configuration documents (P04_table1fix_wt results/plugins_msf/CONFIG.md, results/plugins_msf2/
    CONFIG.md, the table1fix commit 198b975 message, <TOOLS>/official_repro/CODE_DIFF.md 3.4-3.5);
  top-K losses: P04_topk README.txt (N, tau_score of PSL, K, tau_beta and epoch_quantile of SL@K = IR-Benchmark code and
    README defaults; score function and temperature by seed-1 validation Recall@20).
Usage: python tables/src/build_table_hparams.py
"""
import csv
import hashlib
import json
from collections import OrderedDict, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / 'hparams_configs.tsv'
SOURCES = ROOT / 'hparams_sources.tsv'
TOPK = ROOT / 'topk_cells.csv'
COHSMO_MG = ROOT / 'cohsmo_mg_astar.tsv'
MES_PERSEED = ROOT / 'mes_perseed.csv'     # revision log 18-4: lambda and K of the MES runs, read from the config name of each run_dir
OUT = ROOT / 'appendix_hparams.tex'
# Rows of the compared recommenders (author configurations): released with the code (docs/hyperparameters_compared.md),
# kept out of the supplement table so that the supplementary material fits the TMM limit of four pages.
OUT_MD = ROOT / 'hparams_compared.md'
DATASETS = ['baby', 'pantry', 'scientific', 'sports']   # Instruments and Clothing removed (2026-10-05)
DS3 = DATASETS[:3]
BACKBONE = {'objx_s': 'MSCA', 'plg_s': 'MSCA', 'plg2_s': 'MSCA', 'xres': 'MSCA', 'plg_f': 'FREEDOM',
            'plg2_f': 'FREEDOM', 'xret_f': 'FREEDOM', 'plg2_m': 'MMGCN', 'xret_m': 'MMGCN', 'xret_n': 'MENTOR'}
BB_ORDER = ['MSCA', 'MENTOR', 'FREEDOM', 'MMGCN']
# datasets of each plug-in backbone (asserted against the collected configurations; stated in the caption)
BB_DATASETS = {'MSCA': DATASETS, 'MENTOR': DATASETS, 'FREEDOM': DS3, 'MMGCN': DS3}

# (row label, config-name marker, packages, [(yaml key, printed symbol)], source label)
PLUGINS = [
    ('E + R + A design; A: $J$, $M$', '_xre_', ('xres', 'xret_n', 'xret_f', 'xret_m'),
     [('rsc_lambda', '$\\lambda$'), ('xres_k', '$K$'), ('rsc_support_k', '$J$'), ('rsc_neg_m', '$M$')],
     'This paper'),
    ('MG', '_mg_', ('objx_s', 'plg_f', 'plg2_m'),
     [('alpha1', '$\\alpha_1$'), ('beta', '$\\beta$'), ('alpha2', '$\\alpha_2$')],
     'Repo ($\\alpha_1,\\beta$), released grid ($\\alpha_2$)'),
    ('MDVT', '_mdvt_', ('plg_s', 'plg_f', 'plg2_m'),
     [('mdvt_k', '$n$'), ('mdvt_s_weight', '$\\lambda$'), ('mdvt_threshold', '$g$')],
     'Paper ($n,\\lambda$), repo ($g$)'),
    ('DA-MRS', '_damrs_', ('plg2_s', 'plg2_f', 'plg2_m'),
     [('damrs_alpha', '$\\alpha$'), ('damrs_beta', '$\\beta$'), ('damrs_gamma', '$\\gamma$'), ('damrs_knn_k', '$k$'),
      ('damrs_iib_topk', '$\\xi_B$'), ('damrs_kl_weight', '$\\lambda_1$'), ('damrs_neighbor_weight', '$\\lambda_2$'),
      ('damrs_n_mm_layers', 'layers')],
     'Repo ($\\alpha,\\beta,\\gamma$), paper ($k,\\xi_B$), log'),
    ('UltraGCN $\\mathcal{L}_I$', '_ugcn_', ('plg2_s', 'plg2_f', 'plg2_m'),
     [('ugcn_ii_neighbor_num', '$K$'), ('ugcn_lambda', '$\\lambda$')],
     'Repo ($K$), repo loss ratio ($\\lambda$)'),
    ('INA-style', '_ina_', ('plg2_s', 'plg2_f', 'plg2_m'),
     [('ina_k', '$k$'), ('ina_delta', '$\\delta$'), ('ina_lambda', '$\\lambda$')], 'Paper'),
]
# top-K losses on MSCA: (row label, topk_cells row, objx_main, [(fixed yaml key, printed symbol)], source label)
TOPK_ROWS = [
    ('SL@$K$', 'SL@K', 'slk',
     [('objx_neg_num', '$N$'), ('objx_slk_k', '$K$'), ('objx_slk_tau_beta', '$\\tau_\\beta$'),
      ('objx_slk_epoch_quantile', '$E$')],
     'Repo ($N,K,\\tau_\\beta,E$), seed-1 validation (score, $\\tau$)'),
    ('PSL', 'PSL', 'psl', [('objx_neg_num', '$N$'), ('objx_psl_tau_score', '$\\tau_d$')],
     'Repo ($N,\\tau_d$), seed-1 validation (score, $\\tau$)'),
    ('SSM', 'SSM', 'ssm', [('objx_neg_num', '$N$')], 'Repo ($N$), seed-1 validation (score, $\\tau$)'),
]
SCORE = {'cosine': 'cos', 'dot': 'dot'}
# (row label, package, config pattern, source table config pattern or None, fixed source label or None, rule-2 keys,
#  datasets)
MODELS = [
    ('MSCA', 'objx_s', '{ds}_native', None, 'Repo', (), DATASETS),
    ('COHESION', 'cohesion', 'COHESION_A5_{ds}', None, 'Paper, repo', ('reg_weight',), DATASETS),
    ('SMORE', 'smore', 'SMORE_A5_{ds}', None, 'Repo', ('reg_weight',), DATASETS),
    ('PGL', 'pgl', None, '{cfg}', None, (), DATASETS),
    ('MENTOR', 'mentor', 'MENTOR', None, 'Paper, repo', (), DATASETS),
    ('DiffMM', 'diffmm', None, '{cfg}', None, (), DATASETS),
    ('LGMRec', 'lgmrec', None, '{cfg}', None, (), DATASETS),
    ('FREEDOM', 'freedom', None, '{cfg}', None, (), DATASETS),
    ('MMGCN', 'plg2_m', '{ds}_a4_native', 'MMGCN_A4_{ds}', None, (), DS3),
]
MSCA_KEYS_FROM = ('msca', 'MSCA_A5_baby')
# Keys not printed: common to every model (stated in the caption), bookkeeping of the pipeline, or not read by the
# trainer (learning_rate_scheduler: the common cosine schedule replaces it).
SKIP = {'embedding_size', 'feat_embed_dim', 'weight_size', 'hyper_parameters', 'use_amp', 'infonce_chunk',
        'learning_rate_scheduler', 'is_sparse', 'mode', 'cf_model'}


def fmt(v):
    v = json.loads(v) if isinstance(v, str) else v
    if isinstance(v, bool) or v is None:
        return str(v)
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    if x == 0:
        return '0'
    if abs(x) < 1e-2:
        m, e = f'{x:.0e}'.split('e')
        e = int(e)
        return f'$10^{{{e}}}$' if m == '1' else f'${m}{{\\times}}10^{{{e}}}$'
    if x == int(x) and abs(x) >= 1:
        return str(int(x))
    return f'{x:g}'


def tt(k):
    return '\\texttt{' + k.replace('_', '\\_') + '}'


def load():
    cfg = defaultdict(OrderedDict)
    meta = {}
    with open(CONFIGS, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            cfg[(r['package'], r['config'])][r['key']] = r['value']
            meta[(r['package'], r['config'])] = (r['worktree'], r['commit'], r['yaml_sha256'])
    src = defaultdict(dict)
    with open(SOURCES, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            src[r['config']][r['key']] = r
    return cfg, meta, src


def plugin_row(cfg, marker, pkgs, keys):
    cells = defaultdict(dict)  # key -> {(backbone, ds): value}
    for (pkg, c), d in cfg.items():
        if pkg not in pkgs or marker not in c:
            continue
        ds = c.split('_')[0]
        for k, _ in keys:
            cells[k][(BACKBONE[pkg], ds)] = fmt(d[k])
    indep, spec = [], []
    for k, sym in keys:
        vals = cells[k]
        assert vals, (marker, k)
        by_bb = OrderedDict()
        for bb in BB_ORDER:
            have = [ds for ds in DATASETS if (bb, ds) in vals]
            if have:
                assert have == BB_DATASETS[bb], (marker, k, bb, have)
                by_bb[bb] = [vals[(bb, ds)] for ds in have]
        if len(set(vals.values())) == 1:
            indep.append(f'{sym} {next(iter(vals.values()))}')
            continue
        if all(len(set(vv)) == 1 for vv in by_bb.values()):
            groups = OrderedDict()
            for bb, vv in by_bb.items():
                groups.setdefault(vv[0], []).append(bb)
            main = max(groups, key=lambda v: len(groups[v]))
            rest = ', '.join(f'{v} on {", ".join(b)}' for v, b in groups.items() if v != main)
            indep.append(f'{sym} {main} ({rest})')
        else:
            spec.append(f'{sym}: ' + ', '.join(f'{bb} {"/".join(vv)}' for bb, vv in by_bb.items()))
    return indep, spec


def mes_row():
    """MES row (revision log 18-4): lambda and K of the MES runs behind the reported rows, parsed from the config name
    (..._l<lambda x 100>_k<K>) of run_dir in tables/mes_perseed.csv. MSCA: the setting selected per dataset on seed-1 validation
    (row MES); the other backbones: one setting for all datasets. J and M belong to the anchor term A of the E + R + A design
    (next row), not to MES."""
    import re
    found = defaultdict(set)   # (backbone, dataset) -> {(lambda, K)}
    with open(MES_PERSEED, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['row'] != 'MES':
                continue
            m = re.search(r'_l(\d{3})_k(\d+)__' + r['dataset'] + r'__s\d', r['run_dir'])
            assert m, r['run_dir']
            found[(r['backbone'], r['dataset'])].add((int(m.group(1)) / 100, int(m.group(2))))
    assert all(len(v) == 1 for v in found.values()), found
    one = {k: next(iter(v)) for k, v in found.items()}
    msca = [one[('MSCA', ds)] for ds in DATASETS]
    out = ['$\lambda$ ' + '/'.join(f'{l:g}' for l, _ in msca) + ', $K$ ' + '/'.join(str(k) for _, k in msca) + ' on MSCA']
    others = OrderedDict()
    for (bb, ds), lk in one.items():
        if bb != 'MSCA':
            others.setdefault(lk, [])
            if bb not in others[lk]:
                others[lk].append(bb)
    order = ['COHESION', 'SMORE', 'MENTOR', 'FREEDOM', 'MMGCN']
    for (l, k), bbs in others.items():
        out.append(f'$\lambda$ {l:g}, $K$ {k} on ' + ', '.join(sorted(bbs, key=order.index)))
    return out


def topk_row(cfg, topk, row, main, keys):
    per = OrderedDict()
    for ds in DATASETS:
        t = topk[(row, ds)]
        pkg, c = t['source'].split('/')
        d = cfg[(pkg, c)]
        assert json.loads(d['objx_main']) == main, (row, ds, c)
        assert float(json.loads(d['objx_tau'])) == float(t['tau_star']), (row, ds, c, d['objx_tau'], t['tau_star'])
        # objx_score absent = cosine (models/objx_m/src/objloss.py: "absent = cosine")
        score = json.loads(d['objx_score']) if 'objx_score' in d else 'cosine'
        assert score == t['score'], (row, ds, c, score, t['score'])
        # backbone block = the MSCA Native configuration of that dataset
        nat = cfg[('objx_s', f'{ds}_native')]
        for k in cfg[MSCA_KEYS_FROM]:
            assert fmt(d[k]) == fmt(nat[k]), (row, ds, k)
        per[ds] = (SCORE[t['score']], fmt(d['objx_tau']), {k: fmt(d[k]) for k, _ in keys})
    out = []
    for name, vv in (('score', [v[0] for v in per.values()]), ('$\\tau$', [v[1] for v in per.values()])):
        out.append(f'{name} {vv[0]}' if len(set(vv)) == 1 else f'{name} {"/".join(vv)}')
    for k, sym in keys:
        vv = [v[2][k] for v in per.values()]
        out.append(f'{sym} {vv[0]}' if len(set(vv)) == 1 else f'{sym} {"/".join(vv)}')
    return out


def model_row(cfg, src, pkg, pattern, src_pattern, rule2, datasets):
    per_ds = OrderedDict()
    for ds in datasets:
        if pattern is not None:
            c = pattern.format(ds=ds)
        else:
            cands = [c for (p, c) in cfg if p == pkg and c.lower().endswith('_' + ds)]
            assert len(cands) == 1, (pkg, ds, cands)
            c = cands[0]
        d = cfg[(pkg, c)]
        assert d, (pkg, c)
        if src_pattern is not None:
            s = src[src_pattern.format(cfg=c, ds=ds)]
            keys = [k for k in s if k not in SKIP]
            for k in keys:
                assert fmt(d[k]) == fmt(s[k]['value']), (pkg, c, k, d[k], s[k]['value'])
        elif pkg == 'objx_s':
            s = {}
            keys = [k for k in cfg[MSCA_KEYS_FROM] if k not in SKIP]
            ref = cfg.get(('msca', f'MSCA_A5_{ds}'))
            if ref:
                assert list(ref) == list(cfg[MSCA_KEYS_FROM]), ds
                for k in ref:
                    assert fmt(d[k]) == fmt(ref[k]), (ds, k, d[k], ref[k])
        else:
            s = {}
            keys = [k for k in d if k not in SKIP]
        per_ds[ds] = (c, {k: fmt(d[k]) for k in keys}, s)
    keys = list(per_ds[datasets[0]][1])
    assert all(list(v[1]) == keys for v in per_ds.values()), pkg
    indep, spec, kinds = [], [], set()
    for k in keys:
        mark = ''
        rules = {v[2][k]['rule'] for v in per_ds.values() if k in v[2]}
        if k in rule2 or 'R2-L2' in rules:
            mark = '$^{\\dagger}$'
        vv = [per_ds[ds][1][k] for ds in datasets]
        if len(set(vv)) == 1:
            indep.append(f'{tt(k)} {vv[0]}{mark}')
        else:
            spec.append(f'{tt(k)} {"/".join(vv)}{mark}')
        for v in per_ds.values():
            if k in v[2]:
                r = v[2][k]
                text = r['source']
                if r['rule'] == 'R2-L2':
                    continue
                if 'paper' in text.lower():
                    kinds.add('Paper')
                if any(t in text for t in ('README', '.yaml', '.py', 'log', 'Params', '.ini', '.sh')):
                    kinds.add('repo')
    label = ', '.join(x for x in ('Paper', 'repo') if x in kinds)
    return indep, spec, label[0].upper() + label[1:] if label else ''


def main():
    cfg, meta, src = load()
    with open(TOPK, newline='', encoding='utf-8') as f:
        topk = {(r['row'], r['dataset']): r for r in csv.DictReader(f) if r['backbone'] == 'MSCA'}
    ov = {}
    for wt in ('table1fix', 't2'):
        ov[wt] = {k: fmt(v) for k, v in cfg[('_overall', wt)].items()}
    common = ('embedding_size', 'train_batch_size', 'epochs', 'stopping_step', 'learner', 'weight_decay')
    for k in common:
        assert ov['table1fix'][k] == ov['t2'][k], k
    o = ov['t2']
    used = sorted({m[1] + ':' + m[2] for k, m in meta.items() if k[0] != '_overall'})
    digest = hashlib.sha256('\n'.join(used).encode()).hexdigest()[:16]
    lines = [
        '% Generated by tables/src/build_table_hparams.py from tables/hparams_configs.tsv, tables/hparams_sources.tsv and '
        f'tables/topk_cells.csv (sha256 of the commit:yaml-sha list {digest}). Do not edit numbers by hand.',
        '\\begin{table*}[t]',
        # R0415: 'read from the configuration files' (stated in Section supp:hparams) and the sentence on per-backbone
        # plug-in values (visible in the MG row) left the caption, now four lines (supplement page limit).
        '\\caption{Hyperparameters of MES ($\\lambda$, $K$), the anchor term of the controlled variants ($J$, $M$), plug-ins, and top-$K$ losses. '
        f'Under the common protocol, all models use {o["learner"].upper().replace("ADAMW", "AdamW")} '
        f'with weight decay {o["weight_decay"]}. '
        'Per-dataset values are listed per backbone as Baby/Pantry/Scientific/Sports, '
        'without Sports on FREEDOM and MMGCN. '
        'COHESION and SMORE take the MSCA plug-in values except $\\alpha_2$ of MG. '
        'Top-$K$ losses on MSCA: score function, '
        'temperature $\\tau$, '
        '$N$ uniform negatives, score scale $\\tau_d$ of PSL, and $K$, $\\tau_\\beta$, and quantile interval $E$ '
        'in epochs of SL@$K$. '
        'Repo: repository file or log.}',
        '\\label{tab:hparams}',
        '\\centering',
        '\\scriptsize',
        '\\setlength{\\tabcolsep}{3pt}',
        '\\renewcommand{\\arraystretch}{1.05}',
        '\\begin{tabularx}{\\textwidth}{@{}l>{\\raggedright\\arraybackslash}X>{\\raggedright\\arraybackslash}p{5.3cm}@{}}',
        '\\toprule',
        'Method & Values & Source \\\\',
        '\\midrule',
    ]
    mg_cs = OrderedDict()
    if COHSMO_MG.exists():
        with open(COHSMO_MG, newline='', encoding='utf-8') as f:
            for r in csv.DictReader(f, delimiter='\t'):
                mg_cs.setdefault(r['backbone'], {})[r['dataset']] = fmt(r['alpha2'])
    mes_cells = mes_row()
    for label, marker, pkgs, keys, source in PLUGINS:
        indep, spec = plugin_row(cfg, marker, pkgs, keys)
        if label.startswith('E + R + A'):
            # revision log 27 (decision 4): the E + R + A row has the same lambda and K as MES on the other backbones, thus one row.
            assert not spec and indep[:2] == ['$\lambda$ 0.1', '$K$ 40'], (indep, spec)
            assert mes_cells[1] == '$\lambda$ 0.1, $K$ 40 on COHESION, SMORE, MENTOR, FREEDOM, MMGCN', mes_cells
            lines.append(f'MES; E + R + A design; A: $J$, $M$ & {"; ".join(mes_cells)}; E + R + A design: {", ".join(indep)} & '
                         'This paper, seed-1 validation Recall@20 (MSCA) \\\\')
            continue
        if label == 'MG' and mg_cs:
            i = [n for n, x in enumerate(spec) if x.startswith('$\\alpha_2$: ')]
            assert len(i) == 1, spec
            for bb in ('COHESION', 'SMORE'):
                assert list(mg_cs[bb]) == DATASETS, (bb, list(mg_cs[bb]))
                spec[i[0]] += f', {bb} ' + '/'.join(mg_cs[bb][ds] for ds in DATASETS)
        lines.append(f'{label} & {", ".join(indep)}{"; " + "; ".join(spec) if spec else ""} & {source} \\\\')
    lines.append('\\midrule')
    for label, row, main_loss, keys, source in TOPK_ROWS:
        lines.append(f'{label} & {", ".join(topk_row(cfg, topk, row, main_loss, keys))} & {source} \\\\')
    lines += ['\\bottomrule', '\\end{tabularx}', '\\end{table*}', '']
    OUT.write_text('\n'.join(lines), encoding='utf-8', newline='\n')
    print('wrote', OUT)
    md = ['# Hyperparameters of the compared recommenders',
          '',
          f'Generated by tables/src/build_table_hparams.py (sha256 of the commit:yaml-sha list {digest}). '
          f'All models use embedding size {o["embedding_size"]}, batch size {o["train_batch_size"]}, at most {o["epochs"]} epochs, '
          f'patience {o["stopping_step"]}, and {o["learner"].upper().replace("ADAMW", "AdamW")} with weight decay {o["weight_decay"]} '
          'under the cosine schedule. A value that differs across datasets is listed in the order '
          'Baby/Pantry/Scientific/Sports (Baby/Pantry/Scientific for MMGCN). '
          '\u2020 Common L2 coefficient on user and item embeddings. Repo: official repository configuration, README, or released log.',
          '',
          '| Method | Values | Source |',
          '|---|---|---|']
    for label, pkg, pattern, src_pattern, fixed, rule2, datasets in MODELS:
        indep, spec, auto = model_row(cfg, src, pkg, pattern, src_pattern, rule2, datasets)
        md.append(f'| {tex2md(label)} | {tex2md(", ".join(indep + spec))} | {tex2md(fixed or auto)} |')
    OUT_MD.write_text('\n'.join(md) + '\n', encoding='utf-8', newline='\n')
    print('wrote', OUT_MD)


def tex2md(x):
    """LaTeX cell text -> Markdown (texttt, powers of ten, dagger, escaped underscores)."""
    import re
    x = re.sub(r'\\texttt\{([^}]*)\}', lambda m: '`' + m.group(1).replace('\\_', '_') + '`', x)
    x = re.sub(r'\$10\^\{(-?\d+)\}\$', r'1e\1', x)
    x = re.sub(r'\$(\d+(?:\.\d+)?)\{\\times\}10\^\{(-?\d+)\}\$', r'\1e\2', x)
    x = x.replace('$^{\\dagger}$', '\u2020').replace('\\_', '_')
    x = re.sub(r'\$([^$]*)\$', r'\1', x)
    return x


if __name__ == '__main__':
    main()
