"""P1 (thesis B scope (i), CLASSIC, 2026-10-16): same-checkpoint fusion variant of the fusion blocks of Tables sota and fuse_added.

Nothing printed in the manuscript changes.  This script only prepares the data that would let the table generators print the plain
MSCA row from the SAME checkpoints as the fused MSCA (+)F row (the checkpoints of the TABLE_FUSION runs), and reports how the unfused
MSCA + MES minus MSCA numbers change:
  tables/src/data/classic_p1_msca.csv    one row per (dataset, seed): MSCA of Table I (table1_perseed.csv, row Native), MSCA of the
                                         fusion run (fusion_cells_9ds_mes.csv, base = plain MSCA, fused = MSCA (+)F), MSCA + MES of
                                         Table I (mes_perseed.csv, row MES) and of the fusion run, same-checkpoint flags
  tables/src/data/classic_p1_summary.json  per-dataset means and the three paired contrasts (sigtest.py: two-sided paired t over
                                         seeds 1-5, Holm over the datasets of the contrast)
Contrasts (Recall@20, MSCA + MES minus MSCA):
  current   MES of Table I (mes_perseed.csv) - MSCA of Table I (table1_perseed.csv)       [datasets of Table I]
  p1_t1mes  MES of Table I - MSCA of the fusion run (plain MSCA row from the fusion checkpoints)
  p1_fus    MES of the fusion run - MSCA of the fusion run (= the pre-fusion margin of sgPre*)
Usage: python -s tables/src/build_p1_classic.py
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sigtest  # noqa: E402

T = Path(__file__).resolve().parents[1]
DATA = Path(__file__).resolve().parent / 'data'
DS6 = ['baby', 'pantry', 'scientific', 'sports', 'clothing', 'instruments']
DS4 = ['baby', 'pantry', 'scientific', 'sports']
SEEDS = [1, 2, 3, 4, 5]


def read(p):
    return list(csv.DictReader(open(p, newline='', encoding='utf-8')))


t1 = {(r['dataset'], int(r['seed'])): r for r in read(T / 'table1_perseed.csv') if r['backbone'] == 'MSCA' and r['row'] == 'Native'}
t1p = {}
for r in read(T / 'table1_runs.tsv') if False else csv.DictReader(open(T / 'table1_runs.tsv', newline='', encoding='utf-8'), delimiter='\t'):
    if r['backbone'] == 'MSCA' and r['row'] == 'Native' and r['chosen'] == '1':
        t1p[(r['dataset'], int(r['seed']))] = r['path'].rsplit('/', 1)[0]
mes = {(r['dataset'], int(r['seed'])): r for r in read(T / 'mes_perseed.csv') if r['backbone'] == 'MSCA' and r['row'] == 'MES'}
fus = {(r['model'], r['dataset'], int(r['seed'])): r for r in read(DATA / 'fusion_cells_9ds_mes.csv') if r['gate_ok'] == '1' and r['fused_r20']}

rows = []
for d in DS6:
    for s in SEEDS:
        a = t1[(d, s)]; fm = fus[('MSCA', d, s)]; fe = fus[('MSCA+MES', d, s)]; m = mes.get((d, s))
        # same checkpoint = same run directory, or Recall@20 and NDCG@20 equal at the four printed decimals
        same_msca = fm['source'].rstrip('/') == t1p[(d, s)].rstrip('/') or (
            abs(float(a['r20']) - round(float(fm['base_r20']), 4)) < 1e-9 and abs(float(a['n20']) - round(float(fm['base_n20']), 4)) < 1e-9)
        rows.append(dict(
            dataset=d, seed=s, t1_msca_r20=a['r20'], t1_msca_n20=a['n20'], t1_msca_run=t1p[(d, s)],
            fus_msca_r20=fm['base_r20'], fus_msca_n20=fm['base_n20'], fus_msca_fused_r20=fm['fused_r20'], fus_msca_fused_n20=fm['fused_n20'],
            fus_msca_w=fm['weight'], fus_msca_run=fm['source'],
            same_msca_ckpt=int(same_msca),
            t1_mes_r20=(m or {}).get('r20', ''), t1_mes_n20=(m or {}).get('n20', ''),
            fus_mes_r20=fe['base_r20'], fus_mes_n20=fe['base_n20'], fus_mes_fused_r20=fe['fused_r20'], fus_mes_fused_n20=fe['fused_n20'],
            same_mes_ckpt=int(m is not None and abs(float(m['r20']) - round(float(fe['base_r20']), 4)) < 1e-9)))
cols = list(rows[0])
with open(DATA / 'classic_p1_msca.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, cols, lineterminator='\n'); w.writeheader(); w.writerows(rows)


def per(key_a, ds):
    return {d: {r['seed']: float(r[key_a]) for r in rows if r['dataset'] == d and r[key_a] != ''} for d in ds}


def con(name, a, b, ds):
    res = sigtest.contrast(per(a, ds), per(b, ds), ds, fam='P1', name=name, a=a, b=b, metric='r20')
    return [dict(dataset=r['dataset'], mean_a=r['mean_a'], mean_b=r['mean_b'], diff=r['diff'], p=r['p'], p_holm=r['p_holm'],
                 reject=r['reject'], wins=r['wins_a'], n=r['n']) for r in res]


out = {'rule': sigtest.RULE_TEXT, 'datasets_table1': DS4, 'datasets_fusion': DS6,
       'same_ckpt_msca': {d: sum(r['same_msca_ckpt'] for r in rows if r['dataset'] == d) for d in DS6},
       'same_ckpt_mes': {d: sum(r['same_mes_ckpt'] for r in rows if r['dataset'] == d) for d in DS6},
       'msca_means': {d: dict(t1=sum(float(r['t1_msca_r20']) for r in rows if r['dataset'] == d) / 5,
                               fusion_plain=sum(float(r['fus_msca_r20']) for r in rows if r['dataset'] == d) / 5,
                               fusion_fused=sum(float(r['fus_msca_fused_r20']) for r in rows if r['dataset'] == d) / 5) for d in DS6},
       'contrasts': {
           'current_t1mes_minus_t1msca_DS4': con('MES(Table I) - MSCA(Table I)', 't1_mes_r20', 't1_msca_r20', DS4),
           'p1_t1mes_minus_fusmsca_DS4': con('MES(Table I) - MSCA(fusion run)', 't1_mes_r20', 'fus_msca_r20', DS4),
           'p1_fusmes_minus_fusmsca_DS4': con('MES(fusion run) - MSCA(fusion run)', 'fus_mes_r20', 'fus_msca_r20', DS4),
           'p1_fusmes_minus_fusmsca_DS6': con('MES(fusion run) - MSCA(fusion run)', 'fus_mes_r20', 'fus_msca_r20', DS6),
           'current_fusmes_minus_t1msca_DS6': con('MES(fusion run) - MSCA(Table I)', 'fus_mes_r20', 't1_msca_r20', DS6),
       }}
(DATA / 'classic_p1_summary.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
print(json.dumps({k: out[k] for k in ('same_ckpt_msca', 'same_ckpt_mes')}))
for k, v in out['contrasts'].items():
    print(k)
    for r in v:
        print('  %-11s a=%.4f b=%.4f diff=%+.4f p=%.3f p_holm=%.3f %s' % (r['dataset'], r['mean_a'], r['mean_b'], r['diff'], r['p'], r['p_holm'], 'SIG' if r['reject'] else ''))
