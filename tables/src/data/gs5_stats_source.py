"""G-S5 post hoc stratified test (EDIT 19, 2026-10-16). CPU only, read-only inputs (MESC/gate/dumps from mesc_gate_dump.py on the
existing MSCA+MES and MSCA fusion checkpoints), outputs only under MESC/gate_s5/.
Strata: A = classical top-20 targets (rank_C <= 20), B = classical-miss targets (rank_C > 20).  Unit: user-level Recall@20 mass
(identical weights for both models; the two strata masses add up to the overall Recall@20).
Per dataset and stratum: per-seed MES - MSCA (unfused, base route) over seeds 1-5, two-sided paired t (sigtest.paired), Holm over
the four datasets within each stratum (sigtest.family).  Also: overall pre-fusion gap and its split, plus the complementarity numbers.
usage: python mesc_gs5_stats.py  (sigtest.py copied next to this file)"""
import json, csv, sys, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sigtest

M = '<RUN_ROOT>/study/MESC'
DS = ['baby', 'pantry', 'scientific', 'sports']
SEEDS = [1, 2, 3, 4, 5]


def load(tag, ds, sd):
    z = np.load(f'{M}/gate/dumps/{tag}__{ds}__s{sd}.npz')
    j = json.load(open(f'{M}/gate/dumps/{tag}__{ds}__s{sd}.json'))
    routes = list(z['routes'])
    fr = [r for r in routes if r.startswith('f_combo_')]
    best = max(fr, key=lambda r: (j['routes'][r]['valid_r20'], -float(r.split('_')[-1])))
    pu = z['pu']; _, inv, cnt = np.unique(pu, return_inverse=True, return_counts=True)
    w = 1.0 / cnt[inv] / len(cnt)
    H = z['H']
    return dict(w=w, rC=z['rC'], hM=H[:, routes.index('base')], hF=H[:, routes.index(best)], RM_json=j['routes']['base']['test_r20'])


def stats_one(d):
    w = d['w']; hM, hF = d['hM'], d['hF']; hA = d['rC'] <= 20; hB = ~hA
    m = lambda x: float((w * x).sum())
    RC = m(hA); conly = m(hA & ~hM)
    return dict(RM=m(hM), RM_A=m(hM & hA), RM_B=m(hM & hB), RC=RC, massB=m(hB), conly_share=conly / RC,
                capture=m(hF & hA & ~hM) / max(conly, 1e-12), RM_json=d['RM_json'])


per = {}   # per[tag][ds][seed] -> stats
for tag in ('mes', 'msca'):
    per[tag] = {ds: {sd: stats_one(load(tag, ds, sd)) for sd in SEEDS} for ds in DS}
for tag in per:
    for ds in DS:
        for sd in SEEDS:
            o = per[tag][ds][sd]
            assert abs(o['RM'] - o['RM_json']) < 5e-4, (tag, ds, sd, o['RM'], o['RM_json'])   # same measure as the cell json
            assert abs(o['RM_A'] + o['RM_B'] - o['RM']) < 1e-12

fam = {}
for key in ('RM_A', 'RM_B', 'RM'):
    pa = {ds: {sd: per['mes'][ds][sd][key] for sd in SEEDS} for ds in DS}
    pb = {ds: {sd: per['msca'][ds][sd][key] for sd in SEEDS} for ds in DS}
    fam[key] = sigtest.contrast(pa, pb, DS, fam='gs5', name=f'MES-MSCA {key}', a='MES', b='MSCA', metric=key)

# informational: does stratum A carry more of the gap than stratum B?  paired t of (A diff - B diff) over seeds
pa = {ds: {sd: per['mes'][ds][sd]['RM_A'] - per['msca'][ds][sd]['RM_A'] for sd in SEEDS} for ds in DS}
pb = {ds: {sd: per['mes'][ds][sd]['RM_B'] - per['msca'][ds][sd]['RM_B'] for sd in SEEDS} for ds in DS}
fam['AmB'] = sigtest.contrast(pa, pb, DS, fam='gs5', name='(MES-MSCA on A) - (MES-MSCA on B)', a='A', b='B', metric='AmB')
rows = []
out = {}
for i, ds in enumerate(DS):
    a, b, o, ab = fam['RM_A'][i], fam['RM_B'][i], fam['RM'][i], fam['AmB'][i]
    mes, msca = per['mes'][ds], per['msca'][ds]
    avg = lambda tag, k: float(np.mean([per[tag][ds][s][k] for s in SEEDS]))
    out[ds] = dict(
        A=dict(diff=a['diff'], p=a['p'], p_holm=a['p_holm'], reject=a['reject'], ci_lo=a['ci_lo'], ci_hi=a['ci_hi'], wins=a['wins_a'], per_seed=a['d']),
        B=dict(diff=b['diff'], p=b['p'], p_holm=b['p_holm'], reject=b['reject'], ci_lo=b['ci_lo'], ci_hi=b['ci_hi'], wins=b['wins_a'], per_seed=b['d']),
        overall=dict(diff=o['diff'], p=o['p'], p_holm=o['p_holm'], reject=o['reject'], per_seed=o['d']),
        AmB=dict(diff=ab['diff'], p=ab['p'], p_holm=ab['p_holm'], reject=ab['reject'], per_seed=ab['d']),
        share_A=a['diff'] / o['diff'], share_B=b['diff'] / o['diff'],
        massA=avg('mes', 'RC'), massB=avg('mes', 'massB'),
        mes_conly_share=avg('mes', 'conly_share'), mes_capture=avg('mes', 'capture'),
        msca_conly_share=avg('msca', 'conly_share'), msca_capture=avg('msca', 'capture'))
    for st, r in (('A', a), ('B', b), ('overall', o), ('AmB', ab)):
        rows.append(dict(dataset=ds, stratum=st, **{k: r[k] for k in ('n', 'mean_a', 'mean_b', 'diff', 'ci_lo', 'ci_hi', 't', 'p', 'm', 'p_holm', 'reject', 'wins_a')},
                         per_seed_diff=';'.join(f'{x:.6f}' for x in r['d'])))
os.makedirs(f'{M}/gate_s5', exist_ok=True)
json.dump(out, open(f'{M}/gate_s5/gs5_stats.json', 'w'), indent=1)
with open(f'{M}/gate_s5/gs5_stats.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n'); w.writeheader(); w.writerows(rows)

print('ds  stratum  diff  p  p_holm  reject wins  | per-seed diffs')
for ds in DS:
    for st in ('A', 'B', 'overall', 'AmB'):
        r = out[ds][st]
        print(ds.ljust(10), st.ljust(8), f'{r["diff"]:+.5f}', f'p={r["p"]:.4f}', f'ph={r["p_holm"]:.4f}', r['reject'], ' '.join(f'{x:+.4f}' for x in r['per_seed']))
    print(ds.ljust(10), 'share A', f'{out[ds]["share_A"]:.3f}', 'share B', f'{out[ds]["share_B"]:.3f}', 'massA', f'{out[ds]["massA"]:.4f}',
          'conly MES', f'{out[ds]["mes_conly_share"]:.4f}', 'capture', f'{out[ds]["mes_capture"]:.4f}', '| MSCA', f'{out[ds]["msca_conly_share"]:.4f}', f'{out[ds]["msca_capture"]:.4f}')
