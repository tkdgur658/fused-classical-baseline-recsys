"""revision log 18-2 (MES numbers only): one-sided empirical p of the R2NULL3 decoy null N3 per dataset (L = MES), for Holm over datasets.
Same procedure as R2NULL2_P18/n3p.py (revision log 18-3), applied to r2null3.py (sha 106df8e3): re-draws exactly the N3 replicates
(same SeedSequence([20261015 + 1000 di + s, 3]), same strata, same lists) and checks that the 2.5/97.5% quantiles and g_O equal
results/r2null3.json. Read-only on the R2NULL3 folder; writes only the given output file. CPU only.
p_N3 = (1 + #{g_null >= g_O}) / (B + 1), B = 200.  W p is copied from r2null3.json.
usage: python -s -B n3p_mes.py <out_json>"""
import os, sys, json, time
for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[v] = '8'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
G = '<RUN_ROOT>/study/R2NULL3'
OUT = sys.argv[1]
sys.argv = [sys.argv[0]]
sys.path.insert(0, G + '/code')
import numpy as np
import r2null3 as R3   # module import: its __main__ block does not run
sys.path.insert(0, '<RUN_ROOT>/R2')
sys.argv = ['r2s1', 'unused_out']
import r2s1
import torch; torch.set_num_threads(8)
t0 = time.time()
S1 = json.load(open('<RUN_ROOT>/R2/results/r2s1.json'))
REF = json.load(open(G + '/results/r2null3.json'))
ROOT = r2s1.ROOT


def find_dump_L(ds, s):
    for b in ('dumps', 'study/MES9/dumps'):
        p = f'{ROOT}/{b}/{ds}/mes__{ds}__s{s}.npz'
        if os.path.exists(p): return p
    raise FileNotFoundError(('mes', ds, s))


res = dict(source_sha=REF['code_sha256'], B=R3.B_N3, per_ds={})
dss = [d for d in R3.DSS if d != 'microlens']
for di, ds in enumerate(dss):
    assert R3.DSS.index(ds) == di
    ref = REF['per_ds'].get(ds)
    if not ref or ref.get('status') != 'OK':
        res['per_ds'][ds] = dict(status='no reference'); continue
    Ctag = S1['per_ds'][ds]['Ctag']
    D = r2s1.load_ds(ds)
    d0 = np.load(find_dump_L(ds, 1)); users = np.sort(d0['users'].astype(np.int64))
    cl, cj = r2s1.classical_lists(ds, D, users); Clist = cl[Ctag]
    udeg = np.asarray(D['X'][users].sum(1)).ravel(); nU = len(users); nI = D['nI']
    Xc = D['X'].tocoo(); train_keys = np.sort(Xc.row.astype(np.int64) * nI + Xc.col.astype(np.int64))
    per_seed = {}
    for s in R3.SEEDS:
        Z = np.load(find_dump_L(ds, s)); order = np.argsort(Z['users']); assert np.array_equal(Z['users'][order], users)
        TL = Z['top20'][order].astype(np.int64); pu, pi = Z['pu'].astype(np.int64), Z['pi'].astype(np.int64); prow = np.searchsorted(users, pu)
        rL, okp = r2s1.user_recall(TL, prow, pi, nU); rC = r2s1.user_recall(Clist, prow, pi, nU)[0]
        per_seed[s] = dict(TL=TL, prow=prow, pi=pi, okp=okp, RL=float(rL[okp].mean()), RO=float(np.maximum(rL, rC)[okp].mean()))
    okp = per_seed[1]['okp']
    mL = float(np.mean([per_seed[s]['RL'] for s in R3.SEEDS])); g_O = float(np.mean([per_seed[s]['RO'] for s in R3.SEEDS])) / mL - 1
    strat_all, groups = R3.make_strata(udeg[okp])
    strat_user = np.full(nU, -1, np.int64); strat_user[okp] = strat_all
    rngs = lambda s, kind: np.random.default_rng(np.random.SeedSequence([R3.BASE_SEED + 1000 * di + s, kind]))
    RLb3, ROb3 = [], []
    for s in R3.SEEDS:
        ps = per_seed[s]; m = okp[ps['prow']]
        a, c, u = R3.n3_decoy(ps['prow'][m], ps['pi'][m], ps['TL'], Clist, nU, strat_user, train_keys, nI, R3.B_N3, rngs(s, 3))
        RLb3.append(a); ROb3.append(c)
    gn3 = np.mean(ROb3, 0) / np.mean(RLb3, 0) - 1
    cls = R3.classify(g_O, gn3)
    ok = abs(cls['lo'] - ref['N3']['lo']) < 1e-12 and abs(cls['hi'] - ref['N3']['hi']) < 1e-12 and abs(g_O - ref['g_O']) < 1e-12 and cls['cls'] == ref['N3']['cls']
    n_ge = int((gn3 >= g_O).sum())
    p = (1 + n_ge) / (R3.B_N3 + 1)
    res['per_ds'][ds] = dict(status='OK', g_O=g_O, n3_lo=cls['lo'], n3_hi=cls['hi'], n3_cls=cls['cls'], n3_max=float(gn3.max()), n_ge=n_ge,
                             p_n3=p, reproduces_r2null3=bool(ok), W_p=(ref['W'] or {}).get('p'))
    print(f'[{time.time() - t0:6.1f}s] {ds} g_O {g_O:.4f} N3 [{cls["lo"]:.4f},{cls["hi"]:.4f}] {cls["cls"]} max {gn3.max():.4f} n_ge {n_ge} p {p:.4f} repro {ok}', flush=True)
    json.dump(res, open(OUT, 'w'), indent=1)
res['sec'] = round(time.time() - t0, 1)
json.dump(res, open(OUT, 'w'), indent=1)
print('done', res['sec'])
