"""CLASSIC E1a (2026-10-16, thesis B scope (i)): standalone training-free rankers of one dataset, deterministic (no seed).
Rankers: rp3b (RP3beta), iknn (ItemKNN-CBF text), icf (ItemKNN-CF), ease (EASE), F = z(rp3b) + z(iknn) (the fused-baseline
classical part), mix = best validation z-mix  sum_r w_r z(r)  over (rp3b, iknn, icf, ease), w in {0,1,2}^4 (65 vectors).
Selection on validation Recall@20 only: rp3b / icf = W1 chosen configs (W1/classic/<ds>/classic__<ds>.json; MicroLens: the same
W1 grids run here), EASE lambda from LAMS = {10, 30, 100, 300, 1000, 3000, 10000}, mix weights from the 65-vector grid (ties -> first).
Evaluation convention = tf_fuse.py / f_standalone.py: training items masked, validation and test items remain candidates, users with
test targets and training history, Recall@20 = hits / #targets, NDCG@20 with ideal DCG of min(#targets, 20), top-20 ties -> lower item
index; tie-adjusted Recall@20 of W1 (expected value under random tie order) as a gate against W1 test_r20.
usage: python -s -B classic_ds.py <dataset> [--threads 4] [--chunk 512]
writes only under study/CLASSIC/<dataset>/ : ops_rp3b.npz, ops_icf.npz, ops_cbf.npz, ease_B.npy (float32), classic_<ds>.json
"""
import argparse, json, os, sys, time
ap = argparse.ArgumentParser(); ap.add_argument('dataset')
ap.add_argument('--threads', type=int, default=4); ap.add_argument('--chunk', type=int, default=512)
a = ap.parse_args()
for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[v] = str(a.threads)
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import scipy.sparse as sp
import torch
torch.set_num_threads(a.threads)
import classic_common as C

DS = a.dataset; t0 = time.time()
OD = f'{C.CLASSIC}/{DS}'; os.makedirs(OD, exist_ok=True)
LOG = lambda *x: print(time.strftime('%T'), DS, *x, f'[{time.time() - t0:.0f}s]', flush=True)   # noqa: E731
D = C.load_data(DS); X, nU, nI, du, users = D['X'], D['nU'], D['nI'], D['du'], D['users']
LOG('loaded', nU, nI, X.nnz, 'test users', len(users))
GRID = {'itemknn_cf': [dict(topK=50, shrink=0.0), dict(topK=50, shrink=10.0), dict(topK=200, shrink=0.0), dict(topK=200, shrink=10.0)],
        'rp3b': [dict(alpha=0.5, beta=0.3, topK=200), dict(alpha=0.5, beta=0.6, topK=200),
                 dict(alpha=1.0, beta=0.3, topK=200), dict(alpha=1.0, beta=0.6, topK=200)]}

# ---------------------------------------------------------------- 1. configs of rp3b / icf (W1) and operators
W1J = f'{C.S4R}/W1/classic/{DS}/classic__{DS}.json'
df = D['df']; uf, itf = D['uf'], D['itf']
va = df[df.x_label == 1]; te = df[df.x_label == 2]
va = va[du[va[uf].values] > 0]; te = te[du[te[uf].values] > 0]
vk = np.unique(va[uf].values.astype(np.int64) * 10_000_000 + va[itf].values); tk = np.unique(te[uf].values.astype(np.int64) * 10_000_000 + te[itf].values)
va_u, va_i = (vk // 10_000_000).astype(np.int64), (vk % 10_000_000).astype(np.int64)
te_u, te_i = (tk // 10_000_000).astype(np.int64), (tk % 10_000_000).astype(np.int64)
va_users, te_users = np.unique(va_u), np.unique(te_u)
cfg, w1test, grid_rows = {}, {}, {}
if os.path.exists(W1J):
    M1 = json.load(open(W1J))['methods']
    cfg = {'rp3b': M1['rp3b']['chosen'], 'icf': M1['itemknn_cf']['chosen']}
    w1test = {'rp3b': M1['rp3b']['test_r20'], 'icf': M1['itemknn_cf']['test_r20']}
    src_cfg = 'W1 classic json'
else:
    src_cfg = 'grid run here (W1 grids, tie-adjusted validation Recall@20)'
    for meth, key in (('rp3b', 'rp3b'), ('itemknn_cf', 'icf')):
        best = None; rows = []
        for c in GRID[meth]:
            W = C.rp3b(X, **c) if meth == 'rp3b' else C.itemknn_cf(X, **c)
            vpr, vpt, _ = C.evaluate(X, W, va_u, va_i, va_users, nI)
            rv = C.recall_tieadj(va_u, vpr, vpt); rows.append(dict(cfg=c, valid_r20=rv))
            LOG('VALID', meth, c, round(rv, 5))
            if best is None or rv > best[0]:
                best = (rv, c, W)
        cfg[key] = best[1]; grid_rows[key] = rows
        pr, pt, _ = C.evaluate(X, best[2], te_u, te_i, te_users, nI)
        w1test[key] = C.recall_tieadj(te_u, pr, pt)
W_rp = C.rp3b(X, **cfg['rp3b']).tocsr(); W_cf = C.itemknn_cf(X, **cfg['icf']).tocsr(); A_cbf = C.build_cbf(DS, nI)
sp.save_npz(f'{OD}/ops_rp3b.npz', W_rp); sp.save_npz(f'{OD}/ops_icf.npz', W_cf); sp.save_npz(f'{OD}/ops_cbf.npz', A_cbf)
OPS = {'rp3b': W_rp, 'iknn': A_cbf, 'icf': W_cf}
LOG('operators', cfg, src_cfg)

# ---------------------------------------------------------------- 2. EASE: lambda on validation
G = np.asarray((X.T @ X).todense(), dtype=np.float64)
X32 = X.astype(np.float32).tocsr()
vusers = np.array(sorted(u for u in D['VAL'] if du[u] > 0 and u in D['TST']), np.int64)   # users of the evaluation set with validation targets


def val_recall_ease(Bm, chunk=a.chunk):
    rv = []
    for st in range(0, len(vusers), chunk):
        ids = vusers[st:st + chunk]; Xb = X32[ids]
        S = np.asarray(Xb @ Bm, dtype=np.float64)
        mask = Xb.toarray() > 0
        S[mask] = -np.inf
        T = torch.topk(torch.from_numpy(S), C.K, dim=1).indices.numpy()
        tg = C.chunk_targets(ids, D)
        acc = C.new_acc(); C.add_val(acc, T, tg, len(ids)); rv.extend(acc['rv'])
    return float(np.mean(rv))


ease_rows, bestB, best = [], None, None
for lam in C.LAMS:
    t1 = time.time()
    Bm = C.ease_inverse(G, lam)
    rv = val_recall_ease(Bm)
    ease_rows.append(dict(lam=lam, valid_r20=rv, sec=round(time.time() - t1, 1)))
    LOG('EASE', lam, round(rv, 5))
    if best is None or rv > best[0]:
        best = (rv, lam); bestB = Bm
    del Bm
del G
lam_star = best[1]
np.save(f'{OD}/ease_B.npy', bestB)
LOG('EASE chosen lambda', lam_star, 'valid R@20', round(best[0], 5), 'edge' if lam_star in (C.LAMS[0], C.LAMS[-1]) else 'interior')

# ---------------------------------------------------------------- 3. z-mix weights on validation
MG = C.mix_grid()
macc = [C.new_acc() for _ in MG]


def chunk_scores(ids):
    Xb = X[ids]; mask = Xb.toarray() > 0
    Cs = {r: np.asarray((Xb @ OPS[r]).todense(), dtype=np.float64) for r in ('rp3b', 'iknn', 'icf')}
    Cs['ease'] = np.asarray(X32[ids] @ bestB, dtype=np.float64)
    Z0 = {r: np.where(mask, 0.0, C.zrow(Cs[r], mask)) for r in C.RANKERS}
    return mask, Cs, Z0


for st in range(0, len(vusers), a.chunk):
    ids = vusers[st:st + a.chunk]
    mask, Cs, Z0 = chunk_scores(ids)
    tg = C.chunk_targets(ids, D)
    for gi, w in enumerate(MG):
        S = sum(wr * Z0[r] for wr, r in zip(w, C.RANKERS) if wr)
        S = np.where(mask, -np.inf, S)
        C.add_val(macc[gi], C.top20(S), tg, len(ids))
mixv = [float(np.mean(m['rv'])) for m in macc]
gi_star = int(np.argmax(mixv))          # first maximum (ties -> first vector)
w_star = MG[gi_star]
order = np.argsort(-np.array(mixv), kind='stable')[:5]
LOG('MIX chosen', w_star, 'valid R@20', round(mixv[gi_star], 5))

# ---------------------------------------------------------------- 4. final standalone evaluation
R = ('rp3b', 'iknn', 'icf', 'ease', 'F', 'mix')
acc = {r: C.new_acc() for r in R}
for st in range(0, len(users), a.chunk):
    ids = users[st:st + a.chunk]
    mask, Cs, Z0 = chunk_scores(ids)
    tg = C.chunk_targets(ids, D)
    Sd = {r: np.where(mask, -np.inf, Cs[r]) for r in C.RANKERS}
    Sd['F'] = np.where(mask, -np.inf, Z0['rp3b'] + Z0['iknn'])
    Sd['mix'] = np.where(mask, -np.inf, sum(wr * Z0[r] for wr, r in zip(w_star, C.RANKERS) if wr))
    for r in R:
        T = C.top20(Sd[r]); C.add_val(acc[r], T, tg, len(ids)); C.add_test(acc[r], T, tg, len(ids), S=Sd[r])
res = {r: C.summarize(acc[r]) for r in R}
gates = {k: dict(mine=res[k]['test_r20_tieadj'], w1=w1test[k], ok=bool(abs(res[k]['test_r20_tieadj'] - w1test[k]) <= 5e-4)) for k in ('rp3b', 'icf')}
step1 = f'{C.S4R}/study/STEP1/out/f_standalone_{DS}.json'
if os.path.exists(step1):
    s1 = json.load(open(step1))['rankers']
    gates['step1'] = {k0: dict(mine=res[k1]['test_r20'], step1=s1[k0]['test_r20'], ok=bool(abs(res[k1]['test_r20'] - s1[k0]['test_r20']) <= 1e-6))
                      for k0, k1 in (('rp3b', 'rp3b'), ('iknn', 'iknn'), ('combo', 'F'))}
out = dict(dataset=DS, code_sha256=C.sha256_file(os.path.abspath(__file__)), common_sha256=C.sha256_file(C.__file__),
           n_users=nU, n_items=nI, n_train=int(X.nnz), n_test_users=int(len(users)), cfg_source=src_cfg,
           rp3b_cfg=cfg['rp3b'], icf_cfg=cfg['icf'], cbf_feat=C.feat_path(DS), grid_rows=grid_rows,
           ease_lams=C.LAMS, ease_grid=ease_rows, ease_lam=lam_star, ease_lam_at_edge=bool(lam_star in (C.LAMS[0], C.LAMS[-1])),
           mix_rankers=list(C.RANKERS), mix_grid_n=len(MG), mix_w=list(w_star),
           mix_top5=[dict(w=list(MG[i]), valid_r20=mixv[i]) for i in order],
           rankers=res, gates=gates, ease_B=f'{OD}/ease_B.npy', sec=round(time.time() - t0, 1))
json.dump(out, open(f'{OD}/classic_{DS}.json', 'w'), indent=1)
LOG('DONE', {r: round(v['test_r20'], 4) for r, v in res.items()}, 'gates', {k: (v['ok'] if 'ok' in v else {kk: vv['ok'] for kk, vv in v.items()}) for k, v in gates.items()})
