"""CLASSIC (2026-10-16, thesis B scope (i)): shared helpers of classic_ds.py (E1a) and tf_fuse_ce.py (E1b).
Operators rp3b / itemknn_cf / topk_rows / offdiag / evaluate / recall are copied verbatim from W1/code/w1_classic.py (sha 1672385e...),
the CBF operator and zrow from TABLE_FUSION/code/tf_fuse.py (sha 2399e9d0...) and STEP1/code/f_standalone.py.
New: EASE operator (Steck 2019), top20() = deterministic top-20 (ties -> lower item index; identical to the stable argsort of
f_standalone.py, but fast), metric accumulation.  Training-free, CPU only.
"""
import os, json, hashlib
import numpy as np
import scipy.sparse as sp
import pandas as pd

S4R = '<RUN_ROOT>'
DR = '<RUN_ROOT_B>/P04_data/canonical_v2'
CLASSIC = f'{S4R}/study/CLASSIC'
K = 20
LOGD = np.log2(np.arange(2, K + 2))
ICUM = np.concatenate([[1.0], np.cumsum(1.0 / LOGD)])
RANKERS = ('rp3b', 'iknn', 'icf', 'ease')
LAMS = [10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0]
WS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5]        # fusion weight grid of tf_fuse.py


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def mix_grid():
    """z-mix weights over (rp3b, iknn, icf, ease): {0,1,2}^4 without the zero vector and without all-even duplicates (65 vectors)"""
    import itertools
    return [w for w in itertools.product((0, 1, 2), repeat=4) if sum(w) > 0 and not all(x % 2 == 0 for x in w)]


# ------------------------------------------------------------------ operators (verbatim from w1_classic.py)
def offdiag(M):
    M = M.tocoo(); keep = M.row != M.col
    return sp.csr_matrix((M.data[keep], (M.row[keep], M.col[keep])), shape=M.shape)


def topk_rows(M, k):
    M = M.tocsr(); M.sort_indices()
    rows, cols, vals = [], [], []
    for r in range(M.shape[0]):
        a0, a1 = M.indptr[r], M.indptr[r + 1]
        if a1 == a0:
            continue
        c = M.indices[a0:a1]; v = M.data[a0:a1]
        if len(v) > k:
            o = np.argsort(-v, kind='stable')[:k]
            c, v = c[o], v[o]
        rows.append(np.full(len(c), r)); cols.append(c); vals.append(v)
    if not rows:
        return sp.csr_matrix(M.shape)
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=M.shape)


def itemknn_cf(X, topK, shrink):
    G = offdiag((X.T @ X).astype(np.float64)); G.eliminate_zeros()
    nrm = np.sqrt(np.asarray(X.sum(0)).ravel())
    G = G.tocoo()
    S = sp.csr_matrix((G.data / (nrm[G.row] * nrm[G.col] + shrink), (G.row, G.col)), shape=G.shape)
    T = topk_rows(S, topK)
    return T.T.tocsr()


def rp3b(X, alpha, beta, topK):
    du = np.asarray(X.sum(1)).ravel(); di = np.asarray(X.sum(0)).ravel()
    Pui = sp.diags(1.0 / np.maximum(du, 1)) @ X
    Piu = sp.diags(1.0 / np.maximum(di, 1)) @ X.T.tocsr()
    Pui = Pui.tocsr(); Piu = Piu.tocsr()
    Pui.data = Pui.data ** alpha; Piu.data = Piu.data ** alpha
    W = (Piu @ Pui).tocsr()
    W = W @ sp.diags(1.0 / np.power(np.maximum(di, 1), beta))
    W = offdiag(W); W.eliminate_zeros()
    return topk_rows(W, topK)


def evaluate(X, W, pos_u, pos_i, users, n_items, bs=1024):
    """W1 evaluator: ranks (tie-adjusted) of positive pairs under score = X @ W, training items excluded"""
    pr = np.zeros(len(pos_u), np.int64); pt = np.zeros(len(pos_u), np.int64); ps = np.zeros(len(pos_u))
    starts = np.searchsorted(pos_u, users); ends = np.searchsorted(pos_u, users, side='right')
    for s0 in range(0, len(users), bs):
        ub = users[s0:s0 + bs]
        Sc = np.asarray((X[ub] @ W).todense(), dtype=np.float64)
        Xb = X[ub].tocoo()
        Sc[Xb.row, Xb.col] = -np.inf
        for k in range(len(ub)):
            a0, a1 = starts[s0 + k], ends[s0 + k]
            if a1 <= a0:
                continue
            row = Sc[k]; v = row[pos_i[a0:a1]]
            assert np.all(np.isfinite(v)), 'positive item is a training item'
            pr[a0:a1] = (row[None, :] > v[:, None]).sum(1) + 1
            pt[a0:a1] = (row[None, :] == v[:, None]).sum(1) - 1
            ps[a0:a1] = v
    return pr, pt, ps


def recall_tieadj(pu, pr, pt):
    h = np.clip((K - pr + 1) / (pt + 1), 0, 1)
    us, inv = np.unique(pu, return_inverse=True)
    return float((np.bincount(inv, weights=h) / np.bincount(inv)).mean())


# ------------------------------------------------------------------ data
def load_data(ds):
    df = pd.read_csv(f'{DR}/{ds}/{ds}.inter', sep='\t')
    uf = [c for c in df.columns if c.lower().startswith('user')][0]; itf = [c for c in df.columns if c.lower().startswith('item')][0]
    nU, nI = int(df[uf].max()) + 1, int(df[itf].max()) + 1
    trd, vad, ted = df[df.x_label == 0], df[df.x_label == 1], df[df.x_label == 2]
    X = sp.csr_matrix((np.ones(len(trd)), (trd[uf].values, trd[itf].values)), shape=(nU, nI)); X.sum_duplicates(); X.data[:] = 1.0
    X.sort_indices()
    VAL = vad.groupby(uf)[itf].apply(lambda s: np.asarray(s, np.int64)).to_dict()
    TST = ted.groupby(uf)[itf].apply(lambda s: np.asarray(s, np.int64)).to_dict()
    du = np.asarray(X.sum(1)).ravel()
    users = np.array(sorted(u for u in TST if du[u] > 0), np.int64)
    return dict(ds=ds, X=X, nU=nU, nI=nI, VAL=VAL, TST=TST, du=du, users=users, df=df, uf=uf, itf=itf)


def feat_path(ds):
    p = f'{S4R}/feat/{ds}_feat.npz'
    return p if os.path.exists(p) else f'{S4R}/W2/feat/{ds}_feat.npz'


def build_cbf(ds, nI):
    """ItemKNN-CBF(text) operator exactly as f_standalone.py / tf_fuse.py: kNN-10 graph of the L2-normalised text features
    (txt_knn of the feat npz), cosine weights (positive part), symmetrised by max, no diagonal."""
    F = dict(np.load(feat_path(ds)))
    assert len(F['item_deg']) == nI, (len(F['item_deg']), nI)
    Tx = np.load(f'{DR}/{ds}/text_feat.npy').astype(np.float32); nr = np.linalg.norm(Tx, axis=1, keepdims=True); nr[nr == 0] = 1; Tx /= nr
    knn = F['txt_knn'].astype(np.int64); rr = np.repeat(np.arange(nI), knn.shape[1]); cc = knn.ravel()
    wv = np.maximum((Tx[rr] * Tx[cc]).sum(1), 0).astype(np.float64)
    A = sp.csr_matrix((wv, (rr, cc)), shape=(nI, nI)); A = A.maximum(A.T).tocsr(); A = offdiag(A); A.eliminate_zeros()
    return A


def ease_inverse(G, lam):
    """EASE closed form (Steck 2019): P = (X^T X + lam I)^-1, B = I - P diag(1/diag P) (zero diagonal), B[i,j] = -P[i,j] / P[j,j].
    float64 Cholesky inverse with torch (scipy/OpenBLAS dpotrf fails spuriously for the 26k-item Office matrix, see PREREG amendment 1);
    residual check of 32 fixed rows, max |A P - I| < 1e-8."""
    import torch
    n = G.shape[0]
    A = torch.from_numpy(G).clone(); A.diagonal().add_(lam)
    L, info = torch.linalg.cholesky_ex(A)
    assert int(info) == 0, int(info)
    P = torch.cholesky_inverse(L); del L
    g = torch.Generator().manual_seed(0); idx = torch.randperm(n, generator=g)[:32]
    E = torch.zeros(len(idx), n, dtype=P.dtype); E[torch.arange(len(idx)), idx] = 1.0
    res = float((A[idx] @ P - E).abs().max()); del A, E
    assert res < 1e-8, res
    d = torch.diagonal(P).clone()
    P.div_(d[None, :]).neg_(); P.diagonal().zero_()
    out = P.float().numpy(); del P
    return out


# ------------------------------------------------------------------ scoring / metrics
def zrow(S, mask):
    """per-row z over eligible entries (mask True = training item); masked entries -> -inf (population sd, clamp 1e-12)"""
    cnt = (~mask).sum(1, keepdims=True).astype(np.float64)
    S0 = np.where(mask, 0.0, S)
    mu = S0.sum(1, keepdims=True) / cnt
    sd = np.sqrt(np.where(mask, 0.0, (S - mu) ** 2).sum(1, keepdims=True) / cnt)
    Z = (S - mu) / np.maximum(sd, 1e-12)
    Z[mask] = -np.inf
    return Z


def top20(S):
    """S float64 (B, nI), training items = -inf. Top-20 item ids per row, descending score, ties broken by the lower item index
    (= np.argsort(-S, kind='stable')[:, :20]). torch.topk first, rows with ties inside or at the border of the list are redone."""
    import torch
    t = torch.topk(torch.from_numpy(S), K, dim=1)
    v = t.values.numpy(); T = t.indices.numpy().astype(np.int64)
    thr = v[:, -1:]
    n_gt = (S > thr).sum(1); n_eq = (S == thr).sum(1)
    bad = (n_eq > (K - n_gt)) | (v[:, 1:] == v[:, :-1]).any(1)
    if bad.any():
        idx = np.flatnonzero(bad)
        T[idx] = np.argsort(-S[idx], axis=1, kind='stable')[:, :K]
    return T


def chunk_targets(ids, D):
    VAL, TST = D['VAL'], D['TST']
    vr_, vi_, tr_, ti_ = [], [], [], []
    for k, u in enumerate(ids.tolist()):
        if u in VAL:
            vv = VAL[u]; vr_.append(np.full(len(vv), k)); vi_.append(vv)
        if u in TST:
            tt = TST[u]; tr_.append(np.full(len(tt), k)); ti_.append(tt)
    z = np.zeros(0, np.int64)
    cat = lambda a: np.concatenate(a) if a else z   # noqa: E731
    return cat(vr_), cat(vi_), cat(tr_), cat(ti_)


def new_acc():
    return dict(rv=[], rt=[], nd=[], rta=[])


def add_val(acc, T, tg, B):
    vr_, vi_ = tg[0], tg[1]
    if len(vr_):
        hv = (T[vr_] == vi_[:, None]).any(1)
        nv = np.bincount(vr_, minlength=B); hs = np.bincount(vr_, weights=hv, minlength=B)
        acc['rv'].extend((hs[nv > 0] / nv[nv > 0]).tolist())


def add_test(acc, T, tg, B, S=None):
    """test Recall@20 and NDCG@20 (MMRec convention: ideal DCG of min(#targets, 20)); S given -> tie-adjusted recall too"""
    tr_, ti_ = tg[2], tg[3]
    if not len(tr_):
        return
    ht = (T[tr_] == ti_[:, None])
    nt = np.bincount(tr_, minlength=B); hs = np.bincount(tr_, weights=ht.any(1), minlength=B)
    acc['rt'].extend((hs[nt > 0] / nt[nt > 0]).tolist())
    dcg = np.bincount(tr_, weights=(ht / LOGD[None, :]).sum(1), minlength=B)
    idcg = ICUM[np.minimum(nt, K)]
    acc['nd'].extend((dcg[nt > 0] / idcg[nt > 0]).tolist())
    if S is not None:
        Sg = S[tr_]; v = Sg[np.arange(len(tr_)), ti_]
        pr = (Sg > v[:, None]).sum(1) + 1; pt = (Sg == v[:, None]).sum(1) - 1
        h = np.clip((K - pr + 1) / (pt + 1), 0, 1)
        hs = np.bincount(tr_, weights=h, minlength=B)
        acc['rta'].extend((hs[nt > 0] / nt[nt > 0]).tolist())


def summarize(acc):
    return dict(valid_r20=float(np.mean(acc['rv'])), test_r20=float(np.mean(acc['rt'])), test_n20=float(np.mean(acc['nd'])),
                test_r20_tieadj=(float(np.mean(acc['rta'])) if acc['rta'] else None))
