"""STEP1 (EDIT 18-1, 2026-10-16): standalone (classical-only) Recall@20 / NDCG@20 of the two classical rankers of F and of
their z-score combination, with the operators of TABLE_FUSION/code/tf_fuse.py (copied verbatim below):
  rp3b  = RP3beta of W1 w1_classic.py at the validation-chosen config of W1/classic/<ds>/classic__<ds>.json
          (grid alpha {0.5, 1.0} x beta {0.3, 0.6}, topK 200, chosen on validation Recall@20, tie-adjusted)
  iknn  = ItemKNN-CBF(text): s(u,i) = sum_{j in H_u} A[j,i], A = kNN-10 graph of the L2-normalised text features
          (feat npz txt_knn, s4r_feat.py), cosine weights (positive part), symmetrised by max, no diagonal; no tuning
  combo = z(rp3b) + z(iknn), z per user over the eligible (non-training) items, population sd
CPU only, deterministic, no checkpoint. Evaluation as tf_fuse.py (training items masked, validation and test items
candidates, users with test targets, Recall@20 = hits / #targets, NDCG@20 with ideal DCG of min(#targets, 20) targets,
mean over users). Top-20 lists: stable argsort of -score, i.e. ties broken by the lower item index. Also reported:
tie-adjusted Recall@20 of W1 (expected value under random tie order) as a gate against W1 test_r20 for rp3b.
usage: python -s -B f_standalone.py <dataset> <out.json> [--threads 4] [--chunk 512]   (writes only <out.json>)
"""
import argparse, hashlib, json, os, sys, time
ap = argparse.ArgumentParser(); ap.add_argument('dataset'); ap.add_argument('out')
ap.add_argument('--threads', type=int, default=4); ap.add_argument('--chunk', type=int, default=512)
a = ap.parse_args()
for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[v] = str(a.threads)
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np, scipy.sparse as sp, pandas as pd
S4R = '<RUN_ROOT>'
DR = '<RUN_ROOT_B>/P04_data/canonical_v2'
DS = a.dataset; K = 20; t0 = time.time()
FEAT = f'{S4R}/feat/{DS}_feat.npz'
if not os.path.exists(FEAT):
    FEAT = f'{S4R}/W2/feat/{DS}_feat.npz'
CLS = f'{S4R}/W1/classic/{DS}/classic__{DS}.json'
LOGD = np.log2(np.arange(2, K + 2)); ICUM = np.concatenate([[1.0], np.cumsum(1.0 / LOGD)])


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


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


def zrow(S, mask):
    cnt = (~mask).sum(1, keepdims=True).astype(np.float64)
    S0 = np.where(mask, 0.0, S)
    mu = S0.sum(1, keepdims=True) / cnt
    sd = np.sqrt(np.where(mask, 0.0, (S - mu) ** 2).sum(1, keepdims=True) / cnt)
    Z = (S - mu) / np.maximum(sd, 1e-12)
    Z[mask] = -np.inf
    return Z


df = pd.read_csv(f'{DR}/{DS}/{DS}.inter', sep='\t')
uf = [c for c in df.columns if c.lower().startswith('user')][0]; itf = [c for c in df.columns if c.lower().startswith('item')][0]
nU, nI = int(df[uf].max()) + 1, int(df[itf].max()) + 1
trd, vad, ted = df[df.x_label == 0], df[df.x_label == 1], df[df.x_label == 2]
X = sp.csr_matrix((np.ones(len(trd)), (trd[uf].values, trd[itf].values)), shape=(nU, nI)); X.sum_duplicates(); X.data[:] = 1.0
X.sort_indices()
F = dict(np.load(FEAT)); assert len(F['item_deg']) == nI, (len(F['item_deg']), nI)
C1 = json.load(open(CLS))['methods']
W_rp = rp3b(X, **C1['rp3b']['chosen']).tocsr()
Tx = np.load(f'{DR}/{DS}/text_feat.npy').astype(np.float32); nr = np.linalg.norm(Tx, axis=1, keepdims=True); nr[nr == 0] = 1; Tx /= nr
knn = F['txt_knn'].astype(np.int64); rr = np.repeat(np.arange(nI), knn.shape[1]); cc = knn.ravel()
wv = np.maximum((Tx[rr] * Tx[cc]).sum(1), 0).astype(np.float64)
A = sp.csr_matrix((wv, (rr, cc)), shape=(nI, nI)); A = A.maximum(A.T).tocsr(); A = offdiag(A); A.eliminate_zeros()
text_dim = int(Tx.shape[1]); del Tx
du = np.asarray(X.sum(1)).ravel()
VAL = vad.groupby(uf)[itf].apply(lambda s: np.asarray(s, np.int64)).to_dict()
TST = ted.groupby(uf)[itf].apply(lambda s: np.asarray(s, np.int64)).to_dict()
users = np.array(sorted(u for u in TST if du[u] > 0), np.int64)
R = ('rp3b', 'iknn', 'combo')
acc = {r: dict(rv=[], rt=[], nd=[], rta=[]) for r in R}
for st in range(0, len(users), a.chunk):
    ids = users[st:st + a.chunk]
    Xb = X[ids]; mask = Xb.toarray() > 0
    Cs = {'rp3b': np.asarray((Xb @ W_rp).todense(), dtype=np.float64), 'iknn': np.asarray((Xb @ A).todense(), dtype=np.float64)}
    comb = np.where(mask, 0.0, zrow(Cs['rp3b'], mask)) + np.where(mask, 0.0, zrow(Cs['iknn'], mask))
    Cs['combo'] = comb
    for r in R:
        S = Cs[r].copy(); S[mask] = -np.inf
        T = np.argsort(-S, axis=1, kind='stable')[:, :K]
        for k, u in enumerate(ids.tolist()):
            tl = T[k]
            tt = TST[u]; ht = np.isin(tl, tt)
            acc[r]['rt'].append(ht.sum() / len(tt))
            acc[r]['nd'].append(float((ht / LOGD).sum()) / ICUM[min(len(tt), K)])
            v = S[k, tt]; pr = (S[k][None, :] > v[:, None]).sum(1) + 1; pt = (S[k][None, :] == v[:, None]).sum(1) - 1
            acc[r]['rta'].append(float(np.clip((K - pr + 1) / (pt + 1), 0, 1).mean()))
            if u in VAL:
                vv = VAL[u]; acc[r]['rv'].append(np.isin(tl, vv).sum() / len(vv))
out = dict(dataset=DS, code_sha256=sha(os.path.abspath(__file__)), feat=FEAT, feat_sha256=sha(FEAT), classic_json=CLS,
           n_users_test=int(len(users)), n_items=nI, text_dim=text_dim, knn=int(knn.shape[1]),
           rp3b_chosen=C1['rp3b']['chosen'], rp3b_w1_valid_r20=C1['rp3b']['valid_r20'], rp3b_w1_test_r20=C1['rp3b']['test_r20'],
           rankers={r: dict(valid_r20=float(np.mean(acc[r]['rv'])), test_r20=float(np.mean(acc[r]['rt'])),
                            test_n20=float(np.mean(acc[r]['nd'])), test_r20_tieadj=float(np.mean(acc[r]['rta']))) for r in R})
out['gate_rp3b_tieadj_vs_w1'] = abs(out['rankers']['rp3b']['test_r20_tieadj'] - C1['rp3b']['test_r20']) <= 5e-4
g = f'{S4R}/W1/classic/{DS}/rp3b__{DS}__s1.json'
if os.path.exists(g):
    out['rp3b_grid'] = json.load(open(g)).get('grid')
out['sec'] = round(time.time() - t0, 1)
json.dump(out, open(a.out, 'w'), indent=1)
print('FSTAND', DS, {r: round(v['test_r20'], 4) for r, v in out['rankers'].items()}, 'gate', out['gate_rp3b_tieadj_vs_w1'], out['sec'], flush=True)
