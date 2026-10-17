"""MESF(res) classical-score cache. CPU only. Computes s_C = Zs['combo'] of TABLE_FUSION/code/tf_fuse.py (RP3beta at W1's validation-chosen
config + ItemKNN-CBF(text), each z-scored per user over the eligible items, summed, z-scored again) from TRAIN interactions (x_label 0) only.
Three versions per dataset (dense float32 (n_users, n_items) .npy; z statistics over the items NOT in that version's visible history; every
item gets a value, history items included, so that training pairs can be scored):
  full : visible history = all training edges                      (used at inference, and by mesf_sc=full in training)
  f0/f1: visible history = training edges minus fold-f edges, fold = r12.fold_of(u, i) (the masked views of MES)  (mesf_sc=oof)
usage: python mesfres_sc.py <ds> <out_dir> [--versions full f0 f1] [--inter <file>] [--nu N --ni N] [--threads 8]
Writes only inside <out_dir>. The only interaction file read is <inter>; only rows with x_label == 0 are used (asserted)."""
import sys, os, json, time, hashlib, argparse
from pathlib import Path
ap = argparse.ArgumentParser()
ap.add_argument('ds'); ap.add_argument('out')
ap.add_argument('--versions', nargs='+', default=['full', 'f0', 'f1'])
ap.add_argument('--inter', default=None); ap.add_argument('--nu', type=int, default=None); ap.add_argument('--ni', type=int, default=None)
ap.add_argument('--threads', type=int, default=8); ap.add_argument('--chunk', type=int, default=512)
ap.add_argument('--tag', default='')
a = ap.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[v] = str(a.threads)
S4R = '<RUN_ROOT>'
WT = str(Path(__file__).resolve().parents[1] / 'wt')
CANON = '<RUN_ROOT_B>/P04_data/canonical_v2'
DS = a.ds
OUT = Path(a.out).resolve(); OUT.mkdir(parents=True, exist_ok=True)
CODE_SHA = hashlib.sha256(open(os.path.abspath(__file__), 'rb').read()).hexdigest()
os.chdir(WT); sys.path.insert(0, WT)
from core import registry
run_mod = registry.load_run_module('xres0'); registry.activate(run_mod.SRC_DIRS)
import numpy as np, scipy.sparse as sp, pandas as pd, torch
torch.set_num_threads(a.threads)
from models.r12 import fold_of
t0 = time.time()


def offdiag(M):
    M = M.tocoo(); keep = M.row != M.col
    return sp.csr_matrix((M.data[keep], (M.row[keep], M.col[keep])), shape=M.shape)


def topk_rows(M, k):                                     # copied from tf_fuse.py
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


def rp3b(X, alpha, beta, topK):                          # copied from tf_fuse.py
    du = np.asarray(X.sum(1)).ravel(); di = np.asarray(X.sum(0)).ravel()
    Pui = sp.diags(1.0 / np.maximum(du, 1)) @ X
    Piu = sp.diags(1.0 / np.maximum(di, 1)) @ X.T.tocsr()
    Pui = Pui.tocsr(); Piu = Piu.tocsr()
    Pui.data = Pui.data ** alpha; Piu.data = Piu.data ** alpha
    W = (Piu @ Pui).tocsr()
    W = W @ sp.diags(1.0 / np.power(np.maximum(di, 1), beta))
    W = offdiag(W); W.eliminate_zeros()
    return topk_rows(W, topK)


def zfull(S, mask):
    """per-row z with mean / population sd over the ELIGIBLE entries (mask True = visible history), applied to ALL entries"""
    el = ~mask
    cnt = np.maximum(el.sum(1, keepdims=True), 1).astype(np.float64)
    mu = (S * el).sum(1, keepdims=True) / cnt
    sd = np.sqrt((((S - mu) ** 2) * el).sum(1, keepdims=True) / cnt)
    return (S - mu) / np.maximum(sd, 1e-12)


inter = a.inter or f'{CANON}/{DS}/{DS}.inter'
df = pd.read_csv(inter, sep='\t')
uf = [c for c in df.columns if c.lower().startswith('user')][0]; itf = [c for c in df.columns if c.lower().startswith('item')][0]
F = dict(np.load(f'{S4R}/feat/{DS}_feat.npz')); deg = F['item_deg'].astype(np.int64)
nI = a.ni or len(deg); nU = a.nu or int(df[uf].max()) + 1
assert len(deg) == nI and df[itf].max() < nI and df[uf].max() < nU
trd = df[df.x_label == 0]
assert len(trd) > 0
X = sp.csr_matrix((np.ones(len(trd)), (trd[uf].values, trd[itf].values)), shape=(nU, nI)); X.sum_duplicates(); X.data[:] = 1.0; X.sort_indices()
assert np.array_equal(np.asarray(X.sum(0)).ravel().astype(np.int64), deg), 'train degree != feat item_deg'
chk = dict(n_train=int(X.nnz), n_val=int((df.x_label == 1).sum()), n_test=int((df.x_label == 2).sum()))
oth = df[df.x_label != 0]
if len(oth):
    Xo = sp.csr_matrix((np.ones(len(oth)), (oth[uf].values, oth[itf].values)), shape=(nU, nI)); Xo.data[:] = 1.0
    chk['val_test_pairs_also_in_train'] = int(X.multiply(Xo).nnz)
C1 = json.load(open(f'{S4R}/W1/classic/{DS}/classic__{DS}.json'))['methods']
RPCFG = C1['rp3b']['chosen']
Tx = np.load(f'{CANON}/{DS}/text_feat.npy').astype(np.float32); nr = np.linalg.norm(Tx, axis=1, keepdims=True); nr[nr == 0] = 1; Tx /= nr
knn = F['txt_knn'].astype(np.int64); rr = np.repeat(np.arange(nI), knn.shape[1]); cc = knn.ravel()
wv = np.maximum((Tx[rr] * Tx[cc]).sum(1), 0).astype(np.float64)
A = sp.csr_matrix((wv, (rr, cc)), shape=(nI, nI)); A = A.maximum(A.T).tocsr(); A = offdiag(A); A.eliminate_zeros()
del Tx
rows = np.repeat(np.arange(nU, dtype=np.int64), np.diff(X.indptr)); cols = X.indices.astype(np.int64)
ph = fold_of(torch.as_tensor(rows), torch.as_tensor(cols)).numpy()
meta = dict(ds=DS, nU=nU, nI=nI, code_sha256=CODE_SHA, inter=inter, rp3b_cfg=RPCFG, checks=chk, versions={})
for ver in a.versions:
    if ver == 'full':
        Xv = X
    else:
        f = int(ver[1]); ho = ph == f
        Xv = sp.csr_matrix((np.ones(int((~ho).sum())), (rows[~ho], cols[~ho])), shape=(nU, nI)); Xv.sort_indices()
        Rh = sp.csr_matrix((np.ones(int(ho.sum())), (rows[ho], cols[ho])), shape=(nU, nI))
        assert Xv.multiply(Rh).nnz == 0 and Xv.nnz + Rh.nnz == X.nnz, 'held-out edge present in the visible history'
    W = rp3b(Xv, **RPCFG).tocsr()
    path = OUT / f'{DS}_sc_{ver}{a.tag}.npy'
    mm = np.lib.format.open_memmap(str(path), mode='w+', dtype=np.float32, shape=(nU, nI))
    for st in range(0, nU, a.chunk):
        Xb = Xv[st:st + a.chunk]; mask = Xb.toarray() > 0
        z1 = zfull(np.asarray((Xb @ A).todense(), dtype=np.float64), mask)
        z2 = zfull(np.asarray((Xb @ W).todense(), dtype=np.float64), mask)
        mm[st:st + Xb.shape[0]] = zfull(z1 + z2, mask).astype(np.float32)
    mm.flush(); del mm
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 22), b''):
            h.update(b)
    meta['versions'][ver] = dict(path=str(path), nnz_visible=int(Xv.nnz), sha256=h.hexdigest(), sec=round(time.time() - t0, 1))
    print(DS, ver, 'done', meta['versions'][ver], flush=True)
(OUT / f'{DS}_sc_meta{a.tag}.json').write_text(json.dumps(meta, indent=1))
print('SC_DONE', DS, round(time.time() - t0, 1), flush=True)
