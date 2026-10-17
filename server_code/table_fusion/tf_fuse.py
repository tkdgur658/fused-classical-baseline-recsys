"""TABLE_FUSION (2026-10-14) = w2_fuse020b.py + test NDCG@20 (MMRec convention, per user, mean over users with test targets) and
routes restricted to base and f_combo_<w> = z(L) + w z(combo), combo = z(rp3b) + z(iknn) (unchanged W2 recipe); icf route/gate dropped.
W2-020b (= w2_fuse020.py; the MSCA adjacency-cache pre-check applies to MSCA-family packages only - FREEDOM/LGMRec relay
runs keep their own caches in a per-run data root; any write outside the output directory is still refused by the guard)
W2-020 (= w2_fuse019.py with the test EvalDataLoader batch raised to >= 2048 users so that packages without a usable
cached forward recompute full_sort_predict fewer times; scores per user unchanged) -- W2-019 description follows.
W2-019 (= w2_fuse017.py + any MSCA-family package, cached forward tried for every package and checked against
full_sort_predict on the first batch, w grid + .05, validation-target groups 'vgroups'): the same post-hoc classical
score fusion applied to the NATIVE MSCA backbone and to model (a) MSCA+XRE (lambda .1 K40) - the fair post-hoc protocol.
Inference only, CPU, no training; the only file written is <out.json> (+ an empty tmp dir the loader needs next to it).
usage: python w2_fuse020.py <run_dir> <dataset> <feat.npz> <classic.json> <out.json> [--threads 8] [--chunk 512] [--smoke]
Loader = W1 w1_dump.py dump() adapted (package configs of result.json, RecDataset split, load_state_dict,
rebuild_scoring_state; one cached forward (ue, ie) checked against full_sort_predict on the first batch, else
full_sort_predict per batch). Users = test EvalDataLoader users (= the dump users). Training items masked; validation
and test items both candidates (MMRec convention: the same list serves the validation and the test evaluation).
Rankers (training-free, training split only):
  iknn  = ItemKNN-CBF(text) of W2-004/009/014 (w2_hybrid13.py): s(u,i) = sum_{j in H_u} A[j,i], A = kNN-10 graph of the
          feature file with cosine weights (positive part), symmetrised by max, no diagonal
  rp3b  = RP3beta operator of W1 w1_classic.py (copied, sha 1672385e...) at W1's validation-chosen config
  icf   = ItemKNN-CF operator of W1 w1_classic.py at W1's validation-chosen config
  combo = z(rp3b) + z(iknn)
Per user over eligible items: z(s) = (s - mean) / sd (population sd, clamp 1e-12). Routes: base = L (backbone score);
f_<r>_<w> = z(L) + w z(C_r), w in WS; alone_<r> = C_r. Top-20 of every route by torch.topk.
Metrics exactly as w2_hybrid13.py: user-level Recall@20 of validation targets (x_label 1) and of test targets (x_label 2)
from the route's list; test target groups all / deg_le2 / deg_le2_txtlink / deg0: n, hits, hit, rnd = mean 20/n_elig.
Gates: base test R@20 vs result.json best_test recall@20 (4 dp; tol 5e-4); rp3b / icf test R@20 from ranks (tie-adjusted,
= w1_classic.recall) vs W1 classic json test_r20 (5e-4); training mask of the loader == training rows of the .inter file.
"""
import sys, os, json, time, argparse, hashlib
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('run_dir'); ap.add_argument('dataset'); ap.add_argument('feat'); ap.add_argument('classic'); ap.add_argument('out')
ap.add_argument('--threads', type=int, default=8); ap.add_argument('--chunk', type=int, default=512)
ap.add_argument('--smoke', action='store_true', help='operational check only: first chunk, no metrics printed or written')
a = ap.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[v] = str(a.threads)
CODE_SHA = hashlib.sha256(open(os.path.abspath(__file__), 'rb').read()).hexdigest()
OUTF = str(Path(a.out).resolve()); OUTD = os.path.dirname(OUTF); os.makedirs(OUTD, exist_ok=True)
FEAT = str(Path(a.feat).resolve()); CLS = str(Path(a.classic).resolve())
WT_MAP = {'integx': '<HOME>/thesis_integn_wt'}
CANON_V2 = '<RUN_ROOT_B>/P04_data/canonical_v2'
WS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5]
K = 20
t0 = time.time()


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


RUN = Path(a.run_dir).resolve()
res = json.loads((RUN / 'result.json').read_text(encoding='utf-8'))
bc = res.get('best_ckpt')
if bc and Path(bc).is_file() and str(Path(bc).resolve()).startswith(str(RUN)):
    CK = str(Path(bc))
else:
    pts = sorted(RUN.glob('*/*.pt')); assert len(pts) == 1, pts; CK = str(pts[0])
PKG, CFGN, DS, SEED = res['package'], res['config'], res['dataset'], int(res['seed'])
assert DS == a.dataset, (DS, a.dataset)
WT = WT_MAP.get(PKG) or str(Path(res['config_files'][0]).resolve().parents[1])
DR = res['data_root'].rstrip('/')
if not Path(DR).exists():
    assert DR.endswith('canonical_v2'), DR
    DR = CANON_V2
os.chdir(WT); sys.path.insert(0, WT)
from core import registry
run_mod = registry.load_run_module(PKG); registry.activate(run_mod.SRC_DIRS)
import torch
torch.set_num_threads(a.threads)
from core.capture import cpu_torch_load_patch, param_checksum
cpu_torch_load_patch()
torch.Tensor.cuda = lambda self, *x, **k: (self.detach() if isinstance(self, torch.nn.Parameter) else self)
torch.nn.Module.cuda = lambda self, *x, **k: self
from utils.utils import apply_amp_sparse_patch
apply_amp_sparse_patch()
import numpy as np
import scipy.sparse as sp
import pandas as pd
LOGD = np.log2(np.arange(2, K + 2)); ICUM = np.concatenate([[1.0], np.cumsum(1.0 / LOGD)])   # ICUM[n] = ideal DCG of n targets (n<=K)


def _guard(orig, pos):
    def f(*x, **k):
        p = x[pos] if len(x) > pos else k.get('f', k.get('file'))
        if isinstance(p, (str, os.PathLike)) and not str(Path(p).resolve()).startswith(OUTD):
            raise RuntimeError(f'W2 guard: refusing to write {p}')
        return orig(*x, **k)
    return f


torch.save = _guard(torch.save, 1); np.save = _guard(np.save, 0); np.savez = _guard(np.savez, 0)
np.savez_compressed = _guard(np.savez_compressed, 0)
from core.runner import config_files, load_yaml, apply_env_adapter
from core.eval_state import rebuild_scoring_state
files = config_files(PKG, CFGN, DS)
assert [Path(f).name for f in files] == [Path(f).name for f in res['config_files']], (files, res['config_files'])
apply_env_adapter(run_mod, load_yaml(files[2]))
mcfg = load_yaml(files[2])
kk, sparse_ = mcfg.get('knn_k', 10), mcfg.get('is_sparse', True)
MSCA_PKGS = ('integx', 'objx_s', 'xres', 'xres0', 'xret_n', 'mes_c', 'mes_f', 'mes_m', 'mes_n', 'mes_s')
if PKG in MSCA_PKGS:      # MSCA-family adjacency caches; other packages: the save guard installed above refuses any write outside OUTD
    for m in ('image', 'text'):
        p = Path(DR) / DS / f'{m}_adj_{kk}_{sparse_}.pt'
        if not p.exists():
            raise RuntimeError(f'missing adjacency cache {p} (would be written to the data root)')
from utils.configurator import Config
from utils.dataset import RecDataset
from utils.dataloader import TrainDataLoader, EvalDataLoader
from utils.utils import init_seed, get_model
tmp = os.path.join(OUTD, f'_tmp_tf_{CFGN}_s{SEED}')
os.makedirs(tmp + '/saved', exist_ok=True); os.makedirs(tmp + '/topk', exist_ok=True)
ov = {'gpu_id': 0, 'seed': [SEED], 'data_path': DR + '/', 'state': 'silent', 'ex_output_dir': tmp,
      'checkpoint_dir': tmp + '/saved', 'recommend_topk': tmp + '/topk/', 'save_recommended_topk': False, 'use_gpu': False}
config = Config(run_mod.MODEL_CLASS, DS, ov, file_list=[str(f) for f in files])
init_seed(SEED)
dset = RecDataset(config); tr, va, te = dset.split()
train_data = TrainDataLoader(config, tr, batch_size=config['train_batch_size'], shuffle=True)
model = get_model(run_mod.MODEL_CLASS)(config, train_data).to(config['device'])
for n in [n for n in list(model._parameters) if n.startswith('result_embed')]:
    t = model._parameters.pop(n); model.__dict__[n] = t.detach()
state = torch.load(CK, map_location=config['device'])
missing, unexpected = model.load_state_dict(state, strict=False)
del state
pc = param_checksum(model)
model.eval()
prime = rebuild_scoring_state(model)
assert getattr(model, 'integ_ev', 'off') != 'on', 'backbone expected (no support channel)'
nU, nI = int(model.n_users), int(model.n_items)

# ---------------- training-free rankers (training split of the .inter file) ----------------
df = pd.read_csv(f'{DR}/{DS}/{DS}.inter', sep='\t')
uf = [c for c in df.columns if c.lower().startswith('user')][0]; itf = [c for c in df.columns if c.lower().startswith('item')][0]
assert df[uf].max() < nU and df[itf].max() < nI
trd, vad, ted = df[df.x_label == 0], df[df.x_label == 1], df[df.x_label == 2]
X = sp.csr_matrix((np.ones(len(trd)), (trd[uf].values, trd[itf].values)), shape=(nU, nI)); X.sum_duplicates(); X.data[:] = 1.0
X.sort_indices()
F = dict(np.load(FEAT)); deg = F['item_deg'].astype(np.int64)
assert len(deg) == nI and np.array_equal(np.asarray(X.sum(0)).ravel().astype(np.int64), deg)


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


C1 = json.load(open(CLS))['methods']
W_rp = rp3b(X, **C1['rp3b']['chosen']).tocsr()
Tx = np.load(f'{DR}/{DS}/text_feat.npy').astype(np.float32); nr = np.linalg.norm(Tx, axis=1, keepdims=True); nr[nr == 0] = 1; Tx /= nr
knn = F['txt_knn'].astype(np.int64); rr = np.repeat(np.arange(nI), knn.shape[1]); cc = knn.ravel()
wv = np.maximum((Tx[rr] * Tx[cc]).sum(1), 0).astype(np.float64)
A = sp.csr_matrix((wv, (rr, cc)), shape=(nI, nI)); A = A.maximum(A.T).tocsr(); A = offdiag(A); A.eliminate_zeros()
del Tx
OPS = {'iknn': A, 'rp3b': W_rp}
pk = F['pu'].astype(np.int64) * nI + F['pi'].astype(np.int64); o = np.argsort(pk); pk = pk[o]
tl = ((F['txt_in'] + F['txt_out']) > 0)[o]
VAL = vad.groupby(uf)[itf].apply(lambda s: np.asarray(s, np.int64)).to_dict()
TST = ted.groupby(uf)[itf].apply(lambda s: np.asarray(s, np.int64)).to_dict()
RANK = ['iknn', 'rp3b', 'icf', 'combo']
ROUTES = ['base'] + [f'f_combo_{w}' for w in WS]
GR = ('all', 'deg_le2', 'deg_le2_txtlink', 'deg0')
GV = ('all', 'deg_le2', 'deg0')
acc = {rt: dict(rv=[], rt=[], nd=[], g={g: [0, 0, 0.0] for g in GR}, vg={g: [0, 0, 0.0] for g in GV}) for rt in ROUTES}
cls_rank = {r: [[], [], []] for r in ('rp3b',)}       # tie-adjusted rank gate: pu, pr, pt of test positives


def zrow(S, mask):
    """per-row z over eligible entries (mask True = training item); masked entries -> -inf"""
    cnt = (~mask).sum(1, keepdims=True).astype(np.float64)
    S0 = np.where(mask, 0.0, S)
    mu = S0.sum(1, keepdims=True) / cnt
    sd = np.sqrt(np.where(mask, 0.0, (S - mu) ** 2).sum(1, keepdims=True) / cnt)
    Z = (S - mu) / np.maximum(sd, 1e-12)
    Z[mask] = -np.inf
    return Z


EVAL_BS = max(int(config['eval_batch_size']), 2048)
data = EvalDataLoader(config, te, additional_dataset=tr, batch_size=EVAL_BS)
cache = None
if not getattr(model, 'integ_dl', False) and getattr(model, 'integ_p02', 'off') != 'fused':
    try:
        with torch.no_grad():
            _fw = model.forward()
        _ue, _ie = _fw[0], _fw[1]
        if _ue.dim() == 2 and _ie.dim() == 2 and _ue.shape[0] == nU and _ie.shape[0] == nI and _ue.shape[1] == _ie.shape[1]:
            cache = (_ue, _ie.transpose(0, 1))
    except Exception as e:                       # forward needs arguments / returns another structure -> per-batch path
        cache = None
cache_check = None
nusers = 0
with torch.no_grad():
    for bd in data:
        users_t = bd[0]
        if cache is not None:
            s_plain = torch.matmul(cache[0][users_t], cache[1])
            if cache_check is None:
                chk = model.full_sort_predict(bd)
                dmax = float((chk.double() - s_plain.double()).abs().max())
                cache_check = dict(bitwise=bool(torch.equal(chk, s_plain)), max_abs_diff=dmax)
                if dmax > 1e-5 * (float(chk.double().abs().max()) + 1e-12):
                    cache = None; s_plain = chk; cache_check['fallback'] = True
        else:
            s_plain = model.full_sort_predict(bd)
        Lb = s_plain.double().cpu().numpy()
        m = bd[1]
        ub = users_t.cpu().numpy().astype(np.int64)
        nm = np.zeros(Lb.shape, dtype=bool); nm[m[0].cpu().numpy(), m[1].cpu().numpy()] = True
        for st in range(0, len(ub), a.chunk):
            ids = ub[st:st + a.chunk]; B = len(ids)
            Xb = X[ids]
            mask = Xb.toarray() > 0
            if not np.array_equal(mask, nm[st:st + B]):
                raise RuntimeError('loader train mask != .inter training rows')
            nel = (~mask).sum(1)
            L = Lb[st:st + B].copy(); L[mask] = -np.inf
            zL = zrow(np.where(mask, 0.0, L), mask)
            Cs = {r: np.asarray((Xb @ OPS[r]).todense(), dtype=np.float64) for r in ('iknn', 'rp3b')}
            Zs = {r: zrow(Cs[r], mask) for r in Cs}
            comb = np.where(mask, 0.0, Zs['rp3b']) + np.where(mask, 0.0, Zs['iknn'])
            Zs['combo'] = zrow(comb, mask); Cs['combo'] = comb
            # tie-adjusted rank gate of the classical rankers (test positives)
            for r in ('rp3b',):
                Sc = Cs[r].copy(); Sc[mask] = -np.inf
                for k, u in enumerate(ids.tolist()):
                    if u in TST:
                        tt = TST[u]; v = Sc[k, tt]
                        cls_rank[r][0].append(np.full(len(tt), u)); cls_rank[r][1].append((Sc[k][None, :] > v[:, None]).sum(1) + 1)
                        cls_rank[r][2].append((Sc[k][None, :] == v[:, None]).sum(1) - 1)
            # pairs of the chunk
            vr_, vi_, tr_, ti_ = [], [], [], []
            for k, u in enumerate(ids.tolist()):
                if u in VAL:
                    vv = VAL[u]; vr_.append(np.full(len(vv), k)); vi_.append(vv)
                if u in TST:
                    tt = TST[u]; tr_.append(np.full(len(tt), k)); ti_.append(tt)
            vr_ = np.concatenate(vr_) if vr_ else np.zeros(0, np.int64); vi_ = np.concatenate(vi_) if vi_ else np.zeros(0, np.int64)
            tr_ = np.concatenate(tr_) if tr_ else np.zeros(0, np.int64); ti_ = np.concatenate(ti_) if ti_ else np.zeros(0, np.int64)
            key = ids[tr_] * nI + ti_; pos = np.searchsorted(pk, key)
            tlk = np.zeros(len(key), bool); okk = (pos < len(pk)); okk[okk] = pk[pos[okk]] == key[okk]; tlk[okk] = tl[pos[okk]]
            dg = deg[ti_]
            gmask = {'all': np.ones(len(ti_), bool), 'deg_le2': dg <= 2, 'deg_le2_txtlink': (dg <= 2) & tlk, 'deg0': dg == 0}
            rnd = 20.0 / nel[tr_]
            dv = deg[vi_]; vmask = {'all': np.ones(len(vi_), bool), 'deg_le2': dv <= 2, 'deg0': dv == 0}; vrnd = 20.0 / nel[vr_]
            for rt in ROUTES:
                if rt == 'base':
                    S = L
                elif rt.startswith('alone_'):
                    S = Cs[rt[6:]].copy(); S[mask] = -np.inf
                else:
                    _, r, w = rt.split('_'); S = zL + float(w) * Zs[r]
                T = torch.topk(torch.from_numpy(S), K, dim=1).indices.numpy()
                A_ = acc[rt]
                if len(vr_):
                    hv = (T[vr_] == vi_[:, None]).any(1)
                    nv = np.bincount(vr_, minlength=B); hs = np.bincount(vr_, weights=hv, minlength=B)
                    A_['rv'].extend((hs[nv > 0] / nv[nv > 0]).tolist())
                    for g, gm in vmask.items():
                        A_['vg'][g][0] += int(gm.sum()); A_['vg'][g][1] += int(hv[gm].sum()); A_['vg'][g][2] += float(vrnd[gm].sum())
                if len(tr_):
                    ht = (T[tr_] == ti_[:, None]).any(1)
                    nt = np.bincount(tr_, minlength=B); hs = np.bincount(tr_, weights=ht, minlength=B)
                    A_['rt'].extend((hs[nt > 0] / nt[nt > 0]).tolist())
                    gain = ((T[tr_] == ti_[:, None]) / LOGD[None, :]).sum(1)
                    dcg = np.bincount(tr_, weights=gain, minlength=B)
                    idcg = ICUM[np.minimum(nt, K)]
                    A_['nd'].extend((dcg[nt > 0] / idcg[nt > 0]).tolist())
                    for g, gm in gmask.items():
                        A_['g'][g][0] += int(gm.sum()); A_['g'][g][1] += int(ht[gm].sum()); A_['g'][g][2] += float(rnd[gm].sum())
            nusers += B
            if a.smoke:
                print('SMOKE OK', DS, SEED, 'chunk', B, 'nI', nI, 'cache', cache_check, 'mask_eq True', 'routes', len(ROUTES), f'{time.time() - t0:.1f}s', flush=True)
                sys.exit(0)
out = dict(dataset=DS, seed=SEED, run_dir=str(RUN), ckpt=CK, ckpt_sha256=sha256_file(CK), cfg=CFGN, code_sha256=CODE_SHA,
           params_match=(pc['sha256'] == res.get('params', {}).get('sha256')), missing=list(missing), unexpected=list(unexpected),
           prime=str(prime), cache_check=cache_check, eval_bs=EVAL_BS, n_users=nusers, ws=WS, classic_cfg={r: C1[r]['chosen'] for r in ('rp3b',)},
           routes={})
for rt in ROUTES:
    A_ = acc[rt]
    out['routes'][rt] = dict(valid_r20=float(np.mean(A_['rv'])), test_r20=float(np.mean(A_['rt'])), test_n20=float(np.mean(A_['nd'])),
                             groups={g: dict(n=v[0], hits=v[1], hit=(v[1] / v[0] if v[0] else None), rnd=(v[2] / v[0] if v[0] else None))
                                     for g, v in A_['g'].items()},
                             vgroups={g: dict(n=v[0], hits=v[1], hit=(v[1] / v[0] if v[0] else None), rnd=(v[2] / v[0] if v[0] else None))
                                      for g, v in A_['vg'].items()})
ref = res.get('best_test', {}).get('recall@20')
b = out['routes']['base']['test_r20']
refn = res.get('best_test', {}).get('ndcg@20'); bn = out['routes']['base']['test_n20']
out['gate'] = dict(base_test_r20=b, ref=ref, base_test_n20=bn, ref_n20=refn, gate_ok=bool(ref is not None and round(b, 4) == round(ref, 4)),
                   tol_ok=bool(ref is not None and abs(b - ref) <= 5e-4), tol_ok_n=bool(refn is not None and abs(bn - refn) <= 5e-4))
cg = {}
for r, w1k in (('rp3b', 'rp3b'),):
    pu = np.concatenate(cls_rank[r][0]); pr = np.concatenate(cls_rank[r][1]); pt = np.concatenate(cls_rank[r][2])
    h = np.clip((K - pr + 1) / (pt + 1), 0, 1); us, inv = np.unique(pu, return_inverse=True)
    mine = float((np.bincount(inv, weights=h) / np.bincount(inv)).mean())
    cg[r] = dict(mine=mine, w1=C1[w1k]['test_r20'], ok=bool(abs(mine - C1[w1k]['test_r20']) <= 5e-4))
out['classic_gate'] = cg
out['sec'] = round(time.time() - t0, 1)
json.dump(out, open(OUTF, 'w'), indent=1)
print('TFUSE', PKG, CFGN, DS, SEED, 'eval_bs', EVAL_BS, 'base', round(b, 4), 'ref', ref, 'gate', out['gate']['gate_ok'], out['gate']['tol_ok'],
      'classic', {r: (round(v['mine'], 4), v['ok']) for r, v in cg.items()}, 'cache', cache_check, 'sec', out['sec'], flush=True)
