"""COLD step 1 (C re-judgment, cold = train-degree-0 test targets). CPU only, no training, no checkpoints loaded.
Sources: (a) dump npz (<RUN_ROOT>/dumps, W1/dumps, W2/dumps): per-positive tie-aware rank -> hitT@20; (b) A_DESIGN/results/{native,mes}__ds__sN.json deg0 group
(q2_fcombo, q2_ext fills; integer hits); (c) training-free methods recomputed here from canonical_v2 (train split only): rp3b, iknn_txt, combo,
content kNN text/image/text+image (L2-normalised mean-history profile), PCA-whitened variants (sensitivity only).
Metric: per dataset x seed, ratio = sum_t hitT@20 / sum_t (20 / n_elig(u_t)) over cold test targets. Rules: see PREREG_COLD.md.
usage: python -s -B c1.py <out_dir> [--smoke]   (smoke prints structural diagnostics only, no hit ratios, writes nothing)
"""
import os, sys, re, json, glob, time, hashlib, traceback
SMOKE = '--smoke' in sys.argv
OUT = sys.argv[1]
for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[v] = '16'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np, pandas as pd, scipy.sparse as sp

ROOT = '<RUN_ROOT>'
DR = '<RUN_ROOT_B>/P04_data/canonical_v2'
DSS = ['baby', 'pantry', 'scientific', 'sports', 'clothing', 'instruments', 'arts', 'office', 'microlens']
SEEDS = [1, 2, 3, 4, 5]
K = 20; THR = 1.5
LEARNED = ['lightgcn', 'native', 'mes', 'xre', 'xretn', 'freedom', 'lgmrec', 'mentor', 'smore', 'cohesion', 'pgl', 'diffmm']
DETDUMP = ['mmgf', 'mmgf_txt', 'mmgf_cf']
CODE_SHA = hashlib.sha256(open(os.path.abspath(__file__), 'rb').read()).hexdigest()
t0 = time.time()


def log(*a):
    print(f'[{time.time() - t0:7.1f}s]', *a, flush=True)


def hitT(pr, pt):
    return np.clip((K - pr + 1) / (pt + 1.0), 0, 1)


def load_ds(ds):
    d = f'{DR}/{ds}'
    df = pd.read_csv(f'{d}/{ds}.inter', sep='\t')
    uf = [c for c in df.columns if c.lower().startswith('user')][0]; itf = [c for c in df.columns if c.lower().startswith('item')][0]
    T = np.load(f'{d}/text_feat.npy', mmap_mode='r'); I = np.load(f'{d}/image_feat.npy', mmap_mode='r')
    nI = T.shape[0]; assert I.shape[0] == nI and df[itf].max() < nI, (ds, nI)
    nU = int(df[uf].max()) + 1
    tr = df[df.x_label == 0]; te = df[df.x_label == 2].drop_duplicates([uf, itf])
    X = sp.csr_matrix((np.ones(len(tr)), (tr[uf].values, tr[itf].values)), shape=(nU, nI)); X.sum_duplicates(); X.data[:] = 1.0; X.sort_indices()
    deg = np.asarray(X.sum(0)).ravel().astype(np.int64)
    tu, ti = te[uf].values.astype(np.int64), te[itf].values.astype(np.int64)
    cold = deg[ti] == 0
    return dict(ds=ds, X=X, deg=deg, nI=nI, nU=nU, tu=tu, ti=ti, cold=cold, T=T, I=I, dir=d)


def l2(M):
    M = np.array(M, np.float32); n = np.linalg.norm(M, axis=1, keepdims=True); n[n == 0] = 1; M = M / n
    return np.nan_to_num(M)


def offdiag(M):
    M = M.tocoo(); keep = M.row != M.col
    return sp.csr_matrix((M.data[keep], (M.row[keep], M.col[keep])), shape=M.shape)


def topk_rows(M, k):
    M = M.tocsr(); M.sort_indices(); rows, cols, vals = [], [], []
    for r in range(M.shape[0]):
        a0, a1 = M.indptr[r], M.indptr[r + 1]
        if a1 == a0: continue
        c = M.indices[a0:a1]; v = M.data[a0:a1]
        if len(v) > k:
            o = np.argsort(-v, kind='stable')[:k]; c, v = c[o], v[o]
        rows.append(np.full(len(c), r)); cols.append(c); vals.append(v)
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=M.shape)


def rp3b(X, alpha, beta, topK):
    du = np.asarray(X.sum(1)).ravel(); di = np.asarray(X.sum(0)).ravel()
    Pui = (sp.diags(1.0 / np.maximum(du, 1)) @ X).tocsr(); Piu = (sp.diags(1.0 / np.maximum(di, 1)) @ X.T.tocsr()).tocsr()
    Pui.data = Pui.data ** alpha; Piu.data = Piu.data ** alpha
    W = (Piu @ Pui).tocsr(); W = W @ sp.diags(1.0 / np.power(np.maximum(di, 1), beta))
    W = offdiag(W); W.eliminate_zeros()
    return topk_rows(W, topK)


def knn10(Tn):
    nI = Tn.shape[0]; idx = np.zeros((nI, 10), np.int64)
    for s in range(0, nI, 2000):
        S = Tn[s:s + 2000] @ Tn.T; S[np.arange(S.shape[0]), np.arange(s, s + S.shape[0])] = -np.inf
        p = np.argpartition(-S, 9, axis=1)[:, :10]; v = np.take_along_axis(S, p, 1); o = np.argsort(-v, axis=1, kind='stable')
        idx[s:s + 2000] = np.take_along_axis(p, o, 1)
    return idx


def zrow(S, mask):
    cnt = (~mask).sum(1, keepdims=True).astype(np.float64); S0 = np.where(mask, 0.0, S)
    mu = S0.sum(1, keepdims=True) / cnt
    sd = np.sqrt(np.where(mask, 0.0, (S - mu) ** 2).sum(1, keepdims=True) / cnt)
    return (S - mu) / np.maximum(sd, 1e-12)


def pcaw(F, nc=128):
    F = np.asarray(F, np.float32); mu = F.mean(0, keepdims=True); Fc = F - mu
    C = (Fc.T.astype(np.float64) @ Fc.astype(np.float64)) / max(len(F) - 1, 1)
    w, V = np.linalg.eigh(C); o = np.argsort(-w)[:min(nc, F.shape[1])]; w, V = w[o], V[:, o]
    Z = (Fc @ V.astype(np.float32)) / np.sqrt(np.maximum(w, 1e-8 * w.max()))[None, :].astype(np.float32)
    return l2(Z)


def cell(hits, rnd, n):
    return dict(n=int(n), hits=float(hits), rnd=float(rnd), ratio=(float(hits / rnd) if rnd > 0 else None))


def run_ds(D, classic_cfg):
    """training-free methods on cold test targets of dataset D -> {method: cell}"""
    X, nI = D['X'], D['nI']; c = D['cold']; tu, ti = D['tu'][c], D['ti'][c]
    out = {}; diag = {}
    if len(tu) == 0: return out, diag
    users = np.unique(tu); row = {u: k for k, u in enumerate(users.tolist())}
    Xb = X[users]; mask = Xb.toarray() > 0; nel = (~mask).sum(1); diag['n_cold'] = int(len(tu)); diag['n_users'] = int(len(users))
    Tn = l2(D['T']); In = l2(D['I']); TIn = l2(np.concatenate([Tn, In], 1))
    methods = {}
    W_rp = rp3b(X, **classic_cfg).tocsr(); Crp = np.asarray((Xb @ W_rp).todense(), np.float64); methods['rp3b'] = Crp
    knn = knn10(Tn); rr = np.repeat(np.arange(nI), 10); cc = knn.ravel()
    wv = np.maximum((Tn[rr] * Tn[cc]).sum(1), 0).astype(np.float64)
    A = sp.csr_matrix((wv, (rr, cc)), shape=(nI, nI)); A = A.maximum(A.T).tocsr(); A = offdiag(A); A.eliminate_zeros()
    Cik = np.asarray((Xb @ A).todense(), np.float64); methods['iknn_txt'] = Cik
    methods['combo_rp3b_iknn'] = np.where(mask, 0.0, zrow(Crp, mask)) + np.where(mask, 0.0, zrow(Cik, mask))
    dg = np.maximum(np.asarray(Xb.sum(1)).ravel(), 1)[:, None]
    for nm, F in (('cknn_txt', Tn), ('cknn_img', In), ('cknn_txtimg', TIn)):
        P = np.asarray(Xb @ F, np.float64) / dg; methods[nm] = P @ F.T.astype(np.float64)
    for nm, F in (('cknn_txt_pcaw', Tn), ('cknn_img_pcaw', In), ('cknn_txtimg_pcaw', TIn)):
        G = pcaw(F); P = np.asarray(Xb @ G, np.float64) / dg; methods[nm] = P @ G.T.astype(np.float64)
    diag['knn_overlap_vs_feat'] = None
    fp = f'{ROOT}/feat/{D["ds"]}_feat.npz'
    if os.path.exists(fp):
        fk = np.load(fp)['txt_knn']
        if fk.shape == knn.shape:
            diag['knn_overlap_vs_feat'] = float(np.mean([len(set(a.tolist()) & set(b.tolist())) / 10 for a, b in zip(knn[:2000], fk[:2000])]))
    for nm, S in methods.items():
        h = 0.0; r = 0.0
        for u, i in zip(tu.tolist(), ti.tolist()):
            k = row[u]; el = ~mask[k]; s = S[k, i]; v = S[k][el]
            larger = int((v > s).sum()); ties = int((v == s).sum()) - 1
            h += float(hitT(1 + larger, ties)); r += K / nel[k]
        out[nm] = cell(h, r, len(tu))
    diag['nel_mean'] = float(nel.mean())
    return out, diag


def main():
    os.makedirs(OUT, exist_ok=True)
    R = {'cells': {}, 'diag': {}, 'code_sha256': CODE_SHA}   # cells[method][ds][seed] -> cell

    def put(m, ds, s, c):
        R['cells'].setdefault(m, {}).setdefault(ds, {})[str(s)] = c
    dss = ['baby'] if SMOKE else DSS
    D_ = {}
    for ds in dss:
        D = load_ds(ds); D_[ds] = D
        R['diag'][ds] = dict(n_items=D['nI'], n_test=int(len(D['tu'])), n_cold=int(D['cold'].sum()), n_cold_users=int(len(np.unique(D['tu'][D['cold']]))))
        fp = f'{ROOT}/feat/{ds}_feat.npz'
        if os.path.exists(fp):
            R['diag'][ds]['deg_match_feat'] = bool(np.array_equal(np.load(fp)['item_deg'], D['deg']))
        log('loaded', ds, R['diag'][ds])
    # (c) training-free methods
    for ds in dss:
        cj = glob.glob(f'{ROOT}/W1/classic/{ds}/classic__{ds}.json')
        if cj: cfg = json.load(open(cj[0]))['methods']['rp3b']['chosen']
        else: cfg = dict(alpha=1.0, beta=0.5, topK=100)
        R['diag'][ds]['rp3b_cfg'] = cfg; R['diag'][ds]['rp3b_cfg_src'] = cj[0] if cj else 'DEFAULT(no classic json)'
        out, dg = run_ds(D_[ds], cfg); R['diag'][ds].update(dg)
        for m, c in out.items():
            for s in SEEDS: put(m, ds, s, c)      # deterministic: identical for every seed (split fixed by x_label)
        log('trainfree done', ds)
    # (a) dumps
    rg = re.compile(r'^([a-z0-9_]+?)__([a-z]+)__s(\d+)\.npz$')
    seen = {}; nbad = 0; ndump_chk = 0; nel_mis = 0
    for base in (f'{ROOT}/dumps', f'{ROOT}/W1/dumps', f'{ROOT}/W2/dumps'):
        for ds in dss:
            for p in sorted(glob.glob(f'{base}/{ds}/*.npz')):
                m = rg.match(os.path.basename(p))
                if not m or m.group(2) != ds: continue
                tag, s = m.group(1), int(m.group(3))
                if (tag, ds, s) in seen: continue
                try:
                    Z = np.load(p); pu, pi, pr, pt = Z['pu'], Z['pi'], Z['pr'], Z['pt']
                    D = D_[ds]; cm = D['deg'][pi] == 0
                    if int(cm.sum()) != int(D['cold'].sum()):
                        nbad += 1; R['diag'].setdefault('dump_cold_count_mismatch', []).append([p, int(cm.sum()), int(D['cold'].sum())]); continue
                    if not D['cold'].any():
                        seen[(tag, ds, s)] = p; continue
                    nel = dict(zip(Z['users'].tolist(), Z['n_elig'].tolist()))
                    rnd = sum(K / nel[u] for u in pu[cm].tolist())
                    put(tag, ds, s, cell(float(hitT(pr[cm], pt[cm]).sum()), rnd, cm.sum())); seen[(tag, ds, s)] = p
                except Exception as e:
                    nbad += 1; R['diag'].setdefault('dump_errors', []).append([p, repr(e)])
    R['diag']['n_dump_files_used'] = len(seen); R['diag']['n_dump_bad'] = nbad
    log('dumps done', len(seen))
    # (b) A_DESIGN fills (integer hits, top-20 lists)
    rg2 = re.compile(r'^(native|mes)__([a-z]+)__s(\d+)\.json$'); nf = 0
    for p in sorted(glob.glob(f'{ROOT}/A_DESIGN/results/*.json')):
        m = rg2.match(os.path.basename(p))
        if not m or not os.path.exists(p + '.DONE') or m.group(2) not in dss: continue
        tag, ds, s = m.group(1), m.group(2), int(m.group(3)); J = json.load(open(p))
        for rt in ('base', 'q2_fcombo', 'q2_ext'):
            g = J['routes'][rt]['groups']['deg0']
            if g['n']: put(f'adesign_{rt}@{tag}', ds, s, cell(g['hits'], g['n'] * g['rnd'], g['n']))
        nf += 1
    R['diag']['n_adesign_files'] = nf
    if SMOKE:
        print(json.dumps(R['diag'], indent=1)); return
    # ---- decision ----
    GROUPS = {
        'learned': LEARNED, 'trainfree_dump': DETDUMP,
        'fill': ['adesign_q2_fcombo@native', 'adesign_q2_ext@native', 'adesign_q2_fcombo@mes', 'adesign_q2_ext@mes'],
        'classic': ['rp3b', 'iknn_txt', 'combo_rp3b_iknn'],
        'content': ['cknn_txt', 'cknn_img', 'cknn_txtimg'],
    }
    INDEC = set(sum(GROUPS.values(), []))

    def judge(m, strict=False):
        per = {}
        for ds, sd in R['cells'].get(m, {}).items():
            cs = [c for c in sd.values() if c['n'] > 0 and c['rnd'] > 0]
            if not cs: continue
            hold = [(c['ratio'] > THR) and (c['hits'] >= (2 if strict else 1e-9)) for c in cs]
            per[ds] = dict(n=cs[0]['n'], n_seeds=len(cs), ratio_mean=float(np.mean([c['ratio'] for c in cs])), hits_mean=float(np.mean([c['hits'] for c in cs])),
                           rnd_mean=float(np.mean([c['rnd'] for c in cs])), seeds_hold=int(sum(hold)), hold=bool(np.mean(hold) >= 0.8),
                           collapse=bool(np.mean([not h for h in hold]) >= 0.8))
        nh = sum(v['hold'] for v in per.values()); nc = sum(v['collapse'] for v in per.values())
        return dict(per_ds=per, n_ds_hold=nh, n_ds_collapse=nc, S_holds=nh >= 3, S_collapses=nc >= 3)
    J = {}
    for strict in (False, True):
        k = 'strict' if strict else 'main'; J[k] = {}
        for g, ms in GROUPS.items():
            for m in ms:
                if m in R['cells']: J[k][m] = {**judge(m, strict), 'group': g}
        for m in R['cells']:
            if m not in INDEC: J[k][m] = {**judge(m, strict), 'group': 'sens/aux'}
    dec = {}
    for k in ('main', 'strict'):
        dm = {m: v for m, v in J[k].items() if v['group'] != 'sens/aux'}
        holders = [m for m, v in dm.items() if v['S_holds']]
        allc = all(v['S_collapses'] for v in dm.values())
        dec[k] = dict(holders=holders, n_S=len(dm), all_collapse=allc, not_collapsing=[m for m, v in dm.items() if not v['S_collapses']],
                      verdict=('C_REJECTED' if holders else ('C_CONFIRMED' if allc else 'INCONCLUSIVE')))
    R['judge'] = J; R['decision'] = dec; R['sec'] = round(time.time() - t0, 1)
    json.dump(R, open(f'{OUT}/c1.json', 'w'), indent=1)
    L = [f'# COLD C1 result (sha256 {CODE_SHA[:12]}..., {R["sec"]}s)', '', f'- decision (main rule): **{dec["main"]["verdict"]}**; holders = {dec["main"]["holders"]}',
         f'- decision (strict: hits>=2 per cell): **{dec["strict"]["verdict"]}**; holders = {dec["strict"]["holders"]}',
         '- cold targets per dataset: ' + ', '.join(f'{d}={R["diag"][d]["n_cold"]}' for d in dss), '',
         '| group | S | #ds hold | #ds collapse | per-dataset ratio_mean (hits_mean/rnd_mean, seeds>1.5 of n) |', '|---|---|---|---|---|']
    for m, v in J['main'].items():
        cells = '; '.join(f'{d}: {x["ratio_mean"]:.2f}x ({x["hits_mean"]:.2f}/{x["rnd_mean"]:.3f}, {x["seeds_hold"]}/{x["n_seeds"]})' for d, x in v['per_ds'].items())
        L.append(f'| {v["group"]} | {m} | {v["n_ds_hold"]} | {v["n_ds_collapse"]} | {cells} |')
    open(f'{OUT}/c1.md', 'w', encoding='utf-8').write('\n'.join(L))
    log('written', dec['main']['verdict'], dec['strict']['verdict'])


if __name__ == '__main__':
    try:
        main()
        if not SMOKE: open(os.path.join(os.path.dirname(OUT.rstrip('/')), 'C1_DONE'), 'w').write(str(round(time.time() - t0, 1)))
    except Exception:
        tb = traceback.format_exc(); print(tb)
        if not SMOKE: open(os.path.join(os.path.dirname(OUT.rstrip('/')), 'C1_FAIL'), 'w').write(tb)
        sys.exit(1)
