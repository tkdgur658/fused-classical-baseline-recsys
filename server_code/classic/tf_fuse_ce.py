"""CLASSIC E1b (2026-10-16): TABLE_FUSION recipe (tf_fuse.py, sha 2399e9d0...) with the classical part replaced by the E1a operators of
study/CLASSIC/<dataset>/ (rp3b, CBF, ItemKNN-CF, EASE, validation-chosen z-mix).  Inference only, CPU, no training, no GPU.
Per cell (one checkpoint = one model, dataset, seed) the routes are
    base            = L (backbone score)
    f_<r>_<w>       = z(L) + w z(C_r),  w in WS = {.05,.1,.2,.3,.5,.7,1,1.5},  r in
                      ease  = EASE (lambda chosen on validation in E1a),
                      icf   = ItemKNN-CF (W1 validation-chosen config),
                      mix   = z-mix of (rp3b, iknn, icf, ease) with the E1a validation-chosen integer weights,
                      combo = z(rp3b) + z(iknn) = the fused baseline F of TABLE_FUSION (only with --combo, reproduction / datasets without F)
z = per user over the eligible items, population sd (clamp 1e-12).  The weight of every route family is chosen per cell (= per seed) by
validation Recall@20 (ties -> smaller w) in the queue script tf_queue_ce.py.  Top-20 with ties -> lower item index (classic_common.top20).
Gates (as TABLE_FUSION): base test R@20 and N@20 within 5e-4 of result.json best_test; loader training mask == .inter training rows.
usage: python -s -B tf_fuse_ce.py <run_dir> <dataset> <out.json> [--threads 2] [--chunk 512] [--combo] [--classic_root DIR] [--smoke]
"""
import sys, os, json, time, argparse, hashlib
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('run_dir'); ap.add_argument('dataset'); ap.add_argument('out')
ap.add_argument('--threads', type=int, default=2); ap.add_argument('--chunk', type=int, default=512)
ap.add_argument('--combo', action='store_true'); ap.add_argument('--classic_root', default=None)
ap.add_argument('--smoke', action='store_true', help='operational check only: first chunk, no metrics printed or written')
a = ap.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[v] = str(a.threads)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
CODE_SHA = hashlib.sha256(open(os.path.abspath(__file__), 'rb').read()).hexdigest()
OUTF = str(Path(a.out).resolve()); OUTD = os.path.dirname(OUTF); os.makedirs(OUTD, exist_ok=True)
WT_MAP = {'integx': '<HOME>/thesis_integn_wt'}
CANON_V2 = '<RUN_ROOT_B>/P04_data/canonical_v2'
K = 20
t0 = time.time()

RUN = Path(a.run_dir).resolve()
res = json.loads((RUN / 'result.json').read_text(encoding='utf-8'))
bc = res.get('best_ckpt')
if bc and Path(bc).is_file() and str(Path(bc).resolve()).startswith(str(RUN)):
    CK = str(Path(bc))
else:
    pts = sorted(RUN.glob('*/*.pt')); assert len(pts) == 1, pts; CK = str(pts[0])
PKG, CFGN, DS, SEED = res['package'], res['config'], res['dataset'], int(res['seed'])
DS = DS.split('@')[0]
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
import classic_common as C
CROOT = a.classic_root or C.CLASSIC


def _guard(orig, pos):
    def f(*x, **k):
        p = x[pos] if len(x) > pos else k.get('f', k.get('file'))
        if isinstance(p, (str, os.PathLike)) and not str(Path(p).resolve()).startswith(OUTD):
            raise RuntimeError(f'CLASSIC guard: refusing to write {p}')
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
if PKG in MSCA_PKGS:
    for m in ('image', 'text'):
        p = Path(DR) / DS / f'{m}_adj_{kk}_{sparse_}.pt'
        if not p.exists():
            raise RuntimeError(f'missing adjacency cache {p} (would be written to the data root)')
from utils.configurator import Config
from utils.dataset import RecDataset
from utils.dataloader import TrainDataLoader, EvalDataLoader
from utils.utils import init_seed, get_model
tmp = os.path.join(OUTD, f'_tmp_tfce_{CFGN}_s{SEED}')
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

# ---------------- training-free rankers: E1a operators of this dataset ----------------
df = pd.read_csv(f'{DR}/{DS}/{DS}.inter', sep='\t')
uf = [c for c in df.columns if c.lower().startswith('user')][0]; itf = [c for c in df.columns if c.lower().startswith('item')][0]
assert df[uf].max() < nU and df[itf].max() < nI
trd, vad, ted = df[df.x_label == 0], df[df.x_label == 1], df[df.x_label == 2]
X = sp.csr_matrix((np.ones(len(trd)), (trd[uf].values, trd[itf].values)), shape=(nU, nI)); X.sum_duplicates(); X.data[:] = 1.0
X.sort_indices()
X32 = X.astype(np.float32).tocsr()
CD = f'{CROOT}/{DS}'
CJ = json.load(open(f'{CD}/classic_{DS}.json'))
assert CJ['n_items'] == nI and CJ['n_train'] == int(X.nnz), (CJ['n_items'], nI, CJ['n_train'], X.nnz)
OPS = {'rp3b': sp.load_npz(f'{CD}/ops_rp3b.npz').tocsr(), 'iknn': sp.load_npz(f'{CD}/ops_cbf.npz').tocsr(), 'icf': sp.load_npz(f'{CD}/ops_icf.npz').tocsr()}
BE = np.load(f'{CD}/ease_B.npy')
assert BE.shape == (nI, nI) and BE.dtype == np.float32
MIXW = CJ['mix_w']
DD = dict(VAL=vad.groupby(uf)[itf].apply(lambda s: np.asarray(s, np.int64)).to_dict(),
          TST=ted.groupby(uf)[itf].apply(lambda s: np.asarray(s, np.int64)).to_dict())
FAM = ['ease', 'icf', 'mix'] + (['combo'] if a.combo else [])
ROUTES = ['base'] + [f'f_{r}_{w}' for r in FAM for w in C.WS]
acc = {rt: C.new_acc() for rt in ROUTES}

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
    except Exception as e:
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
            L = Lb[st:st + B].copy(); L[mask] = -np.inf
            zL = C.zrow(np.where(mask, 0.0, L), mask)
            Cs = {r: np.asarray((Xb @ OPS[r]).todense(), dtype=np.float64) for r in ('rp3b', 'iknn', 'icf')}
            Cs['ease'] = np.asarray(X32[ids] @ BE, dtype=np.float64)
            Z0 = {r: np.where(mask, 0.0, C.zrow(Cs[r], mask)) for r in C.RANKERS}
            ZR = {'ease': C.zrow(Cs['ease'], mask), 'icf': C.zrow(Cs['icf'], mask)}
            ZR['mix'] = C.zrow(sum(wr * Z0[r] for wr, r in zip(MIXW, C.RANKERS) if wr), mask)
            if a.combo:
                ZR['combo'] = C.zrow(Z0['rp3b'] + Z0['iknn'], mask)
            tg = C.chunk_targets(ids, DD)
            for rt in ROUTES:
                if rt == 'base':
                    S = L
                else:
                    _, r, w = rt.split('_'); S = zL + float(w) * ZR[r]
                T = C.top20(S)
                C.add_val(acc[rt], T, tg, B); C.add_test(acc[rt], T, tg, B)
            nusers += B
            if a.smoke:
                print('SMOKE OK', DS, SEED, 'chunk', B, 'nI', nI, 'cache', cache_check, 'mask_eq True', 'routes', len(ROUTES), f'{time.time() - t0:.1f}s', flush=True)
                sys.exit(0)
out = dict(dataset=DS, seed=SEED, run_dir=str(RUN), ckpt=CK, ckpt_sha256=C.sha256_file(CK), cfg=CFGN, code_sha256=CODE_SHA,
           common_sha256=C.sha256_file(C.__file__), classic_json=f'{CD}/classic_{DS}.json', classic_json_sha256=C.sha256_file(f'{CD}/classic_{DS}.json'),
           params_match=(pc['sha256'] == res.get('params', {}).get('sha256')), missing=list(missing), unexpected=list(unexpected),
           prime=str(prime), cache_check=cache_check, eval_bs=EVAL_BS, n_users=nusers, ws=C.WS, families=FAM, mix_w=MIXW, routes={})
for rt in ROUTES:
    out['routes'][rt] = C.summarize(acc[rt])
    out['routes'][rt].pop('test_r20_tieadj', None)
ref = res.get('best_test', {}).get('recall@20')
b = out['routes']['base']['test_r20']
refn = res.get('best_test', {}).get('ndcg@20'); bn = out['routes']['base']['test_n20']
out['gate'] = dict(base_test_r20=b, ref=ref, base_test_n20=bn, ref_n20=refn, gate_ok=bool(ref is not None and round(b, 4) == round(ref, 4)),
                   tol_ok=bool(ref is not None and abs(b - ref) <= 5e-4), tol_ok_n=bool(refn is not None and abs(bn - refn) <= 5e-4))
out['sec'] = round(time.time() - t0, 1)
json.dump(out, open(OUTF, 'w'), indent=1)
print('TFUSE_CE', PKG, CFGN, DS, SEED, 'eval_bs', EVAL_BS, 'base', round(b, 4), 'ref', ref, 'gate', out['gate']['gate_ok'], out['gate']['tol_ok'],
      'cache', cache_check, 'sec', out['sec'], flush=True)
