"""REFUSE (S4 step 4): re-run the TABLE_FUSION procedure (code/tf_fuse.py UNCHANGED, same classic config, same validation weight rule:
argmax validation R@20 over f_combo_<w>, ties -> smaller w) on exactly the Table-II checkpoints of MSCA (table1_runs.tsv chosen Native
runs) and MSCA+MES (mes_perseed.csv row MES run_dir). Input study/tableII_runs.csv (model,dataset,seed,run_dir,test_r20,test_n20).
A cell whose checkpoint file is absent on HOST_A is NOT run (reported as CKPT_ABSENT). Check: |base_r20 - Table-II test_r20| <= 1e-4.
CPU only, nice 15; writes only under REFUSE/."""
import csv, json, math, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
S4R = '<RUN_ROOT>'
TF = f'{S4R}/TABLE_FUSION'
RF = f'{S4R}/study/REFUSE'
PY = 'python'
WORK, THREADS = 3, 2
os.makedirs(f'{RF}/cells', exist_ok=True); os.makedirs(f'{RF}/logs', exist_ok=True)
rows = list(csv.DictReader(open(f'{S4R}/study/tableII_runs.csv')))
log = open(f'{RF}/queue.log', 'a')
def P(*x):
    s = time.strftime('%F %T ') + ' '.join(map(str, x)); print(s, flush=True); log.write(s + '\n'); log.flush()
TAG = {'MSCA': 'native', 'MSCA+MES': 'mes'}

def ckpt_of(d):
    res = json.load(open(d + '/result.json'))
    bc = res.get('best_ckpt')
    if bc and os.path.isfile(bc) and os.path.realpath(bc).startswith(os.path.realpath(d)):
        return bc
    import glob
    pts = glob.glob(d + '/*/*.pt')
    return pts[0] if len(pts) == 1 else None

def run(r):
    m, ds, sd, d = r['model'], r['dataset'], int(r['seed']), r['run_dir']
    base = dict(model=m, dataset=ds, seed=sd, source=d, t2_r20=float(r['test_r20']))
    if not os.path.isfile(d + '/result.json') or ckpt_of(d) is None:
        return dict(base, gate_ok=0, absent=1, note='CKPT_ABSENT')
    out = f'{RF}/cells/{TAG[m]}__{ds}__s{sd}.json'
    if not os.path.exists(out):
        cmd = ['nice', '-n', '15', PY, '-s', '-B', f'{TF}/code/tf_fuse.py', d, ds, f'{S4R}/feat/{ds}_feat.npz',
               f'{S4R}/W1/classic/{ds}/classic__{ds}.json', out, '--threads', str(THREADS)]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS=str(THREADS), MKL_NUM_THREADS=str(THREADS), OPENBLAS_NUM_THREADS=str(THREADS))
        with open(f'{RF}/logs/{TAG[m]}__{ds}__s{sd}.log', 'w') as lf:
            rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env).returncode
        if rc != 0 or not os.path.exists(out):
            return dict(base, gate_ok=0, error=1, note=f'fuse exit {rc}')
    j2 = json.load(open(out)); g = j2['gate']
    cg_ok = all(v['ok'] for v in j2['classic_gate'].values())
    ok = bool(g['tol_ok'] and g['tol_ok_n'] and cg_ok)
    cand = [x for x in j2['routes'] if x.startswith('f_combo_')]
    sel = max(cand, key=lambda x: (round(j2['routes'][x]['valid_r20'], 6), -cand.index(x)))
    rs = j2['routes'][sel]
    d_t2 = abs(g['base_test_r20'] - base['t2_r20'])
    return dict(base, base_r20=g['base_test_r20'], base_n20=g['base_test_n20'], fused_r20=rs['test_r20'], fused_n20=rs['test_n20'],
                weight=float(sel.split('_')[2]), gate_ok=int(ok), d_t2=d_t2, t2_match=int(d_t2 <= 1e-4),
                cache=str(j2['cache_check']), note='' if ok else 'gate fail')

if __name__ == '__main__':
    res = []
    with ThreadPoolExecutor(WORK) as ex:
        for r in ex.map(run, rows):
            res.append(r); P(r['model'], r['dataset'], r['seed'], r.get('note', ''), r.get('base_r20', ''), r.get('t2_r20'), r.get('d_t2', ''), r.get('fused_r20', ''), r.get('weight', ''))
    cols = ['model', 'dataset', 'seed', 'base_r20', 'base_n20', 'fused_r20', 'fused_n20', 'weight', 'gate_ok', 'source']
    ds_order = ['baby', 'pantry', 'scientific', 'sports']; mo = ['MSCA+MES', 'MSCA']
    res.sort(key=lambda r: (mo.index(r['model']), ds_order.index(r['dataset']), r['seed']))
    with open(f'{RF}/fusion_cells_s4.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, cols, extrasaction='ignore', lineterminator='\n'); w.writeheader()
        for r in res:
            row = {c: (f'{r[c]:.6f}' if isinstance(r.get(c), float) and c != 'weight' else r.get(c, '')) for c in cols}
            if r.get('absent'): row['source'] = 'CKPT_ABSENT:' + r['source']
            w.writerow(row)
    with open(f'{RF}/refuse_check.csv', 'w', newline='') as f:
        c2 = ['model', 'dataset', 'seed', 'status', 't2_r20', 'base_r20', 'abs_diff', 'match_1e-4', 'weight', 'fused_r20']
        w = csv.writer(f, lineterminator='\n'); w.writerow(c2)
        for r in res:
            w.writerow([r['model'], r['dataset'], r['seed'], 'ABSENT' if r.get('absent') else ('OK' if r.get('gate_ok') else 'FAIL'), r['t2_r20'],
                        r.get('base_r20', ''), r.get('d_t2', ''), r.get('t2_match', ''), r.get('weight', ''), r.get('fused_r20', '')])
    open(f'{RF}/REFUSE_DONE', 'w').write(time.strftime('%F %T'))
    P('finished')
