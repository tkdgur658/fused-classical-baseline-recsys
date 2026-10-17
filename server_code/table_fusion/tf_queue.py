"""TABLE_FUSION queue: CPU re-fusion (W2 recipe, tf_fuse.py) of existing checkpoints that have a gated S4R dump.
usage: python tf_queue.py --datasets baby,pantry,scientific,sports --suffix '' [--workers 4] [--threads 2]
Cells: dumps/<ds>/<tag>__<ds>__s<1-5>.json (tag in native, mes, cohesion, smore, pgl, diffmm, lgmrec, freedom). A cell is used only if
(1) the dump json gate holds (|recalc_r20 - json_r20| <= 5e-4) and (2) tf_fuse.py recomputes base test R@20 = result.json best_test R@20
(tol 5e-4), base N@20 likewise, and the rp3b classic gate holds. Weight: argmax of validation R@20 over f_combo_<w> (ties -> smaller w)."""
import argparse, csv, json, math, os, subprocess, time
from concurrent.futures import ThreadPoolExecutor
S4R = '<RUN_ROOT>'
TF = f'{S4R}/TABLE_FUSION'
PY = 'python'
LABEL = {'mes': 'MSCA+MES', 'native': 'MSCA', 'cohesion': 'COHESION', 'smore': 'SMORE', 'pgl': 'PGL', 'diffmm': 'DiffMM',
         'lgmrec': 'LGMRec', 'freedom': 'FREEDOM'}
ap = argparse.ArgumentParser()
ap.add_argument('--datasets', required=True)
ap.add_argument('--suffix', default='')
ap.add_argument('--workers', type=int, default=4)
ap.add_argument('--threads', type=int, default=2)
a = ap.parse_args()
DS = a.datasets.split(',')
SUF = a.suffix
os.makedirs(f'{TF}/cells', exist_ok=True)
os.makedirs(f'{TF}/logs', exist_ok=True)
log = open(f'{TF}/queue{SUF}.log', 'a')


def P(*x):
    s = time.strftime('%F %T ') + ' '.join(map(str, x))
    print(s, flush=True)
    log.write(s + '\n')
    log.flush()


grid = {}   # (tag, ds, seed) -> state of the GRID job
try:
    for ln in open(f'{S4R}/GRID/jobs.tsv'):
        if ln.startswith('#') or not ln.strip():
            continue
        f = ln.rstrip('\n').split('\t')
        jid, kind, tag, ds, sd = f[0], f[2], f[3], f[4], f[5]
        st = 'DONE' if os.path.exists(f'{S4R}/GRID/markers/{jid}.DONE') else (
            'FAIL' if os.path.exists(f'{S4R}/GRID/markers/{jid}.FAIL') else 'pending')
        grid[(tag, ds, sd)] = f'{jid} {kind} {st}'
except Exception as e:
    P('jobs.tsv read failed', e)

jobs, missing, cellrows = [], [], []
for tag in LABEL:
    for ds in DS:
        for sd in range(1, 6):
            p = f'{S4R}/dumps/{ds}/{tag}__{ds}__s{sd}.json'
            if not os.path.exists(p):
                missing.append((LABEL[tag], ds, sd, 'no dump; GRID: ' + grid.get((tag, ds, str(sd)), 'no GRID job')))
                continue
            if not (os.path.exists(f'{S4R}/feat/{ds}_feat.npz') and os.path.exists(f'{S4R}/W1/classic/{ds}/classic__{ds}.json')):
                missing.append((LABEL[tag], ds, sd, 'dump exists but feat npz or W1 classic json of the dataset is missing'))
                continue
            jobs.append((tag, ds, sd, p))
P('datasets', DS, 'jobs', len(jobs), 'missing', len(missing))


def run(j):
    tag, ds, sd, p = j
    d = json.load(open(p))
    out = f'{TF}/cells/{tag}__{ds}__s{sd}.json'
    base = dict(model=LABEL[tag], dataset=ds, seed=sd, source=d['run_dir'])
    if not (d.get('gate_ok') or abs(d['recalc_r20'] - d['json_r20']) <= 5e-4):
        return dict(base, gate_ok=0, note='dump gate failed (recalc %.4f vs json %.4f)' % (d['recalc_r20'], d['json_r20']))
    if not os.path.exists(out):
        cmd = ['nice', '-n', '15', PY, '-s', '-B', f'{TF}/code/tf_fuse.py', d['run_dir'], ds, f'{S4R}/feat/{ds}_feat.npz',
               f'{S4R}/W1/classic/{ds}/classic__{ds}.json', out, '--threads', str(a.threads)]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS=str(a.threads), MKL_NUM_THREADS=str(a.threads))
        with open(f'{TF}/logs/{tag}__{ds}__s{sd}.log', 'w') as lf:
            r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
        if r.returncode != 0 or not os.path.exists(out):
            return dict(base, gate_ok=0, error=1, note=f'fuse exit {r.returncode}')
    j2 = json.load(open(out))
    g = j2['gate']
    cg_ok = all(v['ok'] for v in j2['classic_gate'].values())
    ok = bool(g['tol_ok'] and g['tol_ok_n'] and cg_ok)
    cand = [r for r in j2['routes'] if r.startswith('f_combo_')]
    sel = max(cand, key=lambda r: (round(j2['routes'][r]['valid_r20'], 6), -cand.index(r)))
    rs = j2['routes'][sel]
    return dict(base, base_r20=g['base_test_r20'], base_n20=g['base_test_n20'], fused_r20=rs['test_r20'], fused_n20=rs['test_n20'],
                weight=float(sel.split('_')[2]), gate_ok=int(ok),
                note='' if ok else f"gate fail: R tol {g['tol_ok']} N tol {g['tol_ok_n']} classic {cg_ok} (base {g['base_test_r20']:.4f} vs ref {g['ref']})")


errs = 0
with ThreadPoolExecutor(a.workers) as ex:
    for r in ex.map(run, jobs):
        cellrows.append(r)
        if r.get('error'):
            errs += 1
        P(r['model'], r['dataset'], r['seed'], 'gate_ok', r['gate_ok'], r.get('fused_r20', ''), r.get('note', ''))

cols = ['model', 'dataset', 'seed', 'base_r20', 'base_n20', 'fused_r20', 'fused_n20', 'weight', 'gate_ok', 'source']
order = list(LABEL.values())
cellrows.sort(key=lambda r: (order.index(r['model']), DS.index(r['dataset']), r['seed']))
with open(f'{TF}/fusion_cells{SUF}.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, cols, extrasaction='ignore', lineterminator='\n')
    w.writeheader()
    for r in cellrows:
        w.writerow({c: (f'{r[c]:.6f}' if isinstance(r.get(c), float) and c != 'weight' else r.get(c, '')) for c in cols})


def ms(v):
    m = sum(v) / len(v)
    s = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else float('nan')
    return m, s


with open(f'{TF}/fusion_summary{SUF}.csv', 'w', newline='') as f:
    w = csv.writer(f, lineterminator='\n')
    w.writerow(['model', 'dataset', 'n', 'seeds', 'r20_mean', 'r20_std', 'n20_mean', 'n20_std', 'base_r20_mean', 'base_n20_mean'])
    for m in order:
        for ds in DS:
            rs = [r for r in cellrows if r['model'] == m and r['dataset'] == ds and r['gate_ok']]
            if not rs:
                continue
            fr, fn = ms([r['fused_r20'] for r in rs]), ms([r['fused_n20'] for r in rs])
            w.writerow([m, ds, len(rs), '/'.join(str(r['seed']) for r in rs), f'{fr[0]:.6f}', f'{fr[1]:.6f}', f'{fn[0]:.6f}',
                        f'{fn[1]:.6f}', f"{sum(r['base_r20'] for r in rs) / len(rs):.6f}",
                        f"{sum(r['base_n20'] for r in rs) / len(rs):.6f}"])
with open(f'{TF}/MISSING{SUF}.md', 'w') as f:
    f.write(f'# TABLE_FUSION MISSING / EXCLUDED cells (datasets: {",".join(DS)}; generated {time.strftime("%F %T")})\n\n')
    f.write('## No dump (not fused; this queue never trains)\n| model | dataset | seed | reason |\n|---|---|---|---|\n')
    for m, ds, sd, why in missing:
        f.write(f'| {m} | {ds} | {sd} | {why} |\n')
    f.write('\n## Excluded by the gate (base test R@20/N@20 did not reproduce result.json) or failed\n'
            '| model | dataset | seed | note |\n|---|---|---|---|\n')
    for r in cellrows:
        if not r['gate_ok']:
            f.write(f"| {r['model']} | {r['dataset']} | {r['seed']} | {r.get('note', '')} |\n")
open(f'{TF}/TABLE_FUSION{SUF.upper()}_{"FAIL" if errs else "DONE"}', 'w').write(time.strftime('%F %T') + f' cells {len(cellrows)} errs {errs}\n')
P('finished', 'errs', errs)
