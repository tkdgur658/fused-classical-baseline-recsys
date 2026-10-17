"""CLASSIC E1b queue: CPU fusion (tf_fuse_ce.py) of existing checkpoints that already have a gated dump / TABLE_FUSION cell.
Cells = (a) every gated row of study/MES9/fusion_cells_9ds_mes.csv (the TABLE_FUSION + MES9 cells; 'source' = checkpoint run dir),
        (b) the comparator dumps <RUN_ROOT>/dumps/<ds>/<tag>__<ds>__s<seed>.json of arts / office / microlens (dump gate |recalc - json| <= 5e-4).
Cells without a dump are NOT created (this queue never trains).  Weight of every route family = argmax of validation R@20 over WS
(ties -> smaller w).  Waits for the marker CLASSIC/E1A_DONE before it starts.  Writes only under CLASSIC/E1B/ and CLASSIC/QUEUE_DONE.
usage: python tf_queue_ce.py [--workers 4] [--threads 2]
"""
import argparse, csv, json, math, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
S4R = '<RUN_ROOT>'
CL = f'{S4R}/study/CLASSIC'
PY = 'python'
LABEL = {'mes': 'MSCA+MES', 'native': 'MSCA', 'cohesion': 'COHESION', 'smore': 'SMORE', 'pgl': 'PGL', 'diffmm': 'DiffMM', 'lgmrec': 'LGMRec', 'freedom': 'FREEDOM'}
TAG = {v: k for k, v in LABEL.items()}
NEWDS = ('arts', 'office', 'microlens')
COMBO_DS = ('arts', 'office', 'microlens', 'baby', 'scientific')    # F recomputed here: datasets without an F, and two datasets for the reproduction check
ORDER = ['office', 'microlens', 'arts', 'clothing', 'sports', 'instruments', 'baby', 'pantry', 'scientific']   # AMENDMENT 4: big datasets first (load balance)
FAMS = ['ease', 'icf', 'mix', 'combo']
ap = argparse.ArgumentParser()
ap.add_argument('--workers', type=int, default=4); ap.add_argument('--threads', type=int, default=2)
ap.add_argument('--dry', action='store_true', help='print the job list summary and exit (no waiting, no run)')
a = ap.parse_args()
if not a.dry:
    os.makedirs(f'{CL}/E1B/cells', exist_ok=True); os.makedirs(f'{CL}/E1B/logs', exist_ok=True)
log = open(f'{CL}/E1B/queue.log' if not a.dry else '/dev/null', 'a')


def P(*x):
    s = time.strftime('%F %T ') + ' '.join(map(str, x)); print(s, flush=True); log.write(s + '\n'); log.flush()


while not a.dry and not os.path.exists(f'{CL}/E1A_DONE'):
    time.sleep(60)
P('E1A_DONE seen')
jobs, missing = [], []
for r in csv.DictReader(open(f'{S4R}/study/MES9/fusion_cells_9ds_mes.csv', newline='')):
    if r.get('gate_ok', '1') != '1' or not r['fused_r20']:
        missing.append((r['model'], r['dataset'], int(r['seed']), 'TABLE_FUSION gate'))
        continue
    jobs.append((TAG[r['model']], r['model'], r['dataset'], int(r['seed']), r['source']))
for ds in NEWDS:
    for tag in ('cohesion', 'diffmm', 'freedom', 'lgmrec', 'pgl', 'smore'):
        for sd in range(1, 6):
            p = f'{S4R}/dumps/{ds}/{tag}__{ds}__s{sd}.json'
            if not os.path.exists(p):
                missing.append((LABEL[tag], ds, sd, 'no dump')); continue
            d = json.load(open(p))
            if not (d.get('gate_ok') or abs(d['recalc_r20'] - d['json_r20']) <= 5e-4):
                missing.append((LABEL[tag], ds, sd, 'dump gate failed (recalc %.4f vs json %.4f)' % (d['recalc_r20'], d['json_r20']))); continue
            jobs.append((tag, LABEL[tag], ds, sd, d['run_dir']))
jobs.sort(key=lambda j: (ORDER.index(j[2]), j[0], j[3]))
P('jobs', len(jobs), 'missing at start', len(missing))
if a.dry:
    import collections
    print(collections.Counter((j[2]) for j in jobs)); print(collections.Counter((j[1]) for j in jobs)); print(missing[:12]); sys.exit(0)


def run(j):
    tag, model, ds, sd, src = j
    out = f'{CL}/E1B/cells/{tag}__{ds}__s{sd}.json'
    base = dict(model=model, dataset=ds, seed=sd, source=src)
    if not os.path.exists(out):
        cmd = ['nice', '-n', '15', PY, '-s', '-B', f'{CL}/code/tf_fuse_ce.py', src, ds, out, '--threads', str(a.threads)]
        if ds in COMBO_DS:
            cmd.append('--combo')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS=str(a.threads), MKL_NUM_THREADS=str(a.threads), OPENBLAS_NUM_THREADS=str(a.threads))
        with open(f'{CL}/E1B/logs/{tag}__{ds}__s{sd}.log', 'w') as lf:
            r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
        if r.returncode != 0 or not os.path.exists(out):
            return dict(base, gate_ok=0, error=1, note=f'fuse exit {r.returncode}')
    j2 = json.load(open(out)); g = j2['gate']
    ok = bool(g['tol_ok'] and g['tol_ok_n'])
    row = dict(base, base_r20=g['base_test_r20'], base_n20=g['base_test_n20'], gate_ok=int(ok),
               note='' if ok else f"gate fail: R tol {g['tol_ok']} N tol {g['tol_ok_n']} (base {g['base_test_r20']:.4f} vs ref {g['ref']})")
    for fam in FAMS:
        cand = [rt for rt in j2['routes'] if rt.startswith(f'f_{fam}_')]
        if not cand:
            continue
        sel = max(cand, key=lambda rt: (round(j2['routes'][rt]['valid_r20'], 6), -cand.index(rt)))   # ties -> smaller w
        rs = j2['routes'][sel]
        row[f'{fam}_r20'], row[f'{fam}_n20'], row[f'{fam}_w'] = rs['test_r20'], rs['test_n20'], float(sel.split('_')[2])
    return row


rows, errs = [], 0
with ThreadPoolExecutor(a.workers) as ex:
    for r in ex.map(run, jobs):
        rows.append(r)
        errs += int(bool(r.get('error')))
        P(r['model'], r['dataset'], r['seed'], 'gate_ok', r['gate_ok'], r.get('note', ''))
cols = ['model', 'dataset', 'seed', 'base_r20', 'base_n20'] + [f'{f}_{k}' for f in FAMS for k in ('r20', 'n20', 'w')] + ['gate_ok', 'note', 'source']
with open(f'{CL}/E1B/fusion_cells_ce.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, cols, extrasaction='ignore', lineterminator='\n')
    w.writeheader()
    for r in rows:
        w.writerow({c: (f'{r[c]:.6f}' if isinstance(r.get(c), float) and not c.endswith('_w') else r.get(c, '')) for c in cols})
with open(f'{CL}/E1B/missing_ce.tsv', 'w') as f:
    f.write('model\tdataset\tseed\treason\n')
    for m in missing:
        f.write('\t'.join(map(str, m)) + '\n')
with open(f'{CL}/E1B/all_routes.json', 'w') as f:      # every route of every gated cell (weight sweep), for the local generator
    allr = {}
    for r in rows:
        if r.get('error'):
            continue
        j2 = json.load(open(f"{CL}/E1B/cells/{TAG[r['model']]}__{r['dataset']}__s{r['seed']}.json"))
        allr[f"{r['model']}|{r['dataset']}|{r['seed']}"] = {rt: [v['valid_r20'], v['test_r20'], v['test_n20']] for rt, v in j2['routes'].items()}
    json.dump(allr, f)
open(f'{CL}/QUEUE_DONE', 'w').write(time.strftime('%F %T') + f' jobs {len(jobs)} errors {errs}\n')
P('QUEUE_DONE', len(jobs), 'errors', errs)
