"""Collect the CACHEFIX build-correction re-runs of Baby (PREREG_CACHEFIX, study; author decision) and simulate their effect on
tab:main / tab:sota. Only reads remote files; writes two per-seed files and a simulation report; never edits tables/*.csv, *.tex or any
generator (the simulation runs the unchanged generators on a temporary copy of tables/).

Usage (local machine, any directory; Python with scipy, e.g. python, 3.10+):
  python tables/src/collect_cachefix.py                 collect after markers/QUEUE_DONE and simulate
  python tables/src/collect_cachefix.py --no-sim        collect only
  python tables/src/collect_cachefix.py --selftest shift|identity   offline check of the simulation machinery (no server, temp files only)
The script sends itself over ssh to HOST_A (argument `scan`), reads study/CACHEFIX/{markers,runs,summary}, and writes
  tables/src/data/cachefix_perseed.csv       block 1 (tab:main MSCA DA-MRS / UltraGCN-LI, Baby): the columns of tables/table1_perseed.csv
                                              (backbone,row,dataset,seed,r20,n20,r10,n10,gpu,root,fingerprint; root = cachefix)
  tables/src/data/cachefix_sota_perseed.csv  block 2 (tab:sota COHESION Baby): the columns of tables/table2_runs.tsv (the per-seed source
                                              of build_table2_main.py), chosen = 1
  tables/src/data/cachefix_sim_report.md     which printed means, bold marks and triangle marks change (tab:main, tab:sota)
A cell is written only when all five seeds are DONE with lane gate PASS and the expected fingerprint (otherwise it is reported as incomplete
and the previous values stay; they are replaced, never pooled, per the prereg). Replacement rule of the simulation: for each replaced
(row, seed) the new run is the only value of that seed (table1_perseed.csv / table2_runs.tsv rows, plus the mean / SD rows of
table1_cells.csv / table2_cells.csv recomputed with the formulas of collect_table1.py / collect_table2.py).
"""
import csv
import datetime as dt
import json
import os
import shutil
import statistics as st
import subprocess
import sys
import tempfile
from pathlib import Path

G = '<RUN_ROOT>/study/CACHEFIX'
SEEDS = [1, 2, 3, 4, 5]
T1_ROWS = {'plg2_s/baby_damrs_a1b2_k10_x2': 'DA-MRS', 'plg2_s/baby_ugcn_li_k10_l0007': 'UltraGCN-LI'}
T2_KEY = 'cohesion/COHESION_A5_baby'


# ----------------------------------------------------------------------------------------------- remote side
def scan():
    import re
    out = {'queue': {}, 'jobs': []}
    for n in ('QUEUE_DONE', 'QUEUE_FAIL', 'LAUNCHED'):
        p = f'{G}/markers/{n}'
        out['queue'][n] = open(p).read().strip() if os.path.exists(p) else None
    for n in ('summary/setup_gate.json', 'summary/postcheck.json'):
        p = f'{G}/{n}'
        out[n.split('/')[1]] = json.load(open(p)) if os.path.exists(p) else None
    for l in open(f'{G}/jobs_cachefix.tsv').read().splitlines():
        if not l.strip() or l.startswith('#'):
            continue
        c = l.split('\t')
        j = dict(id=c[0], group=c[1], pkg=c[2], cfg=c[3], ds=c[4], seed=int(c[5]), copy=c[6], expected_fp=c[9], marker='NONE', gate='', result=None)
        for s in ('DONE', 'FAIL', 'SKIP_CAP', 'RUNNING'):
            p = f'{G}/markers/{j["group"]}/{j["id"]}.{s}'
            if os.path.exists(p):
                j['marker'] = s
                txt = open(p).read()
                m = re.search(r'gate=(\S+)', txt)
                j['gate'] = m.group(1) if m else ''
                break
        rj = sorted(__import__('glob').glob(f'{G}/runs/{j["id"]}/*/*/result.json'))
        if rj:
            d = json.load(open(rj[-1]))
            bt, bv = d.get('best_test') or {}, d.get('best_valid_primed') or {}
            j['result'] = dict(status=d.get('status'), mode=d.get('mode'), test_eval=d.get('test_eval'), data_root=d.get('data_root'),
                               gpu=(d.get('gpu') or '').replace('NVIDIA ', ''), started=d.get('started'), elapsed_sec=d.get('elapsed_sec'),
                               best_epoch=d.get('best_epoch'), fingerprint=d.get('fingerprint'), r10=bt.get('recall@10'), n10=bt.get('ndcg@10'),
                               r20=bt.get('recall@20'), n20=bt.get('ndcg@20'), vr20=bv.get('recall@20'), vn20=bv.get('ndcg@20'), path=rj[-1])
        out['jobs'].append(j)
    print(json.dumps(out))


# ------------------------------------------------------------------------------------------------ local side
HERE = Path(__file__).resolve().parent
TABLES = HERE.parent
DATA = HERE / 'data'
SSH = ['ssh', '-o', 'BatchMode=yes', 'HOST_A']
T1_COLS = ['backbone', 'row', 'dataset', 'seed', 'r20', 'n20', 'r10', 'n10', 'gpu', 'root', 'fingerprint']


def remote():
    with open(os.path.abspath(__file__), 'rb') as fh:
        out = subprocess.run(SSH + ['python3', '-', 'scan'], stdin=fh, capture_output=True, check=True)
    if out.stderr.strip():
        print(out.stderr.decode(errors='replace'), file=sys.stderr)
    return json.loads(out.stdout.decode())


def f6(x):
    return '' if x is None else f'{x:.6f}'


def completed(started, elapsed):
    try:
        return (dt.datetime.strptime(started, '%Y-%m-%d %H:%M:%S') + dt.timedelta(seconds=float(elapsed or 0))).strftime('%Y-%m-%d %H:%M:%S')
    except (TypeError, ValueError):
        return ''


def build(meta):
    """-> (new1 {(row, seed): dict}, new2 {seed: dict}, status lines). Only cells with all five seeds DONE, gate PASS, expected fingerprint."""
    lines = []
    cells = {}
    for j in meta['jobs']:
        cells.setdefault((j['pkg'] + '/' + j['cfg']), []).append(j)
    new1, new2 = {}, {}
    for key, js in cells.items():
        ok = True
        for j in sorted(js, key=lambda x: x['seed']):
            r = j['result']
            why = []
            if j['marker'] != 'DONE':
                why.append(f"marker {j['marker']}")
            if j['gate'] != 'PASS':
                why.append(f"gate {j['gate'] or '-'}")
            if not r or r['fingerprint'] != j['expected_fp']:
                why.append('fingerprint')
            if r and (r['status'] != 'SUCCESS' or r['test_eval'] != 'best_primed' or r['mode'] != 'train'):
                why.append('status/mode/test_eval')
            if why:
                ok = False
                lines.append(f"  {j['id']}: not usable ({', '.join(why)})")
        if sorted(j['seed'] for j in js) != SEEDS or not ok:
            lines.append(f'cell {key}: INCOMPLETE, not written (previous values stay)')
            continue
        for j in js:
            r = j['result']
            if key in T1_ROWS:
                new1[(T1_ROWS[key], j['seed'])] = dict(backbone='MSCA', row=T1_ROWS[key], dataset='baby', seed=j['seed'], r20=r['r20'], n20=r['n20'],
                                                     r10=r['r10'], n10=r['n10'], gpu=r['gpu'], root='cachefix', fingerprint=r['fingerprint'])
            elif key == T2_KEY:
                new2[j['seed']] = dict(model='COHESION', dataset='baby', seed=j['seed'], chosen='1', ineligible_reason='', rank='0',
                                       duplicate_note='CACHEFIX build-correction re-run (replaces the previous run of this seed)', host='HOST_A',
                                       queue='cachefix', config=j['cfg'], status=r['status'], mode=r['mode'], test_eval=r['test_eval'], data='canonical',
                                       superseded='0', completed=completed(r['started'], r['elapsed_sec']), best_epoch=r['best_epoch'], test_r10=r['r10'],
                                       test_n10=r['n10'], test_r20=r['r20'], test_n20=r['n20'], gpu=r['gpu'], origin=r['path'])
        lines.append(f'cell {key}: complete (5 seeds, gate PASS)')
    return new1, new2, lines


def write_new(new1, new2, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    p1, p2 = outdir / 'cachefix_perseed.csv', outdir / 'cachefix_sota_perseed.csv'
    with open(p1, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=T1_COLS, lineterminator='\n')
        w.writeheader()
        for k in sorted(new1):
            w.writerow(new1[k])
    cols = ['model', 'dataset', 'seed', 'chosen', 'ineligible_reason', 'rank', 'duplicate_note', 'host', 'queue', 'config', 'status', 'mode', 'test_eval',
            'data', 'superseded', 'completed', 'best_epoch', 'test_r10', 'test_n10', 'test_r20', 'test_n20', 'gpu', 'origin']
    with open(p2, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator='\n')
        w.writeheader()
        for k in sorted(new2):
            w.writerow(new2[k])
    return p1, p2, cols


# ------------------------------------------------------------------------------------------------ simulation
def rd(path, delim=','):
    with open(path, newline='', encoding='utf-8') as f:
        r = csv.DictReader(f, delimiter=delim)
        return list(r), r.fieldnames


def wr(path, rows, cols, delim=','):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter=delim, lineterminator='\n', extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def apply_new(tables, new1, new2):
    """replace the old per-seed values and cell rows in a temporary copy of tables/"""
    if new1:
        rows, cols = rd(tables / 'table1_perseed.csv')
        done = set()
        for r in rows:
            k = (r['row'], int(r['seed']))
            if r['backbone'] == 'MSCA' and r['dataset'] == 'baby' and k in new1:
                n = new1[k]
                r.update({c: str(n[c]) for c in ('r20', 'n20', 'r10', 'n10', 'gpu', 'root', 'fingerprint')})
                done.add(k)
        assert done == set(new1), ('table1_perseed rows not found', set(new1) - done)
        wr(tables / 'table1_perseed.csv', rows, cols)
        cells, ccols = rd(tables / 'table1_cells.csv')
        for row in sorted({k[0] for k in new1}):
            sv = [new1[(row, s)] for s in SEEDS]
            c = next(c for c in cells if (c['backbone'], c['row'], c['dataset']) == ('MSCA', row, 'baby'))
            r20, n20 = [float(x['r20']) for x in sv], [float(x['n20']) for x in sv]
            c.update(n='5', r20_mean=f6(st.mean(r20)), r20_std=f6(st.stdev(r20)), n20_mean=f6(st.mean(n20)), n20_std=f6(st.stdev(n20)),
                     seeds='1/2/3/4/5', gpus='/'.join(sorted({x['gpu'] for x in sv})), roots='/'.join(x['root'] for x in sv))
        wr(tables / 'table1_cells.csv', cells, ccols)
    if new2:
        rows, cols = rd(tables / 'table2_runs.tsv', '\t')
        for r in rows:
            if r['model'] == 'COHESION' and r['dataset'] == 'baby' and r['chosen'] == '1' and int(r['seed']) in new2:
                r['chosen'] = '0'
                r['ineligible_reason'] = 'superseded by CACHEFIX build-correction re-run'
        rows += [{c: str(new2[s].get(c, '')) for c in cols} for s in SEEDS if s in new2]
        wr(tables / 'table2_runs.tsv', rows, cols, '\t')
        cells, ccols = rd(tables / 'table2_cells.csv')
        c = next(c for c in cells if (c['model'], c['dataset']) == ('COHESION', 'baby'))
        sv = [new2[s] for s in SEEDS]
        col = lambda k: [float(x[k]) for x in sv]
        c.update(n='5', seeds='1/2/3/4/5', missing_seeds='', r20_mean=f6(st.mean(col('test_r20'))), n20_mean=f6(st.mean(col('test_n20'))),
                 r20_std=f6(st.stdev(col('test_r20'))), n20_std=f6(st.stdev(col('test_n20'))), r10_mean=f6(st.mean(col('test_r10'))),
                 n10_mean=f6(st.mean(col('test_n10'))), queues='cachefix/cachefix/cachefix/cachefix/cachefix')
        wr(tables / 'table2_cells.csv', cells, ccols)


def run_builders(tables, py):
    msg = []
    for b in ('build_table1_main.py', 'build_table2_main.py'):
        r = subprocess.run([py, b], cwd=tables / 'src', capture_output=True, text=True, env=dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONDONTWRITEBYTECODE='1'))
        if r.returncode != 0:
            raise SystemExit(f'{b} failed in {tables}:\n{r.stderr[-1500:]}')
        msg.append(r.stdout.strip().splitlines()[-1][:160] if r.stdout.strip() else b)
    return msg


def sig_map(path, keycols):
    rows, _ = rd(path)
    return {tuple(r[c] for c in keycols): (int(r['reject']), ('+' if float(r['diff']) > 0 else '-') if int(r['reject']) else '', float(r['p_holm']), float(r['diff'])) for r in rows}


def diff_report(base, new, label):
    out = []
    # ---- tab:main
    bm, _ = rd(base / 'table1_marks.csv')
    nm, _ = rd(new / 'table1_marks.csv')
    bk = {(r['backbone'], r['row'], r['dataset'], r['metric']): r for r in bm}
    chg_val, chg_best = [], []
    for r in nm:
        k = (r['backbone'], r['row'], r['dataset'], r['metric'])
        o = bk[k]
        if (o['printed'], o['std']) != (r['printed'], r['std']):
            chg_val.append(f"{k[0]}/{k[1]}/{k[2]}/{k[3]}: {o['printed']} ({o['std']}) -> {r['printed']} ({r['std']})")
        if o['mark'] != r['mark']:
            chg_best.append(f"{k[0]}/{k[1]}/{k[2]}/{k[3]}: best mark {o['mark'] or '-'} -> {r['mark'] or '-'}")
    s_b = sig_map(base / 'sigtests_table1.csv', ['contrast', 'dataset', 'metric'])
    s_n = sig_map(new / 'sigtests_table1.csv', ['contrast', 'dataset', 'metric'])
    chg_sig = [f"{k[0]} | {k[1]} | {k[2]}: mark {'^' + s_b[k][1] if s_b[k][0] else 'none'} -> {'^' + s_n[k][1] if s_n[k][0] else 'none'} "
               f"(p_holm {s_b[k][2]:.4g} -> {s_n[k][2]:.4g})" for k in s_n if s_b[k][:2] != s_n[k][:2]]
    near = [f"{k[0]} | {k[1]} | {k[2]}: p_holm {s_b[k][2]:.4g} -> {s_n[k][2]:.4g}, diff {s_b[k][3]:+.4f} -> {s_n[k][3]:+.4f}" for k in s_n
            if k[0].startswith('MSCA: ') and any(x in k[0] for x in ('DA-MRS', 'UltraGCN')) and s_b[k][:3] != s_n[k][:3] and s_b[k][:2] == s_n[k][:2]]
    out += [f'### tab:main ({label})', f'- printed mean / SD cells that change: {len(chg_val)}'] + [f'  - {x}' for x in chg_val]
    out += [f'- bold (best) marks that change: {len(chg_best)}'] + [f'  - {x}' for x in chg_best]
    out += [f'- significance marks (triangles) that change: {len(chg_sig)}'] + [f'  - {x}' for x in chg_sig]
    out += [f'- unchanged marks whose Holm-adjusted p moves (MSCA DA-MRS / UltraGCN-LI rows, informational): {len(near)}'] + [f'  - {x}' for x in near]
    # ---- tab:sota
    bm, _ = rd(base / 'table2_marks.csv')
    nm, _ = rd(new / 'table2_marks.csv')
    bk = {(r['row'], r['dataset'], r['metric']): r for r in bm}
    chg_val, chg_best, chg_sig = [], [], []
    for r in nm:
        k = (r['row'], r['dataset'], r['metric'])
        o = bk[k]
        if o['printed'] != r['printed']:
            chg_val.append(f"{k[0]}/{k[1]}/{k[2]}: {o['printed']} -> {r['printed']}")
        if o['mark'] != r['mark']:
            chg_best.append(f"{k[0]}/{k[1]}/{k[2]}: rank mark {o['mark'] or '-'} -> {r['mark'] or '-'}")
        if o['sig_vs_ours'] != r['sig_vs_ours']:
            chg_sig.append(f"{k[0]}/{k[1]}/{k[2]}: vs MSCA + MES {o['sig_vs_ours'] or 'n.s.'} -> {r['sig_vs_ours'] or 'n.s.'}")
    s_b = sig_map(base / 'sigtests_table2.csv', ['contrast', 'dataset', 'metric'])
    s_n = sig_map(new / 'sigtests_table2.csv', ['contrast', 'dataset', 'metric'])
    near = [f"{k[0]} | {k[1]} | {k[2]}: p_holm {s_b[k][2]:.4g} -> {s_n[k][2]:.4g}, diff {s_b[k][3]:+.4f} -> {s_n[k][3]:+.4f}" for k in s_n
            if k[0].startswith('COHESION') and s_b[k][:3] != s_n[k][:3]]
    out += [f'### tab:sota ({label})', f'- printed mean / SD cells that change: {len(chg_val)}'] + [f'  - {x}' for x in chg_val]
    out += [f'- best-mean (bold) marks that change: {len(chg_best)}'] + [f'  - {x}' for x in chg_best]
    out += [f'- significance marks vs MSCA + MES that change: {len(chg_sig)}'] + [f'  - {x}' for x in chg_sig]
    out += [f'- COHESION tests whose Holm-adjusted p or sign moves (informational): {len(near)}'] + [f'  - {x}' for x in near]
    return out


def simulate(new1, new2, py, label):
    tmp = Path(tempfile.mkdtemp(prefix='cachefix_sim_'))
    ig = shutil.ignore_patterns('__pycache__', '_old')
    base, new = tmp / 'base' / 'tables', tmp / 'new' / 'tables'
    shutil.copytree(TABLES, base, ignore=ig)
    shutil.copytree(TABLES, new, ignore=ig)
    apply_new(new, new1, new2)
    run_builders(base, py)
    run_builders(new, py)
    note = []
    for f in ('table1_marks.csv', 'sigtests_table1.csv', 'table2_marks.csv', 'sigtests_table2.csv'):
        same = (base / f).read_bytes() == (TABLES / f).read_bytes()
        note.append(f'baseline rebuild of {f} {"identical to" if same else "DIFFERS from"} the manuscript copy')
    rep = [f'# CACHEFIX simulation ({label}), {dt.datetime.now():%F %T}', '', f'temporary copy: {tmp} (deleted by nothing; remove by hand)', ''] + [f'- {x}' for x in note] + ['']
    rep += diff_report(base, new, label)
    return rep, tmp


def main():
    args = sys.argv[1:]
    py = sys.executable
    if '--selftest' in args:
        mode = args[args.index('--selftest') + 1]
        r1, _ = rd(TABLES / 'table1_perseed.csv')
        new1 = {}
        for r in r1:
            if r['backbone'] == 'MSCA' and r['dataset'] == 'baby' and r['row'] in ('DA-MRS', 'UltraGCN-LI'):
                d = 0.0 if mode == 'identity' else (0.004 if r['row'] == 'DA-MRS' else -0.004)
                new1[(r['row'], int(r['seed']))] = dict(backbone='MSCA', row=r['row'], dataset='baby', seed=int(r['seed']), r20=f"{float(r['r20']) + d:.4f}",
                                                      n20=f"{float(r['n20']) + d / 2:.4f}", r10=r['r10'], n10=r['n10'], gpu=r['gpu'], root='cachefix', fingerprint='selftest')
        r2, _ = rd(TABLES / 'table2_runs.tsv', '\t')
        new2 = {}
        for r in r2:
            if r['model'] == 'COHESION' and r['dataset'] == 'baby' and r['chosen'] == '1':
                d = 0.0 if mode == 'identity' else -0.006
                x = dict(r)
                x.update(test_r20=f"{float(r['test_r20']) + d:.4f}", test_n20=f"{float(r['test_n20']) + d / 2:.4f}", queue='cachefix', data='canonical', chosen='1')
                new2[int(r['seed'])] = x
        rep, tmp = simulate(new1, new2, py, f'SELFTEST {mode}: synthetic values, not results')
        print('\n'.join(rep))
        return
    meta = remote()
    print('queue markers:', json.dumps(meta['queue']))
    print('setup gate PASS:', bool((meta.get('setup_gate.json') or {}).get('PASS')), '| postcheck PASS:', (meta.get('postcheck.json') or {}).get('PASS'))
    new1, new2, lines = build(meta)
    print('\n'.join(lines))
    if not new1 and not new2:
        raise SystemExit('nothing complete yet; no file written')
    p1, p2, _ = write_new(new1, new2, DATA)
    print('wrote', p1, len(new1), 'rows |', p2, len(new2), 'rows')
    if '--no-sim' in args:
        return
    rep, tmp = simulate(new1, new2, py, 'real CACHEFIX values')
    rep = ['## collection status'] + [f'- {x}' for x in lines] + [''] + rep
    (DATA / 'cachefix_sim_report.md').write_text('\n'.join(rep) + '\n', encoding='utf-8')
    print('\n'.join(rep))
    print('wrote', DATA / 'cachefix_sim_report.md')


if __name__ == '__main__':
    if len(sys.argv) >= 2 and sys.argv[1] == 'scan':
        scan()
    else:
        main()
