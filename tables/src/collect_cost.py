"""Collect the training cost recorded by the runs behind Table tab:xmask (MSCA: Native, anchor term A1, A + masked BPR /
SL@K / DNS, XRE). Reads remote result.json and stdout.log files only; trains and times nothing.

Runs: the run_dir of tables/xmask_perseed.csv (masked rows), the eligible MSCA XRE / A1 rows of tables/xre_runs.tsv
(baby / pantry / scientific) and the chosen MSCA Native runs of tables/table1_runs.tsv.
Per run: host, CUDA_VISIBLE_DEVICES, start time, elapsed_sec (whole run: data loading, training, validation every epoch,
test of the best checkpoint), n_batches (len(losses) = n_batches = training steps, one loss per batch, not per epoch),
epochs run = number of 'N EP(...)' lines of stdout.log (asserted equal to n_batches / ceil(n_train / 256)),
sec_per_epoch = elapsed_sec / epochs, epoch_wall_median = median gap between consecutive epoch-end timestamps of the log
(training + validation of one epoch, 1 s resolution), and own_concurrent_same_gpu = number of other P04 runs (every
result.json under <DISK>/user/P04*) on the same host and device whose time window covers more than half of
this run. Jobs of other users are invisible to this count. result.json records no peak-memory field.
Output: tables/cost_runs.csv (one row per run) and a summary on stdout. No macros: the rows ran on different machines
and dates under different loads (see the summary), so their wall-clock times are not a controlled comparison. The
controlled measurement of the manuscript is tables/timing_summary.tsv (build_timing_macros.py: Native / A1 / XRE alone
on one GPU).
Usage (repository root): python -s tables/src/collect_cost.py
"""
import csv
import json
import math
import statistics as st
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / 'tables'
OUT = T / 'cost_runs.csv'
DATASETS = ['baby', 'pantry', 'scientific']
ORDER = ['Native', 'A1', 'mbpr', 'mslk', 'mdns', 'XRE']

REMOTE = r'''
import glob, json, os, re, sys
from datetime import datetime
PATHS = json.loads(sys.stdin.readline())
EP = re.compile(r'(\d+) EP\((\d{6}_\d{6})\)')
out = []
for row, path in PATHS:
    j = json.load(open(path))
    f = os.path.join(os.path.dirname(path), 'stdout.log')
    ts = []
    if os.path.exists(f):
        seen = set()
        for e, t in EP.findall(open(f, errors='replace').read().replace('\r', '\n')):
            if e not in seen:
                seen.add(e)
                ts.append(t)
    mem = [k for k in j if re.search('mem|peak|alloc', k, re.I)]
    out.append(dict(row=row, path=path, config=j.get('config'), dataset=j['dataset'], seed=j['seed'], host=j.get('host'), cvd=j.get('cuda_visible_devices'),
                    gpu=j.get('gpu'), started=j.get('started'), elapsed_sec=j.get('elapsed_sec'), n_batches=j.get('n_batches'),
                    n_losses=len(j.get('losses') or []), best_epoch=j.get('best_epoch'), ep_ts=ts, mem_fields=mem))
allr, seen = [], set()
for p in ('<RUN_ROOT_B>/P04*/**/result.json', '<HOME>/P04*/**/result.json'):
    for f in glob.glob(p, recursive=True):
        try:
            j = json.load(open(f))
        except Exception:
            continue
        k = (j.get('package'), j.get('config'), j.get('seed'), j.get('started'), j.get('host'))
        if k in seen or not j.get('started') or not j.get('elapsed_sec'):
            continue
        seen.add(k)
        allr.append([j.get('host'), str(j.get('cuda_visible_devices')), j['started'], j['elapsed_sec'], j.get('config'), j.get('seed')])
print(json.dumps(dict(runs=out, all=allr)))
'''


def paths():
    out = []
    for r in csv.DictReader(open(T / 'xre_runs.tsv', encoding='utf-8'), delimiter='\t'):
        if r['backbone'] == 'MSCA' and r['row'] in ('XRE', 'A1') and not r['ineligible_reason'] and r['dataset'] in DATASETS:
            out.append((r['row'], r['path']))
    for r in csv.DictReader(open(T / 'table1_runs.tsv', encoding='utf-8'), delimiter='\t'):
        if r['backbone'] == 'MSCA' and r['row'] == 'Native' and r['chosen'] == '1' and r['dataset'] in DATASETS:
            out.append(('Native', r['path']))
    for r in csv.DictReader(open(T / 'xmask_perseed.csv', encoding='utf-8')):
        out.append((r['row'], r['run_dir'] + '/result.json'))
    return out


def main():
    P = paths()
    script = REMOTE.replace('json.loads(sys.stdin.readline())', repr(P))   # the program goes through stdin
    res = subprocess.run(['ssh', '-o', 'BatchMode=yes', 'HOST_A', 'python3', '-'], input=script,
                         capture_output=True, text=True, check=True).stdout
    data = json.loads(res)
    nb = {r['dataset']: math.ceil(int(r['train']) / 256) for r in csv.DictReader(open(T / 'dataset_stats.csv', encoding='utf-8'))}
    win = []
    for h, c, s, e, cfg, sd in data['all']:
        t0 = datetime.strptime(s, '%Y-%m-%d %H:%M:%S')
        win.append((h, c, t0, t0 + timedelta(seconds=float(e)), cfg, sd))
    rows = []
    for r in data['runs']:
        assert not r['mem_fields'], r['mem_fields']
        assert r['n_losses'] == r['n_batches'], r
        ep = len(r['ep_ts'])
        assert ep * nb[r['dataset']] >= r['n_batches'] > (ep - 1) * nb[r['dataset']], (r['path'], ep, r['n_batches'])
        t = [datetime.strptime(x, '%y%m%d_%H%M%S') for x in r['ep_ts']]
        gaps = [(b - a).total_seconds() for a, b in zip(t, t[1:])]
        t0 = datetime.strptime(r['started'], '%Y-%m-%d %H:%M:%S')
        t1 = t0 + timedelta(seconds=float(r['elapsed_sec']))
        conc = sum(1 for h, c, a0, a1, cfg, sd in win if h == r['host'] and c == str(r['cvd']) and (cfg, sd, a0) != (r['config'], r['seed'], t0)
                   and (min(t1, a1) - max(t0, a0)).total_seconds() > 0.5 * (t1 - t0).total_seconds())
        rows.append(dict(row=r['row'], dataset=r['dataset'], seed=r['seed'], host=r['host'], cvd=r['cvd'], gpu=r['gpu'],
                         started=r['started'], elapsed_sec=r['elapsed_sec'], n_batches=r['n_batches'], epochs=ep,
                         best_epoch=r['best_epoch'], sec_per_epoch=round(float(r['elapsed_sec']) / ep, 2),
                         epoch_wall_median=st.median(gaps) if gaps else '', own_concurrent_same_gpu=conc, path=r['path']))
    rows.sort(key=lambda r: (DATASETS.index(r['dataset']), ORDER.index(r['row']), int(r['seed'])))
    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    print('wrote', OUT, len(rows), 'runs')
    for d in DATASETS:
        for k in ORDER:
            v = [r for r in rows if r['dataset'] == d and r['row'] == k]
            s = [r['sec_per_epoch'] for r in v]
            print(f"{d:10s} {k:6s} s/epoch median {st.median(s):6.1f} (min {min(s):5.1f}, max {max(s):5.1f}) "
                  f"hosts {len({r['host'] for r in v})} dates {sorted({r['started'][:10] for r in v})} "
                  f"own concurrent runs on the GPU {[r['own_concurrent_same_gpu'] for r in v]}")


if __name__ == '__main__':
    main()
