"""Collect the per-run results of Table 2 (SOTA comparison under the common protocol). Only reads remote files.

Usage (local machine, repository root):  python tables/src/collect_table2.py
The script sends itself over ssh to the two hosts (argument `scan <host>`), gathers one row per candidate run, applies
the eligibility and duplicate rules below, and writes
  tables/table2_runs.tsv   every candidate run, its eligibility, the chosen flag, and the duplicate resolution
  tables/table2_cells.csv  one row per (model, dataset): mean over the chosen runs, seeds, duplicates, missing seeds

Sources
  HOST_A  <RUN_ROOT_B>/P04_pod_archive/p*/...runs/<pkg>/<run>/result.json   (pods, archived)
           <HOME>/P04_t2fill/runs/<pkg>/<run>/result.json              (gap cells)
           P04_CrowdedTop_journal_pods/results/best_eval/baseline_paper6_perseed.csv (MENTOR Baby only, see RULE 4)
  HOST_D <POD_ROOT>/p04e/q45/runs/<pkg>/<run>/result.json                          (seeds 4-5)
           <POD_ROOT>/p04e/qt2s123/runs/<pkg>/<run>/result.json                      (seeds 1-3 re-run)

Rules (fixed before the values of duplicate runs were compared)
  1. Configuration: only the author-derived configuration of each (model, dataset) in ACTIVE.
  2. Eligible run: status SUCCESS, mode train, test_eval best_primed, seed 1-5, run directory not marked
     `.superseded_*`, data canonical_v2 (Baby also accepts canonical v1, whose Baby files are byte-identical).
     Test metrics are best_test, i.e. the test metrics of the checkpoint selected by validation Recall@20.
  3. Duplicates of one (model, dataset, seed): lowest queue rank wins, then the earliest completion
     (started + elapsed_sec).  Rank 0 = Table-2 protocol queue designated for that seed
     (seeds 1-3: HOST_D qt2s123, see RULE 5; seeds 4-5: HOST_D q45).
     Rank 1 = earlier Table-2 protocol queues for seeds 1-3 (HOST_A authorcfg, newbl and t2fill queues).
     Rank 2 = any other pod queue with the same configuration.  Rank 3 = RULE 4 source.
  4. MENTOR Baby has no seed-1 pod run with the MENTOR configuration.  Seed-level rows of baseline_paper6_perseed.csv
     (best-checkpoint re-evaluation with the scoring cache rebuilt) enter as rank-3 candidates for MENTOR Baby only.
  5. Pre-declared rule (author decision log entry #100, 2026-09-29T12:34:11Z, fixed before any qt2s123 result was
     seen): for seeds 1-3 prefer the re-run from qt2s123 over the archived pod run.
     qt2s123 re-ran all 8 models x 3 datasets x seeds 1-3 on HOST_D with the same repository, environment and lane
     script as q45 (seeds 4-5), so it is rank 0 for seeds 1-3 and every earlier seed-1-3 run (archived pods and the HOST_A t2fill gap runs) falls to rank 1 or 2.  Every chosen run is then an HOST_D run.
"""
import csv
import datetime as dt
import glob
import json
import os
import subprocess
import sys

MODELS = ['MSCA', 'COHESION', 'SMORE', 'PGL', 'MENTOR', 'DiffMM', 'LGMRec', 'FREEDOM']
DATASETS = ['baby', 'pantry', 'scientific']
SEEDS = [1, 2, 3, 4, 5]


def active_config(model, dataset):
    if model in ('MSCA', 'COHESION', 'SMORE'):
        return f'{model}_A5_{dataset}'
    if model in ('PGL', 'DiffMM'):
        return f'{model}_A4_baby' if dataset == 'baby' else f'{model}_A6_{dataset}'
    if model in ('LGMRec', 'FREEDOM'):
        return f'{model}_A4_{dataset}'
    if model == 'MENTOR':
        return 'MENTOR'
    raise KeyError(model)


ACTIVE = {(m, d): active_config(m, d) for m in MODELS for d in DATASETS}
FIELDS = ['model', 'dataset', 'seed', 'host', 'queue', 'config', 'status', 'mode', 'test_eval', 'data', 'superseded',
          'started', 'elapsed_sec', 'best_epoch', 'test_r10', 'test_n10', 'test_r20', 'test_n20', 'gpu', 'origin']

# ----------------------------------------------------------------------------------------------- remote scan side
HOST_A_GLOBS = ['<RUN_ROOT_B>/P04_pod_archive/p*/runs/*/*/result.json',
                '<RUN_ROOT_B>/P04_pod_archive/p*/*/runs/*/*/result.json',
                '<RUN_ROOT_B>/P04_pod_archive/p*/*/*/runs/*/*/result.json',
                '<HOME>/P04_t2fill/runs/*/*/result.json']
EUR_GLOBS = ['<POD_ROOT>/p04e/q45/runs/*/*/result.json', '<POD_ROOT>/p04e/qt2s123/runs/*/*/result.json']
PAPER6 = ('<HOME>/Desktop/local/P04_CrowdedTop_journal_pods/results/best_eval/'
          'baseline_paper6_perseed.csv')


def queue_of(path):
    if path.startswith('<HOME>/P04_t2fill/'):
        return 't2fill'
    if path.startswith('<POD_ROOT>/p04e/q45/'):
        return 'q45'
    if path.startswith('<POD_ROOT>/p04e/qt2s123/'):
        return 'qt2s123'
    rel = path.split('/P04_pod_archive/')[1].split('/runs/')[0]  # e.g. p7/authorcfg, p9/newbl/cohesion, p3
    return 'archive/' + rel


def scan(host):
    globs = HOST_A_GLOBS if host == 'HOST_A' else EUR_GLOBS
    files = sorted({f for g in globs for f in glob.glob(g)})
    w = csv.writer(sys.stdout, delimiter='\t', lineterminator='\n')
    w.writerow(FIELDS)
    for f in files:
        try:
            d = json.load(open(f))
        except Exception as e:  # unreadable file: report on stderr, skip
            print(f'UNREADABLE {f} {e}', file=sys.stderr)
            continue
        model = d.get('model_class') or d.get('model')
        dataset = str(d.get('dataset', '')).split('@')[0]
        if model not in MODELS or dataset not in DATASETS or d.get('config') != ACTIVE[(model, dataset)]:
            continue
        bt = d.get('best_test') or {}
        run_dir = os.path.basename(os.path.dirname(f))
        w.writerow([model, dataset, d.get('seed'), host, queue_of(f), d.get('config'), d.get('status'), d.get('mode'),
                    d.get('test_eval'), os.path.basename(str(d.get('data_root', '')).rstrip('/')),
                    int('.superseded' in run_dir), d.get('started'), d.get('elapsed_sec'), d.get('best_epoch'),
                    bt.get('recall@10'), bt.get('ndcg@10'), bt.get('recall@20'), bt.get('ndcg@20'),
                    (d.get('gpu') or '').replace('NVIDIA ', ''), f])
    if host == 'HOST_A':
        with open(PAPER6, newline='') as fh:
            for r in csv.DictReader(fh):
                if r['model'] == 'MENTOR' and r['dataset'] == 'baby':
                    w.writerow(['MENTOR', 'baby', r['seed'], host, 'paper6/' + r['source'], 'MENTOR', 'SUCCESS',
                                'train', 'reeval_best', 'paper6_ledger', 0, '', '', '', '', '', r['R20'], r['N20'], '',
                                PAPER6])


# ------------------------------------------------------------------------------------------------ local merge side
HERE = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.dirname(HERE)
SSH = {'HOST_A': ['ssh', '-o', 'BatchMode=yes', 'HOST_A'],
       'eur': ['ssh', '-i', os.path.expanduser('<SSH_KEY>'), '-o', 'BatchMode=yes', '-p', '<PORT>',
               'user@HOST_D']}
T2_QUEUES_S123 = ('archive/p6/authorcfg', 'archive/p7/authorcfg', 'archive/p8/authorcfg', 'archive/p9/authorcfg',
                  'archive/p10/authorcfg', 'archive/p9/newbl/', 'archive/p10/newbl/', 't2fill')


def remote_rows(host):
    with open(os.path.abspath(__file__), 'rb') as fh:
        out = subprocess.run(SSH[host] + ['python3', '-', 'scan', host], stdin=fh, capture_output=True, check=True)
    if out.stderr.strip():
        print(out.stderr.decode(), file=sys.stderr)
    return list(csv.DictReader(out.stdout.decode().splitlines(), delimiter='\t'))


def eligibility(r):
    seed = int(r['seed'])
    if r['queue'].startswith('paper6/'):
        return ('', 3) if (r['model'], r['dataset']) == ('MENTOR', 'baby') else ('paper6 row outside RULE 4', 9)
    reasons = []
    if r['status'] != 'SUCCESS':
        reasons.append(f"status {r['status']}")
    if r['mode'] != 'train':
        reasons.append(f"mode {r['mode']}")
    if r['test_eval'] != 'best_primed':
        reasons.append(f"test_eval {r['test_eval'] or 'none'}")
    if seed not in SEEDS:
        reasons.append('seed outside 1-5')
    if r['superseded'] == '1':
        reasons.append('superseded')
    if not (r['data'] == 'canonical_v2' or (r['dataset'] == 'baby' and r['data'] == 'canonical')):
        reasons.append(f"data {r['data']}")
    if seed <= 3:
        rank = 0 if r['queue'] == 'qt2s123' else 1 if r['queue'].startswith(T2_QUEUES_S123) else 2  # RULE 5
    else:
        rank = 0 if r['queue'] == 'q45' else 2
    return '; '.join(reasons), rank


def completed(r):
    try:
        t = dt.datetime.strptime(r['started'], '%Y-%m-%d %H:%M:%S') + dt.timedelta(seconds=float(r['elapsed_sec']))
        return t.strftime('%Y-%m-%d %H:%M:%S')
    except (ValueError, TypeError):
        return '9999'  # unknown completion time sorts last within its rank


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def std(xs):
    if len(xs) < 2:
        return None
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def fmt(x):
    return '' if x is None else f'{x:.6f}'


def main():
    rows = remote_rows('HOST_A') + remote_rows('eur')
    for r in rows:
        r['ineligible_reason'], r['rank'] = eligibility(r)
        r['completed'] = completed(r)
        r['chosen'] = 0
        r['duplicate_note'] = ''
    groups = {}
    for r in rows:
        if not r['ineligible_reason']:
            groups.setdefault((r['model'], r['dataset'], int(r['seed'])), []).append(r)
    dup_notes = {}
    for key, rr in groups.items():
        rr.sort(key=lambda r: (r['rank'], r['completed']))
        rr[0]['chosen'] = 1
        if len(rr) > 1:
            note = (f"s{key[2]}: chose {rr[0]['queue']} (rank {rr[0]['rank']}) over "
                    + ', '.join(f"{r['queue']} (rank {r['rank']})" for r in rr[1:]))
            for r in rr:
                r['duplicate_note'] = note
            dup_notes.setdefault(key[:2], []).append(note)
    rows.sort(key=lambda r: (MODELS.index(r['model']), DATASETS.index(r['dataset']), int(r['seed']), -r['chosen'],
                             r['queue']))
    cols = ['model', 'dataset', 'seed', 'chosen', 'ineligible_reason', 'rank', 'duplicate_note', 'host', 'queue',
            'config', 'status', 'mode', 'test_eval', 'data', 'superseded', 'completed', 'best_epoch', 'test_r10',
            'test_n10', 'test_r20', 'test_n20', 'gpu', 'origin']
    with open(os.path.join(TABLES, 'table2_runs.tsv'), 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter='\t', lineterminator='\n', extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(TABLES, 'table2_cells.csv'), 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh, lineterminator='\n')
        w.writerow(['model', 'dataset', 'config', 'n', 'seeds', 'missing_seeds', 'r20_mean', 'n20_mean', 'r20_std',
                    'n20_std', 'r10_mean', 'n10_mean', 'queues', 'duplicates'])
        for m in MODELS:
            for d in DATASETS:
                ch = sorted((g[0] for k, g in groups.items() if k[:2] == (m, d)), key=lambda r: int(r['seed']))
                seeds = [int(r['seed']) for r in ch]
                r20 = [float(r['test_r20']) for r in ch]
                n20 = [float(r['test_n20']) for r in ch]
                r10 = [float(r['test_r10']) for r in ch if r['test_r10'] not in ('', 'None')]
                n10 = [float(r['test_n10']) for r in ch if r['test_n10'] not in ('', 'None')]
                full10 = len(r10) == len(ch)
                w.writerow([m, d, ACTIVE[(m, d)], len(ch), '/'.join(map(str, seeds)),
                            '/'.join(str(s) for s in SEEDS if s not in seeds), fmt(mean(r20)), fmt(mean(n20)),
                            fmt(std(r20)), fmt(std(n20)), fmt(mean(r10)) if full10 else '',
                            fmt(mean(n10)) if full10 else '', '/'.join(r['queue'] for r in ch),
                            ' | '.join(dup_notes.get((m, d), []))])
    print(f"{len(rows)} candidate rows, {sum(r['chosen'] for r in rows)} chosen, "
          f"{sum(len(g) > 1 for g in groups.values())} duplicate groups", file=sys.stderr)


if __name__ == '__main__':
    if len(sys.argv) >= 3 and sys.argv[1] == 'scan':
        scan(sys.argv[2])
    else:
        main()
