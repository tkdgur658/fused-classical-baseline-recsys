"""Collect the per-run results of the top-K loss rows of Table tab:main (MSCA block: SL@K, PSL, SSM, each replacing the
BPR term of MSCA). Only reads remote files.

Usage (local machine, repository root):  python tables/src/collect_topk.py   (after collect_table1.py: the MSCA Native
runs chosen in tables/table1_runs.tsv are the reference of the configuration checks below)
The script sends itself over ssh to HOST_A (argument `scan` followed by the Native result.json paths), gathers one row per
candidate run, applies the rules below, and writes
  tables/topk_runs.tsv     every candidate run with its eligibility and the chosen flag (ineligible_reason empty and
                           chosen 1 = used), its temperature, queue attempt and the selected temperature of its cell
  tables/topk_perseed.csv  one row per (backbone, row, dataset, seed): the chosen run of that seed
  tables/topk_cells.csv    one row per (backbone, row, dataset): mean and sample SD over the chosen seeds, in the columns of
                           tables/table1_cells.csv plus tau_star and score (cosine | dot). A cell with fewer than five seeds is still written (n < 5);
                           tables/src/build_table1_main.py prints it as \\todo{pending} until every seed exists.
table1_cells.csv is left untouched (it feeds native_rows.py and the other tables); build_table1_main.py merges this file.

Source (HOST_A, <RUN_ROOT_B>/P04_topk/README.txt): every result.json under
<RUN_ROOT_B>/P04_topk/runs_topk*/objx_s/ (queues A = baby, pantry, scientific and B = instruments, clothing, sports,
cosine; E = the dot grid and the seeds 2-5 appended by the score selection; D = lr diagnostics, never collected), with the
queue attempts from claims_topk*/<job>__a<n>/run_id.
Repo <HOME>/P04_topk_wt branch topk e2b5b7a (from xre_s2 406a7a7; topkE commit 108a1a4 adds the score
function), package objx_s, configs <ds>_<loss>_t<XXX>[_dot] = the committed <ds>_native.yaml (the Table-I Native config)
with the row comment, objx_main and objx_tau replaced (tools/topk_package.py), plus the line 'objx_score: dot' for _dot
(tools/topk_score_package.py). Losses (models/objx_m/src/objloss.py, re-implementation of IR-Benchmark @ 1dd8de9):
  ssm  SSM, sampled softmax loss (optim_Softmax)                                   grid tau  {0.1, 0.2, 0.5}
  psl  PSL (Yang et al., NeurIPS 2024; optim_PSL method 1, tau_score 2)            grid tau* {0.05, 0.1, 0.25}
  slk  SL@K (Yang et al., KDD 2025; optim_SLatK, K 20, tau_beta 2.25, quantile every 20 epochs)  grid tau {0.1, 0.2, 0.5}
  All three: 1000 uniform negatives. Score function (README topkE): cosine (objx_score absent; IR-Benchmark convention,
  cosine in the loss and at inference) or dot (suffix _dot; the same formulas with the inner product in the loss and the
  SL@K quantile, ranking by the dot product of MSCA). Diagnostic configs outside the grid are never collected: _lr<NN>
  (topkD) and _dotrank (topkE step 2) do not match CFG.
Rules (fixed before this collection)
  1. Eligible run: status SUCCESS, mode train, test_eval best_primed, seed 1-5, data copy canonical_v2 (Baby
     also accepts canonical v1, byte-identical, as the Table-I Baby rows), config <ds>_<loss>_t<XXX>[_dot] of the grid.
  2. Same MSCA configuration as the Native cell except the loss (checked per run; a failed check makes the run ineligible
     and is printed): (a) the run's model config, after dropping its objx_score line, differs from <ds>_native.yaml of its
     repo only in the row comment, objx_main (= the loss) and objx_tau (= the tau of the config name), and objx_score is
     'dot' for a _dot config and absent for a cosine config; (b) that <ds>_native.yaml, overall.yaml and the
     dataset yaml are byte-identical (sha256) to the files of every chosen MSCA Native run of tables/table1_runs.tsv whose
     files still exist (pod runs whose files are gone are skipped, and the number of compared Native runs is reported);
     (c) the sha256 of the resolved model sources (models.msca, models.objloss, models.objx_s, models.refset) and the
     parameter count equal those of the Native runs of that dataset, except that models.objloss may carry either sha of
     OBJLOSS_OK (e2b5b7a, or 108a1a4 whose cosine path is bitwise identical); a dot run must carry 108a1a4. All runs of a cell must share one run fingerprint
     (main.py cell_fingerprint: config, package and core files, data copy); otherwise the collector stops.
  3. Score function and temperature: (score, tau)* per (dataset, loss) = topk_selection_score.json of select_topk.py
     (README topkE step 3, pre-registered: over score {cosine, dot} x the three tau of the loss, max seed-1 VALID Recall@20
     of best_valid_primed, ties valid NDCG@20, then cosine before dot, then smaller tau; the earliest-started eligible
     seed-1 grid run per (score, tau); selected only once all six grid runs exist). The collector recomputes this rule from
     the scanned runs and stops if it disagrees with topk_selection_score.json (score, tau, config or a grid run path), or
     if a selection exists while fewer than six eligible grid runs are found. Without a selection the cell is pending (its
     seed-1 grid is printed on stderr). The old cosine-only topk_selection.json is history only and is not read.
  4. Cell = the runs of config <ds>_<loss>_<tau*>[_dot], seeds 1-5 (seed 1 = the grid run at (score, tau)*). Runs of the
     other score (e.g. cosine seeds 2-5 appended under the old rule in a group where dot won) stay chosen 0. ONE run per seed (no
     replicates): if a seed has more than one eligible run, the run claimed by queue attempt 1 (claims_topk<X>/<job>__a1)
     is used, and the collector stops when this does not single out exactly one run.
  5. Cell mean over the chosen seeds, SD = sample SD over seeds (ddof 1).
"""
import csv
import datetime as dt
import glob
import hashlib
import json
import os
import re
import statistics as st
import subprocess
import sys

BASE = '<RUN_ROOT_B>/P04_topk'
DATASETS = ['baby', 'pantry', 'scientific', 'instruments', 'clothing', 'sports']
LOSSES = {'ssm': 'SSM', 'psl': 'PSL', 'slk': 'SL@K'}   # config tag -> row key of Table tab:main
GRID = {'ssm': {'t010': 0.1, 't020': 0.2, 't050': 0.5},
        'psl': {'t005': 0.05, 't010': 0.1, 't025': 0.25},
        'slk': {'t010': 0.1, 't020': 0.2, 't050': 0.5}}
SEEDS = [1, 2, 3, 4, 5]
SCORES = {'cosine': '', 'dot': '_dot'}   # score function -> config suffix; order = tie-break order (cosine first)
# grid configs only: the diagnostics <cfg>_lr<NN> (topkD) and <cfg>_dotrank (topkE step 2) do not match, are not collected
CFG = re.compile(r'^(?P<ds>[a-z]+)_(?P<loss>ssm|psl|slk)_(?P<tag>t\d{3})(?P<score>_dot)?$')
# the only keys allowed to differ from <ds>_native.yaml (plus the row comment); objx_score is absent from the Native config,
# so cfg_check drops it (after checking its value) before the line-by-line comparison
LOSS_KEYS = ('objx_main', 'objx_tau', 'objx_score')
# accepted sha256[:16] of models.objloss: c7dffbb26a453a59 = e2b5b7a (= the code of the Native / XRE runs), 8f0102a249bc1e80 =
# 108a1a4 (topkE, adds objx_score; runs started after 2026-10-03 ~14:47 UTC). The cosine path of 108a1a4 is bitwise identical
# to e2b5b7a (loss, gradient, SL@K quantile, Objective end to end): tools/objx_score_unit_test.py ->
# results/topk/objx_score_unit.json of the P04_topk_wt repo (= <RUN_ROOT_B>/P04_topk/check_score_cpu.json). Both
# shas checked against `git show <commit>:models/objx_m/src/objloss.py | sha256sum` and the result.json files of the runs.
OBJLOSS_OLD, OBJLOSS_NEW = 'c7dffbb26a453a59', '8f0102a249bc1e80'
OBJLOSS_OK = {OBJLOSS_OLD, OBJLOSS_NEW}
FIELDS = ['package', 'config', 'dataset', 'seed', 'status', 'mode', 'test_eval', 'data', 'gpu', 'started', 'elapsed_sec',
          'best_epoch', 'fingerprint', 'test_r10', 'test_n10', 'test_r20', 'test_n20', 'valid_r20', 'valid_n20',
          'numel', 'src_sha', 'cfg_check', 'sha_overall', 'sha_dataset', 'sha_native_cfg', 'path']


# ----------------------------------------------------------------------------------------------- remote side
def sha(p):
    try:
        with open(p, 'rb') as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except OSError:
        return ''


def flat_yaml(p):
    """the 'key: value' lines of a flat config file (comments and blank lines dropped)"""
    with open(p, encoding='utf-8') as fh:
        return [ln for ln in fh.read().split('\n') if ln.strip() and not ln.lstrip().startswith('#')]


def cfg_check(model_cfg, ds, loss, tau, score):
    """'ok' if the model config, without its objx_score line, equals <ds>_native.yaml of the same directory up to the row
    comment, objx_main and objx_tau, with objx_main = loss, objx_tau = tau, and objx_score = dot for score dot (key absent
    for cosine); otherwise the reason"""
    nat = os.path.join(os.path.dirname(model_cfg), f'{ds}_native.yaml')
    try:
        a, b = flat_yaml(model_cfg), flat_yaml(nat)
    except OSError as e:
        return f'unreadable config ({type(e).__name__})', nat
    key = (lambda x: x.split(':', 1)[0].strip())
    sc = [x.split(':', 1)[1].strip() for x in a if key(x) == 'objx_score']
    if sc != (['dot'] if score == 'dot' else []):
        return f'objx_score lines {sc} do not match score {score}', nat
    if any(key(x) == 'objx_score' for x in b):
        return f'{ds}_native.yaml carries objx_score', nat
    a = [x for x in a if key(x) != 'objx_score']
    if len(a) != len(b):
        return f'{len(a)} config lines vs {len(b)} in {ds}_native.yaml', nat
    diff = [(x, y) for x, y in zip(a, b) if x != y]
    bad = [x for x, y in diff if x.split(':')[0].strip() not in LOSS_KEYS or y.split(':')[0] != x.split(':')[0]]
    if bad:
        return 'differs from native beyond the loss: ' + ' | '.join(bad), nat
    kv = dict((x.split(':', 1)[0].strip(), x.split(':', 1)[1].strip()) for x in a)
    if kv.get('objx_main') != loss or abs(float(kv.get('objx_tau', 'nan')) - tau) > 1e-12:
        return f"objx_main {kv.get('objx_main')} / objx_tau {kv.get('objx_tau')} != {loss} / {tau}", nat
    return 'ok', nat


def meta(d):
    cf = d.get('config_files') or ['', '', '']
    src = d.get('resolved_sources') or {}
    return dict(numel=(d.get('params') or {}).get('numel'),
                src_sha=';'.join(f"{k}={(v or {}).get('sha256', '')[:16]}" for k, v in sorted(src.items())),
                sha_overall=sha(cf[0]), sha_dataset=sha(cf[1]), sha_model=sha(cf[-1]))


def scan(native_paths):
    out = {'runs': [], 'native': {}, 'selection': {}, 'claims': {}, 'absent': []}
    for f in sorted(glob.glob(f'{BASE}/runs_topk*/objx_s/*/result.json')):
        try:
            with open(f, encoding='utf-8') as fh:
                d = json.load(fh)
        except Exception as e:  # e.g. a file caught mid-write: reported, skipped
            print(f'UNREADABLE {f} {type(e).__name__}', file=sys.stderr)
            continue
        bt, bv = d.get('best_test') or {}, d.get('best_valid_primed') or {}
        ds = str(d.get('dataset', '')).split('@')[0]
        m = CFG.match(str(d.get('config')))
        if m and m['ds'] == ds and m['tag'] in GRID.get(m['loss'], {}):
            chk, nat = cfg_check((d.get('config_files') or [''])[-1], ds, m['loss'], GRID[m['loss']][m['tag']],
                                 'dot' if m['score'] else 'cosine')
        else:
            chk, nat = 'config name outside the top-K grid (diagnostic config, not collected)', ''
        mt = meta(d)
        out['runs'].append(dict(
            package=d.get('package'), config=d.get('config'), dataset=ds, seed=d.get('seed'), status=d.get('status'),
            mode=d.get('mode'), test_eval=d.get('test_eval'),
            data=os.path.basename(str(d.get('data_root', '')).rstrip('/')), gpu=(d.get('gpu') or '').replace('NVIDIA ', ''),
            started=d.get('started'), elapsed_sec=d.get('elapsed_sec'), best_epoch=d.get('best_epoch'),
            fingerprint=d.get('fingerprint'), test_r10=bt.get('recall@10'), test_n10=bt.get('ndcg@10'),
            test_r20=bt.get('recall@20'), test_n20=bt.get('ndcg@20'), valid_r20=bv.get('recall@20'),
            valid_n20=bv.get('ndcg@20'), numel=mt['numel'], src_sha=mt['src_sha'], cfg_check=chk,
            sha_overall=mt['sha_overall'], sha_dataset=mt['sha_dataset'], sha_native_cfg=sha(nat) if nat else '', path=f))
    for f in native_paths:
        try:
            with open(f, encoding='utf-8') as fh:
                d = json.load(fh)
        except Exception as e:
            out['absent'].append(f'{f} {type(e).__name__}')
            continue
        out['native'][f] = meta(d)
    sel = os.path.join(BASE, 'topk_selection_score.json')   # topk_selection.json (old cosine-only rule): history, not read
    if os.path.isfile(sel):
        with open(sel, encoding='utf-8') as fh:
            out['selection'] = json.load(fh)
    dv = os.path.join(BASE, 'diverged_grid.txt')             # amendment 2026-10-04 (README): configs that failed twice
    out['diverged'] = []
    if os.path.isfile(dv):
        with open(dv, encoding='utf-8') as fh:
            out['diverged'] = sorted({l.split('#')[0].strip() for l in fh} - {''})
    for c in glob.glob(f'{BASE}/claims_topk*/*/run_id'):
        q = os.path.basename(os.path.dirname(os.path.dirname(c)))[len('claims_topk'):]
        with open(c, encoding='utf-8') as fh:
            rid = fh.read().strip()
        job = os.path.basename(os.path.dirname(c))
        att = job.rsplit('__a', 1)[-1] if '__a' in job else ''
        out['claims'].setdefault(f'{BASE}/runs_topk{q}/objx_s/{rid}/result.json', []).append(att)
    json.dump(out, sys.stdout)


# ------------------------------------------------------------------------------------------------ local side
HERE = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.dirname(HERE)
SSH = ['ssh', '-o', 'BatchMode=yes', 'HOST_A']


def remote(args):
    with open(os.path.abspath(__file__), 'rb') as fh:
        out = subprocess.run(SSH + ['python3', '-', 'scan'] + args, stdin=fh, capture_output=True, check=True)
    if out.stderr.strip():
        print(out.stderr.decode(errors='replace'), file=sys.stderr)
    return json.loads(out.stdout.decode())


def native_runs():
    """dataset -> result.json paths of the chosen MSCA Native runs (tables/table1_runs.tsv, collect_table1.py)"""
    out = {}
    with open(os.path.join(TABLES, 'table1_runs.tsv'), newline='', encoding='utf-8') as fh:
        for r in csv.DictReader(fh, delimiter='\t'):
            if (r['backbone'], r['row'], r['chosen']) == ('MSCA', 'Native', '1'):
                out.setdefault(r['dataset'], []).append(r['path'])
    return out


def src_dict(s):
    """'k=v;k=v' (meta src_sha) -> dict"""
    return dict(x.split('=', 1) for x in (s or '').split(';') if x)


def ineligible(r):
    why = []
    if r['status'] != 'SUCCESS':
        why.append(f"status {r['status']}")
    if r['mode'] != 'train':
        why.append(f"mode {r['mode']}")
    if r['test_eval'] != 'best_primed':
        why.append(f"test_eval {r['test_eval'] or 'none'}")
    if r['test_r20'] is None or r['test_n20'] is None:
        why.append('no test metrics')
    if r['seed'] not in SEEDS:
        why.append('seed outside 1-5')
    # (device-name filter of the original collector removed in the public copy; other rules unchanged)
    if not (r['data'] == 'canonical_v2' or (r['dataset'] == 'baby' and r['data'] == 'canonical')):
        why.append(f"data {r['data']}")
    if r['package'] != 'objx_s' or not CFG.match(str(r['config'])) or r['dataset'] not in DATASETS:
        why.append('not a top-K configuration')
    if r['cfg_check'] != 'ok':
        why.append(f"config check: {r['cfg_check']}")
    return why


def completed(r):
    try:
        t = dt.datetime.strptime(r['started'], '%Y-%m-%d %H:%M:%S')
        return t + dt.timedelta(seconds=float(r['elapsed_sec'] or 0))
    except (TypeError, ValueError):
        return dt.datetime.max


def f6(x):
    return '' if x is None else f'{x:.6f}'


def main():
    nat_paths = native_runs()
    res = remote([p for ds in DATASETS for p in nat_paths.get(ds, [])])
    for a in res['absent']:
        print(f'Native result.json unreadable: {a}', file=sys.stderr)
    # reference of RULE 2 per dataset: the Native runs of that dataset (code / size always; files where they exist)
    ref = {}
    for ds in DATASETS:
        nm = [res['native'][p] for p in nat_paths.get(ds, []) if p in res['native']]
        src, numel = {m['src_sha'] for m in nm}, {m['numel'] for m in nm}
        if len(src) > 1 or len(numel) > 1:
            raise SystemExit(f'MSCA Native runs of {ds} disagree among themselves: sources {src}, numel {numel}')
        files = {k: {m[k] for m in nm if m[k]} for k in ('sha_overall', 'sha_dataset', 'sha_model')}
        if any(len(v) > 1 for v in files.values()):
            raise SystemExit(f'MSCA Native config files of {ds} disagree among themselves: {files}')
        ref[ds] = dict(n=len(nm), src=next(iter(src), None), numel=next(iter(numel), None),
                       n_files=sum(1 for m in nm if m['sha_model']), **{k: next(iter(v), None) for k, v in files.items()})
        print(f"{ds}: RULE 2 reference = {len(nm)} MSCA Native runs ({ref[ds]['n_files']} with config files on disk)",
              file=sys.stderr)
    rows = res['runs']
    for r in rows:
        m = CFG.match(str(r['config']))
        r['loss'], r['row'] = (m['loss'], LOSSES[m['loss']]) if m else ('', '')
        r['tau'] = GRID[m['loss']].get(m['tag'], '') if m else ''
        r['score'] = ('dot' if m['score'] else 'cosine') if m else ''
        r['backbone'] = 'MSCA'
        r['attempt'] = '/'.join(res['claims'].get(r['path'], [])) or '?'
        r['root'] = r['path'][len(BASE) + 1:].split('/')[0]
        why = ineligible(r)
        g = ref.get(r['dataset'])
        if g and not why:
            if g['n'] == 0:
                why.append('no MSCA Native run of this dataset to compare with')
            if g['src'] is not None:   # equal sources, models.objloss within OBJLOSS_OK (see there)
                rs, gs = src_dict(r['src_sha']), src_dict(g['src'])
                if gs.get('models.objloss') != OBJLOSS_OLD:
                    raise SystemExit(f"{r['dataset']}: Native models.objloss sha {gs.get('models.objloss')} != {OBJLOSS_OLD}")
                if set(rs) != set(gs) or any(rs[k] != gs[k] for k in gs if k != 'models.objloss'):
                    why.append('model sources differ from the Native runs')
                elif rs['models.objloss'] not in OBJLOSS_OK:
                    why.append(f"models.objloss sha {rs['models.objloss']} not in {sorted(OBJLOSS_OK)}")
                elif r['score'] == 'dot' and rs['models.objloss'] != OBJLOSS_NEW:
                    why.append('dot config run without the objx_score code (models.objloss sha)')
            if g['numel'] is not None and r['numel'] != g['numel']:
                why.append(f"parameter count {r['numel']} != Native {g['numel']}")
            for k, kk in (('sha_overall', 'sha_overall'), ('sha_dataset', 'sha_dataset'), ('sha_native_cfg', 'sha_model')):
                if g[kk] and r[k] != g[kk]:
                    why.append(f'{k} differs from the Native run files')
        r['ineligible_reason'] = '; '.join(why)
        r['chosen'] = '0'
        r['tau_star'] = ''
        if why:
            print(f"INELIGIBLE {r['config']} s{r['seed']} {r['root']}: {r['ineligible_reason']}", file=sys.stderr)
    ok = [r for r in rows if not r['ineligible_reason']]
    # RULE 3: (score, tau)* from topk_selection_score.json, re-derived from the six seed-1 grid runs
    tau_star = {}   # (ds, loss) -> (score, tag)
    order = list(SCORES)
    for ds in DATASETS:
        for loss, grid in GRID.items():
            got = {}
            for sc, suf in SCORES.items():
                for tag in grid:
                    rr = [r for r in ok if r['config'] == f'{ds}_{loss}_{tag}{suf}' and r['seed'] == 1
                          and r['valid_r20'] is not None]
                    if rr:
                        got[(sc, tag)] = sorted(rr, key=lambda x: (x['started'] or '', x['path']))[0]
            sel = res['selection'].get(f'{ds}/{loss}')
            info = ', '.join(f"{sc}:{grid[t]} {got[(sc, t)]['valid_r20']:.4f}/{got[(sc, t)]['valid_n20'] or 0:.4f}"
                             for sc in SCORES for t in grid if (sc, t) in got)
            # diverged grid configs (failed twice from non-finite losses, P04_topk/diverged_grid.txt) leave the grid
            excl = {(sc, t) for sc, suf in SCORES.items() for t in grid
                    if f'{ds}_{loss}_{t}{suf}' in set(res.get('diverged') or [])}
            if excl & set(got):
                raise SystemExit(f'{ds}/{loss}: a config in diverged_grid.txt has an eligible seed-1 run: {sorted(excl)}')
            n_grid = len(SCORES) * len(grid) - len(excl)
            if len(got) < n_grid:
                if sel:
                    raise SystemExit(f'topk_selection_score.json selects {ds}/{loss} = {sel.get("config")}, but the scan '
                                     f'finds only {len(got)}/{n_grid} eligible seed-1 grid runs: {sorted(got)}')
                print(f'{ds}/{LOSSES[loss]}: (score, tau)* pending (seed-1 grid {len(got)}/{n_grid}; valid R@20/N@20 '
                      f'{info})', file=sys.stderr)
                continue
            best = sorted((-x['valid_r20'], -(x['valid_n20'] or 0.0), order.index(sc), grid[t], sc, t)
                          for (sc, t), x in got.items())[0]
            b_sc, b_tag = best[4], best[5]
            b_cfg = f'{ds}_{loss}_{b_tag}{SCORES[b_sc]}'
            if not sel:
                print(f'{ds}/{LOSSES[loss]}: seed-1 grid complete (rule gives {b_cfg}), topk_selection_score.json has no '
                      'selection yet -> pending', file=sys.stderr)
                continue
            sel_runs = {k: (v or {}).get('run') for k, v in (sel.get('grid') or {}).items()}
            our_runs = {f'{sc}:{t}': x['path'] for (sc, t), x in got.items()}
            if (sel.get('config'), sel.get('score'), sel.get('tau_star')) != (b_cfg, b_sc, grid[b_tag]) \
                    or sel_runs != our_runs:
                raise SystemExit(f"topk_selection_score.json selects {sel.get('config')} ({sel.get('score')}, tau "
                                 f"{sel.get('tau_star')}) for {ds}/{loss}, the rule on the scanned runs gives {b_cfg} "
                                 f"({info}); grid runs equal: {sel_runs == our_runs}")
            tau_star[(ds, loss)] = (b_sc, b_tag)
            print(f'{ds}/{LOSSES[loss]}: (score, tau)* = ({b_sc}, {grid[b_tag]}) (valid R@20/N@20 {info})',
                  file=sys.stderr)
    # RULE 4: one run per seed of the selected config
    chosen = {}
    for (ds, loss), (sc, tag) in tau_star.items():
        cfg = f'{ds}_{loss}_{tag}{SCORES[sc]}'
        for r in rows:
            if r['config'] == cfg:
                r['tau_star'] = GRID[loss][tag]
        for s in SEEDS:
            rr = [r for r in ok if r['config'] == cfg and r['seed'] == s]
            if len(rr) > 1:
                a1 = [r for r in rr if r['attempt'] == '1']
                if len(a1) != 1:
                    raise SystemExit(f'{cfg} seed {s}: {len(rr)} eligible runs and {len(a1)} of them claimed by attempt 1; '
                                     'one run per seed is required (no replicates): ' + ', '.join(r['path'] for r in rr))
                for x in rr:
                    if x is not a1[0]:
                        x['ineligible_reason'] = f'second run of seed {s} (RULE 4: attempt 1 is used)'
                print(f'WARNING {cfg} seed {s}: {len(rr)} eligible runs, attempt 1 used', file=sys.stderr)
                rr = a1
            if rr:
                rr[0]['chosen'] = '1'
                chosen[('MSCA', LOSSES[loss], ds, s)] = rr[0]
    for (ds, loss), (sc, tag) in tau_star.items():
        fps = {r['fingerprint'] for k, r in chosen.items() if k[1:3] == (LOSSES[loss], ds)}
        if len(fps) > 1:
            raise SystemExit(f'{ds}_{loss}_{tag}{SCORES[sc]}: chosen runs carry several fingerprints {sorted(fps)}')
    rows.sort(key=lambda r: (r['row'], r['dataset'], r['score'], str(r['tau']), r['seed'] or 0, r['path']))
    cols = ['backbone', 'row', 'chosen', 'ineligible_reason', 'root', 'attempt', 'score', 'tau', 'tau_star'] + FIELDS
    with open(os.path.join(TABLES, 'topk_runs.tsv'), 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter='\t', lineterminator='\n', extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(TABLES, 'topk_perseed.csv'), 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh, lineterminator='\n')
        w.writerow(['backbone', 'row', 'dataset', 'seed', 'r20', 'n20', 'r10', 'n10', 'gpu', 'root', 'fingerprint'])
        for k in sorted(chosen):
            c = chosen[k]
            w.writerow(list(k) + [c['test_r20'], c['test_n20'], c['test_r10'], c['test_n10'], c['gpu'], c['root'],
                                  c['fingerprint']])
    missing = []
    with open(os.path.join(TABLES, 'topk_cells.csv'), 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh, lineterminator='\n')
        w.writerow(['backbone', 'row', 'dataset', 'n', 'r20_mean', 'r20_std', 'n20_mean', 'n20_std', 'seeds', 'source',
                    'gpus', 'roots', 'tau_star', 'score'])
        for loss, row in LOSSES.items():
            for ds in DATASETS:
                got = [chosen[('MSCA', row, ds, s)] for s in SEEDS if ('MSCA', row, ds, s) in chosen]
                missing += [f'{row}/{ds}/s{s}' for s in SEEDS if ('MSCA', row, ds, s) not in chosen]
                if not got:
                    continue
                sc, tag = tau_star[(ds, loss)]
                r20 = [float(x['test_r20']) for x in got]
                n20 = [float(x['test_n20']) for x in got]
                sd = (lambda v: st.stdev(v) if len(v) > 1 else None)
                w.writerow(['MSCA', row, ds, len(got), f6(st.mean(r20)), f6(sd(r20)), f6(st.mean(n20)), f6(sd(n20)),
                            '/'.join(str(x['seed']) for x in got), f'objx_s/{ds}_{loss}_{tag}{SCORES[sc]}',
                            '/'.join(sorted({x['gpu'] for x in got})), '/'.join(x['root'] for x in got),
                            GRID[loss][tag], sc])
    n_ok = sum(r['chosen'] == '1' for r in rows)
    print(f'{len(rows)} candidate runs, {n_ok} chosen, {len(tau_star)}/18 (score, tau) selected; missing (cell/seed): '
          f'{len(missing)}', file=sys.stderr)


if __name__ == '__main__':
    if len(sys.argv) >= 2 and sys.argv[1] == 'scan':
        scan(sys.argv[2:])
    else:
        main()
