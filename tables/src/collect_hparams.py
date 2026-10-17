"""Collect the configuration YAMLs behind every reported run for the supplementary hyperparameter table. Only reads files.

Usage (local machine, repository root):  python tables/src/collect_hparams.py
Runs read (the chosen runs of the manuscript tables):
  tables/table1_runs.tsv        chosen = 1 (Native and plug-in rows on MSCA, FREEDOM, MMGCN)
  tables/topk_runs.tsv          chosen = 1 (top-K loss rows SL@K, PSL, SSM on MSCA)
  tables/xre_runs.tsv           row XRE without ineligible_reason (setting collect_xre.XRE_SETTING)
  tables/table2_runs.tsv        chosen = 1 (compared recommenders on Baby, Pantry, Scientific; HOST_D runs)
  tables/table2_runs_extra.tsv  chosen = 1 (compared recommenders on Instruments, Clothing, Sports; HOST_A runs)
The script sends itself over ssh to HOST_A (argument `remote`) with the list of (package, config, worktree family, result.json
paths). For every (package, config) the remote side reads `config_files` of the result.json of every chosen run that is on
HOST_A and takes the model YAML (last entry). Rule:
  recorded   at least one recorded model YAML exists on HOST_A -> that file. Every existing recorded copy must parse to the
             same mapping (assert).
  worktree   no recorded model YAML exists on HOST_A (runs of the rented pods record <POD_ROOT>/... paths; the HOST_D runs
             of tables/table2_runs.tsv are not on HOST_A) -> the copy of that YAML in the worktree of its package family:
               objx_s, plg_s, plg2_s, plg_f, plg2_f, plg2_m -> <HOME>/P04_table1fix_wt  (branch table1fix)
               xres                                         -> <HOME>/P04_xres_wt       (branch xre_s2)
               xret_n, xret_f, xret_m                       -> <HOME>/P04_xret_wt       (branch xre_t2)
               compared recommenders                        -> <HOME>/P04_t2b2048L_wt   (branch t2-b2048L)
             (the worktrees whose configs the run hosts copied).
Outputs
  tables/hparams_configs.tsv   one row per (package, config, key): parsed YAML value, worktree (root of the YAML read), commit
                               of that worktree, YAML sha256, rule (recorded / worktree), YAML path, and whether the
                               worktree-family copy parses to the same mapping (worktree_copy: equal / differs / absent / -)
  tables/hparams_sources.tsv   one row per (config, key) of the author-configuration tables (rule, source location)
Common conditions: configs/overall.yaml of P04_table1fix_wt and P04_t2b2048L_wt (rows package _overall).
Author-configuration source tables: P04_t2b2048L_wt results/author_cfg/CONFIG_TABLE.csv and CONFIG_TABLE_A6.csv.
"""
import base64
import csv
import hashlib
import json
import os
import subprocess
import sys
import zlib

WT = {'table1fix': '<HOME>/P04_table1fix_wt', 'xres': '<HOME>/P04_xres_wt',
      'xret': '<HOME>/P04_xret_wt', 't2': '<HOME>/P04_t2b2048L_wt',
      'topk': '<HOME>/P04_topk_wt'}
PKG_WT = {'objx_s': 'table1fix', 'plg_s': 'table1fix', 'plg2_s': 'table1fix', 'plg_f': 'table1fix',
          'plg2_f': 'table1fix', 'plg2_m': 'table1fix', 'xres': 'xres', 'xret_n': 'xret', 'xret_f': 'xret',
          'xret_m': 'xret'}
T2_PKG = {'MSCA': 'msca', 'COHESION': 'cohesion', 'SMORE': 'smore', 'PGL': 'pgl', 'MENTOR': 'mentor',
          'DiffMM': 'diffmm', 'LGMRec': 'lgmrec', 'FREEDOM': 'freedom'}
REMOTE_PY = 'python'


def remote(arg):
    import yaml
    todo = json.loads(zlib.decompress(base64.b64decode(arg)).decode())
    w = csv.writer(sys.stdout, delimiter='\t', lineterminator='\n')
    w.writerow(['table', 'package', 'config', 'worktree', 'commit', 'yaml_sha256', 'key', 'value', 'rule', 'yaml_path',
                'worktree_copy'])
    commits = {}

    def commit(root):
        if root not in commits:
            r = subprocess.run(['git', '-C', root, 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True)
            commits[root] = r.stdout.strip() if r.returncode == 0 else ''
        return commits[root]

    for pkg, cfg, wt, paths in todo:
        recorded = set()
        for p in paths:
            if p.startswith('/mnt/') and os.path.exists(p):
                cf = json.load(open(p)).get('config_files') or []
                assert cf and cf[-1].endswith(f'/models/{pkg}/configs/{cfg}.yaml'), (p, cf)
                recorded.add(cf[-1])
        existing = sorted(f for f in recorded if os.path.exists(f))
        wt_path = f'{WT[wt]}/models/{pkg}/configs/{cfg}.yaml'
        wt_d = yaml.safe_load(open(wt_path, 'rb').read()) if os.path.exists(wt_path) else None
        if existing:
            parsed = [yaml.safe_load(open(f, 'rb').read()) for f in existing]
            assert all(d == parsed[0] for d in parsed), (pkg, cfg, existing)
            path, rule = existing[0], 'recorded'
            copy = 'absent' if wt_d is None else ('equal' if wt_d == parsed[0] else 'differs')
        else:
            assert wt_d is not None, (pkg, cfg, wt_path, sorted(recorded))
            path, rule, copy = wt_path, 'worktree', '-'
        raw = open(path, 'rb').read()
        root = path.split('/models/')[0]
        for k, v in yaml.safe_load(raw).items():
            w.writerow(['config', pkg, cfg, root, commit(root), hashlib.sha256(raw).hexdigest()[:12], k, json.dumps(v),
                        rule, path, copy])
    # common conditions: configs/overall.yaml of the plug-in worktree and of the compared-recommender worktree
    for wt in ('table1fix', 't2'):
        root = WT[wt]
        raw = open(f'{root}/configs/overall.yaml', 'rb').read()
        for k, v in yaml.safe_load(raw).items():
            w.writerow(['config', '_overall', wt, root, commit(root), hashlib.sha256(raw).hexdigest()[:12], k,
                        json.dumps(v), 'worktree', f'{root}/configs/overall.yaml', '-'])
    for fn in ('CONFIG_TABLE.csv', 'CONFIG_TABLE_A6.csv'):
        with open(f"{WT['t2']}/results/author_cfg/{fn}", newline='') as f:
            for r in csv.DictReader(f):
                w.writerow(['source', r['package'], r['config'], fn, r['rule'], r['source'].replace('\t', ' '),
                            r['key'], json.dumps(r['value'])])


def read_tsv(fn):
    with open(fn, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f, delimiter='\t'))


def main():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import collect_xre
    todo = {}  # (package, config) -> (worktree family, set of result.json paths)

    def add(pkg, cfg, wt, path):
        old = todo.setdefault((pkg, cfg), (wt, set()))
        assert old[0] == wt, (pkg, cfg, old[0], wt)
        if path:
            old[1].add(path)

    for r in read_tsv('tables/table1_runs.tsv'):
        if r['chosen'] == '1':
            add(r['package'], r['config'], PKG_WT[r['package']], r['path'])
    for r in read_tsv('tables/topk_runs.tsv'):
        if r['chosen'] == '1':
            add(r['package'], r['config'], 'topk', r['path'])
    for r in read_tsv('tables/xre_runs.tsv'):
        if not r['ineligible_reason'] and r['row'] == 'XRE':
            assert collect_xre.XRE_SETTING in r['config'], r['config']
            add(r['package'], r['config'], PKG_WT[r['package']], r['path'])
    for r in read_tsv('tables/table2_runs.tsv'):
        if r['chosen'] == '1':
            add(T2_PKG[r['model']], r['config'], 't2', r['origin'])
    for r in read_tsv('tables/table2_runs_extra.tsv'):
        if r['chosen'] == '1':
            add(T2_PKG[r['model']], r['config'], 't2', r['origin'])
    items = sorted([pkg, cfg, wt, sorted(paths)] for (pkg, cfg), (wt, paths) in todo.items())
    arg = base64.b64encode(zlib.compress(json.dumps(items).encode(), 9)).decode()
    src = open(os.path.abspath(__file__), 'rb').read()
    out = subprocess.run(['ssh', '-o', 'BatchMode=yes', 'HOST_A', REMOTE_PY, '-', 'remote', arg], input=src,
                         capture_output=True, check=True).stdout.decode()
    rows = list(csv.reader(out.splitlines(), delimiter='\t'))
    hdr = rows[0]
    rules = {}
    with open('tables/hparams_configs.tsv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f, delimiter='\t', lineterminator='\n')
        w.writerow(hdr[1:])
        n = 0
        for r in rows[1:]:
            if r[0] == 'config':
                w.writerow(r[1:])
                n += 1
                if r[1] != '_overall':
                    rules[(r[1], r[2])] = (r[8], r[10])
    with open('tables/hparams_sources.tsv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f, delimiter='\t', lineterminator='\n')
        w.writerow(['package', 'config', 'table', 'rule', 'source', 'key', 'value'])
        m = 0
        for r in rows[1:]:
            if r[0] == 'source':
                w.writerow(r[1:])
                m += 1
    assert len(rules) == len(items), (len(rules), len(items))
    n_rec = sum(v[0] == 'recorded' for v in rules.values())
    diff = sorted(k for k, v in rules.items() if v[1] == 'differs')
    print(f'{len(items)} configs ({n_rec} recorded YAML, {len(items) - n_rec} worktree copy), {n} config keys, '
          f'{m} source rows; XRE setting {collect_xre.XRE_SETTING}; recorded YAML differing from the worktree copy: '
          f'{diff or "none"}')


if __name__ == '__main__':
    if len(sys.argv) > 2 and sys.argv[1] == 'remote':
        remote(sys.argv[2])
    else:
        main()
