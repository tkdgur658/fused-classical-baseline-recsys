"""STEP1 (EDIT 18-1, 2026-10-16): per-seed values of the classical fill C and of the learned models on targets with at most two
training edges. Read-only on existing result files; writes only <out.json>.
Sources
  A_DESIGN/results/<bb>__<ds>__s<seed>.json  (a_strat.py, code sha in each file): routes base, q2_ext, q2_fcombo with
      test_r20 (all test users) and groups.deg_le2 {n, hits, hit, rnd}; bb in native (MSCA), xre (E + R + A), mes (MES).
  TABLE_FUSION/cells/<tag>__<ds>__s<seed>.json (tf_fuse.py): route base groups.deg_le2 of the eight learned models of F.
Hit multiple = hit / rnd (hit rate over the hits expected from a random list of 20 among the eligible items).
usage: python -s -B c_fill_extract.py <out.json>
"""
import glob, json, os, sys, hashlib
S4R = '<RUN_ROOT>'
out = dict(code_sha256=hashlib.sha256(open(os.path.abspath(__file__), 'rb').read()).hexdigest(), cells={}, learned={})
for f in sorted(glob.glob(f'{S4R}/A_DESIGN/results/*__*__s*.json')):
    tag = os.path.basename(f)[:-5]
    bb, ds, s = tag.split('__'); s = int(s[1:])
    j = json.load(open(f))
    assert j['dataset'] == ds and int(j['seed']) == s
    rec = dict(run_dir=j['run_dir'], code_sha256=j['code_sha256'], gate_ok=bool(j['gate']['gate_ok'] or j['gate']['tol_ok']))
    for rt in ('base', 'q2_ext', 'q2_fcombo'):
        r = j['routes'][rt]; g = r['groups']['deg_le2']
        rec[rt] = dict(test_r20=r['test_r20'], valid_r20=r['valid_r20'], le2_n=g['n'], le2_hits=g['hits'],
                       le2_x=(g['hit'] / g['rnd']) if g['n'] else None)
    out['cells'].setdefault(f'{bb}__{ds}', {})[str(s)] = rec
for f in sorted(glob.glob(f'{S4R}/TABLE_FUSION/cells/*__*__s*.json')):
    tag = os.path.basename(f)[:-5]
    m, ds, s = tag.split('__'); s = int(s[1:])
    j = json.load(open(f))
    g = j['routes']['base']['groups']['deg_le2']
    ok = bool(j['gate']['tol_ok'] and j['gate']['tol_ok_n'])
    out['learned'].setdefault(f'{m}__{ds}', {})[str(s)] = dict(gate_ok=ok, le2_n=g['n'], le2_hits=g['hits'],
                                                            le2_x=(g['hit'] / g['rnd']) if g['n'] else None)
json.dump(out, open(sys.argv[1], 'w'), indent=1)
print('cells', len(out['cells']), 'learned', len(out['learned']))
