"""Collect the novelty-control experiments of the TMM revision (internal review 2026-10-07, items (2)-(5)) and their
paired contrasts with XRE. Only reads remote files; trains nothing; edits no .tex file.

Runs (HOST_A = HOST_C): <RUN_ROOT_B>/P04_novctrl/runs_nc/{xres,objx_s}/*/result.json, repo
<HOME>/P04_novctrl_wt branch novctrl (ca3d98e + unit-test fix ae04ee0; = e2128b2 fast-forwarded to topk
108a1a4, flags off bitwise == e2128b2 XRE and 108a1a4 top-K / Native, tools/novctrl_unit_test.py PASS), MSCA, lambda 0.1,
K 40, J 10, M 50, seeds 1-5, one run per seed (no replicates). Data: Baby canonical, the others canonical_v2 (as Table III).
README: <RUN_ROOT_B>/P04_novctrl/README.txt; status: <TOOLS>/ops/NOVCTRL_STATUS.md.
Rows (config <ds>_xres_<arm>_l010_k40 or objx_s <ds>_add<loss>_<tau>_dot_mu<MMM>):
  xmd       (2) matched competitor selection: XRE with D_u and C_u replaced by the highest masked-view score of 50 uniform
                draws from I minus H_u (entry activation, retention pairs, pair counts, lambda, anchor term, masked view = XRE)
  xeo / xto (3) entry-only / retention-only XRE (anchor term kept, the other exchange part exactly zero, no rescaling)
  xph       (5b) XRE with anchor probes uniform on I minus H_u
  add_ssm / add_psl / add_slk  (4) BPR + mu * top-K loss (Table I temperature and dot score of that loss); mu selected per
                (dataset, loss) on seed-1 valid Recall@20 over {0.1, 0.3, 1.0} (P04_novctrl/mu_selection.json, recomputed and
                checked here); grid rows add_<loss>_mu<MMM> (seed 1 only) are written with role 'grid'.
  (5a) anchor-collision rates (training-free, P04_novctrl/probe5a/probe5a_perseed.csv) -> tables/novctrl_probe5a.csv.
References (same seeds, tables/ CSVs of the paper): XRE = MSCA row XRE of xre_perseed.csv; mdns = 'A + masked DNS' of
xmask_perseed.csv; wo_retention = MSCA row wo_retention of xre_perseed.csv (entry only,
mass-rescaled); SSM / PSL / SL@K replacement rows of topk_perseed.csv; Native of table1_perseed.csv.
Eligible run: status SUCCESS, mode train, test_eval best_primed; a second eligible run of one (config, seed) stops the script.
Metrics: test Recall@20 / NDCG@20 at the best-validation checkpoint (best_test), valid = best_valid_primed.
Outputs
  tables/novctrl_perseed.csv    row, role (new | grid | ref), dataset, seed, test_r20, test_n20, valid_r20, valid_n20,
                                best_epoch, mu, config, gpu, run_dir
  tables/novctrl_contrasts.csv  family, contrast (a - b), dataset, metric (R@20 | N@20), n, mean_a, mean_b, diff,
                                ci_lo, ci_hi (95% paired t-interval, df n-1), t, p (two-sided paired t), p_holm, holm_reject
                                (alpha 0.05), wins_a (seeds with a > b). Holm runs within one family and one metric over all
                                its (contrast, dataset) cells that have seeds 1-5 on both sides.
     families: matched    XRE - xmd                        (2)
               matched_ref xmd - mdns (A + masked DNS of Table III, unmatched)  (2, descriptive)
               parts      XRE - xeo, XRE - xto             (3)
               parts_ref  xeo - wo_retention (mass-rescaled entry-only of the supplement)  (3, descriptive)
               additive   XRE - add_<loss>                 (4)
               add_vs_rep add_<loss> - <loss> replacement   (4)
               probe      XRE - xph                        (5b)
  tables/novctrl_probe5a.csv    (5a) per dataset and seed, plus the seed mean rows (seed 'mean')
Usage (repository root):  python -s tables/src/collect_novctrl.py            (ssh to HOST_A)
                          python -s tables/src/collect_novctrl.py --no-collect (contrasts from the CSV already in tables/)
"""
import csv
import io
import json
import math
import statistics as st
import subprocess
import sys
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / 'tables'
OUT = TABLES / 'novctrl_perseed.csv'
CON = TABLES / 'novctrl_contrasts.csv'
P5A = TABLES / 'novctrl_probe5a.csv'
W = '<RUN_ROOT_B>/P04_novctrl'
DS = ['baby', 'pantry', 'scientific', 'sports']
SEEDS = [1, 2, 3, 4, 5]
ARMS = {'xmd': 'xmd', 'xeo': 'xeo', 'xto': 'xto', 'xph': 'xph'}
LOSSES = {'ssm': 'SSM', 'psl': 'PSL', 'slk': 'SL@K'}
MUS = {'mu010': 0.1, 'mu030': 0.3, 'mu100': 1.0}
ALPHA = 0.05
SCAN = (f"for f in $(find {W}/runs_nc -name result.json 2>/dev/null); do "
        "python3 -c \"import json,sys,os; j=json.load(open(sys.argv[1])); b=j.get('best_test') or {}; v=j.get('best_valid_primed') or {}; "
        "print(json.dumps(dict(cfg=j['config'],pkg=j['package'],ds=j['dataset'],seed=j['seed'],status=j.get('status'),"
        "mode=j.get('mode'),te=j.get('test_eval'),r20=b.get('recall@20'),n20=b.get('ndcg@20'),vr20=v.get('recall@20'),"
        "vn20=v.get('ndcg@20'),ep=j.get('best_epoch'),gpu=j.get('gpu'),run=os.path.dirname(sys.argv[1]),data=j.get('data_root'))))\" $f; done; "
        f"echo '#SEL'; cat {W}/mu_selection.json 2>/dev/null; echo; echo '#P5A'; cat {W}/probe5a/probe5a_perseed.csv")


def collect():
    out = subprocess.run(['ssh', '-o', 'BatchMode=yes', 'HOST_A', SCAN], capture_output=True, text=True, check=True).stdout
    runs_txt, rest = out.split('#SEL', 1)
    sel_txt, p5a_txt = rest.split('#P5A', 1)
    sel = json.loads(sel_txt.strip()) if sel_txt.strip() else {}
    rows, seen = [], set()
    for line in runs_txt.splitlines():
        if not line.strip():
            continue
        j = json.loads(line)
        if not (j['status'] == 'SUCCESS' and j['mode'] == 'train' and j['te'] == 'best_primed'):
            print('ineligible', j['run'], j['status'], file=sys.stderr)
            continue
        ds = j['ds'].split('@')[0]
        want = 'canonical' if ds == 'baby' else 'canonical_v2'
        assert str(j['data']).rstrip('/').endswith('/' + want), (j['run'], j['data'])
        key = (j['cfg'], ds, int(j['seed']))
        assert key not in seen, f'second eligible run of {key} (no replicates)'
        seen.add(key)
        cfg = j['cfg']
        if j['pkg'] == 'xres':
            arm = cfg.split('_xres_')[1].split('_')[0]
            row, role, mu = ARMS[arm], 'new', ''
        else:
            loss = cfg.split('_')[1][3:]
            tag = cfg.rsplit('_', 1)[1]
            mu = MUS[tag]
            chosen = sel.get(f'{ds}/{loss}', {}).get('config')
            if chosen == cfg:
                row, role = f'add_{loss}', 'new'
            else:
                row, role = f'add_{loss}_{tag}', 'grid'
                assert int(j['seed']) == 1, f'seed > 1 of an unselected mu: {cfg}'
            if role == 'new' and int(j['seed']) == 1:      # the selected seed-1 run is also a grid point
                rows.append(dict(row=f'add_{loss}_{tag}', role='grid', dataset=ds, seed=1, test_r20=j['r20'],
                                 test_n20=j['n20'], valid_r20=j['vr20'], valid_n20=j['vn20'], best_epoch=j['ep'], mu=mu,
                                 config=cfg, gpu=(j['gpu'] or '').replace('NVIDIA ', ''), run_dir=j['run']))
        rows.append(dict(row=row, role=role, dataset=ds, seed=int(j['seed']), test_r20=j['r20'], test_n20=j['n20'],
                         valid_r20=j['vr20'], valid_n20=j['vn20'], best_epoch=j['ep'], mu=mu, config=cfg,
                         gpu=(j['gpu'] or '').replace('NVIDIA ', ''), run_dir=j['run']))
    # recompute the mu selection from the grid rows (valid metrics only) and check it against mu_selection.json
    for key, s in sel.items():
        ds, loss = key.split('/')
        g = [r for r in rows if r['role'] == 'grid' and r['dataset'] == ds and r['row'].startswith(f'add_{loss}_mu')]
        assert len(g) == 3, (key, len(g))
        best = sorted(g, key=lambda r: (-r['valid_r20'], -r['valid_n20'], r['mu']))[0]
        assert best['config'] == s['config'], (key, best['config'], s['config'])
    # references from the paper CSVs
    for r in csv.DictReader(open(TABLES / 'xre_perseed.csv', encoding='utf-8')):
        if r['backbone'] == 'MSCA' and r['row'] in ('XRE', 'wo_retention') and r['dataset'] in DS:
            rows.append(dict(row=r['row'], role='ref', dataset=r['dataset'], seed=int(r['seed']), test_r20=float(r['r20']),
                             test_n20=float(r['n20']), valid_r20='', valid_n20='', best_epoch='', mu='', config='xre_perseed.csv',
                             gpu='', run_dir=''))
    for r in csv.DictReader(open(TABLES / 'xmask_perseed.csv', encoding='utf-8')):
        if r['row'] == 'mdns' and r['dataset'] in DS:
            rows.append(dict(row='mdns', role='ref', dataset=r['dataset'], seed=int(r['seed']), test_r20=float(r['r20']),
                             test_n20=float(r['n20']), valid_r20='', valid_n20='', best_epoch=r['best_epoch'], mu='',
                             config=r['config'], gpu=r['gpu'], run_dir=r['run_dir']))
    for r in csv.DictReader(open(TABLES / 'topk_perseed.csv', encoding='utf-8')):
        if r['backbone'] == 'MSCA' and r['dataset'] in DS:
            rows.append(dict(row='rep_' + {v: k for k, v in LOSSES.items()}[r['row']], role='ref', dataset=r['dataset'],
                             seed=int(r['seed']), test_r20=float(r['r20']), test_n20=float(r['n20']), valid_r20='', valid_n20='',
                             best_epoch='', mu='', config='topk_perseed.csv', gpu=r['gpu'], run_dir=''))
    for r in csv.DictReader(open(TABLES / 'table1_perseed.csv', encoding='utf-8')):
        if r['backbone'] == 'MSCA' and r['row'] == 'Native' and r['dataset'] in DS:
            rows.append(dict(row='Native', role='ref', dataset=r['dataset'], seed=int(r['seed']), test_r20=float(r['r20']),
                             test_n20=float(r['n20']), valid_r20='', valid_n20='', best_epoch='', mu='',
                             config='table1_perseed.csv', gpu=r['gpu'], run_dir=''))
    rows.sort(key=lambda r: (r['role'], r['row'], DS.index(r['dataset']), r['seed']))
    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    print('wrote', OUT, len(rows), 'rows')
    p5 = list(csv.DictReader(io.StringIO(p5a_txt.strip())))
    keys = ['hist', 'self', 'nbr', 'hist_nbr', 'eqj', 'eqj_w', 'probe_hist']
    out5 = []
    for d in DS:
        R = sorted([r for r in p5 if r['dataset'] == d], key=lambda r: int(r['seed']))
        for r in R:
            out5.append(dict(dataset=d, seed=r['seed'], n_pairs=r['n_pairs'], **{k: float(r[k]) for k in keys}))
        if R:
            out5.append(dict(dataset=d, seed='mean', n_pairs=len(R), **{k: st.mean(float(r[k]) for r in R) for k in keys}))
    if out5:
        with open(P5A, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(out5[0]), lineterminator='\n')
            w.writeheader()
            w.writerows(out5)
        print('wrote', P5A, len(out5), 'rows')


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - k) * ps[i]))
        adj[i] = run
    return adj


def contrasts():
    V = {}
    for r in csv.DictReader(open(OUT, encoding='utf-8')):
        if r['role'] in ('new', 'ref'):
            V[(r['row'], r['dataset'], int(r['seed']))] = (float(r['test_r20']), float(r['test_n20']))
    fam = {'matched': [('XRE', 'xmd')], 'matched_ref': [('xmd', 'mdns')],
           'parts': [('XRE', 'xeo'), ('XRE', 'xto')], 'parts_ref': [('xeo', 'wo_retention')],
           'additive': [('XRE', f'add_{l}') for l in LOSSES], 'add_vs_rep': [(f'add_{l}', f'rep_{l}') for l in LOSSES],
           'probe': [('XRE', 'xph')]}
    out = []
    tq = stats.t.ppf(0.975, len(SEEDS) - 1)
    for f, pairs in fam.items():
        for mi, mname in ((0, 'R@20'), (1, 'N@20')):
            cells = []
            for a, b in pairs:
                for d in DS:
                    if not all((a, d, s) in V and (b, d, s) in V for s in SEEDS):
                        continue
                    x = [V[(a, d, s)][mi] for s in SEEDS]
                    y = [V[(b, d, s)][mi] for s in SEEDS]
                    dd = [p - q for p, q in zip(x, y)]
                    m, sd = st.mean(dd), st.stdev(dd)
                    se = sd / math.sqrt(len(dd))
                    t = m / se if se > 0 else (math.inf if m else 0.0)
                    p = 2 * stats.t.sf(abs(t), len(dd) - 1) if se > 0 else (0.0 if m else 1.0)
                    cells.append(dict(family=f, contrast=f'{a} - {b}', dataset=d, metric=mname, n=len(dd),
                                      mean_a=round(st.mean(x), 6), mean_b=round(st.mean(y), 6), diff=round(m, 6),
                                      ci_lo=round(m - tq * se, 6), ci_hi=round(m + tq * se, 6), t=round(t, 4), p=p,
                                      wins_a=sum(1 for p_, q_ in zip(x, y) if p_ > q_)))
            if cells:
                adj = holm([c['p'] for c in cells])
                for c, pa in zip(cells, adj):
                    c['p'] = round(c['p'], 6)
                    c['p_holm'] = round(pa, 6)
                    c['holm_reject'] = int(pa <= ALPHA)
                out += cells
    if not out:
        print('no complete contrast yet')
        return
    with open(CON, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(out)
    print('wrote', CON, len(out), 'contrast cells')
    for c in out:
        if c['metric'] == 'R@20':
            print(f"{c['family']:11s} {c['contrast']:22s} {c['dataset']:10s} {c['mean_a']:.4f} vs {c['mean_b']:.4f} "
                  f"diff {c['diff']:+.4f} [{c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}] p={c['p']:.3f} holm={c['p_holm']:.3f} "
                  f"wins {c['wins_a']}/5")


if __name__ == '__main__':
    if '--no-collect' not in sys.argv:
        collect()
    contrasts()
