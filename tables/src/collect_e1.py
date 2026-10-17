"""Collect the full-view DNS control (reviewer control E1) and write its paired macros.

Claim under test: masking matters with a sampled high-scoring partner, i.e. the masked view of Table III row
"A + masked DNS" (xmask <ds>_xmask_mdns) does better than the same term whose DNS partner is picked by the FULL-view score
and whose loss is computed on the full view. Control "A + full-view DNS": the identical anchor term A1 (same 50 global-RNG
probes, lambda 0.1), the same 2-fold hash(u,i)&1 held rows, the same 50 uniform non-positive candidates (same private
generator draw), one term per held row, sum / B; the partner is the candidate with the highest full-view score and the
term softplus(s_F(u,j*) - s_F(u,p)) is scaled in every batch to the coefficient mass of the masked DNS term on the same
candidates (as the full-view rows xrf / xfo of Table III), so the gradient flows into the full-view representation only.
Runs: HOST_A <RUN_ROOT_B>/P04_e1e2/runs_e1/xmask/*/result.json, config <ds>_xmask_mdnsfv (repo
<HOME>/P04_xmask_fv_wt, branch xmask-fvdns 17126c3 = xmask 7951ebc + config key xmask_view: full; unit test
PASS: flag off is bitwise the 7951ebc code; README.txt and collect_e1e2.py in the work dir;
<TOOLS>/hyp_figs_20261007/E1E2_RUNS.md), MSCA, Baby / Pantry / Scientific, seeds 1-5, data tokens of the
Table III mdns runs. Sports (added 2026-10-07): HOST_B runs synced to HOST_A
<RUN_ROOT_B>/P04_sports_ctrl_HOST_B/runs_ctrl/xmask (branch xmask-fvdns-sports c2f340a = 17126c3 on the HOST_B Sports
mdns code f2947de; <TOOLS>/hyp_figs_20261007/E1E2_RUNS.md 'Sports 대조 2행'), reference = the Sports mdns runs of the same
machine (<RUN_ROOT_B>/P04_sports_abl_HOST_B/runs_abl/xmask). The Sports XRE reference ran on another machine, so
eXre... of Sports mixes machines (not quoted in the prose). Eligible run: status SUCCESS, mode train, test_eval best_primed. One run per (dataset, seed): a second
eligible run is an error (no replicates), the script stops.
References (same seed): row mdns of tables/xmask_perseed.csv (collect_xmask.py) and MSCA row XRE of
tables/xre_perseed.csv (collect_xre.py). The mdns reference runs on HOST_A (<RUN_ROOT_B>/P04_xmask/runs_xmask/xmask)
are read as well and must equal xmask_perseed.csv digit for digit (cross-check, the script stops otherwise).
Outputs
  tables/e1_perseed.csv   row mdnsfv, dataset, seed, r20, n20, best_epoch, gpu, config, run_dir (test metrics of the
                          best-validation checkpoint, best_test of result.json), plus ref_r20 / ref_n20 (row mdns) and
                          xre_r20 / xre_n20 (row XRE).
  tables/macros_e1.tex    paired differences (same seed, seeds 1-5), macro names e..., no digits; D in Baby, Pantry,
                          Scientific, Sports; X in R (Recall@20), N (NDCG@20):
    eMean<X>Full<D>      mean of the full-view DNS row (4 dp, Table III format, leading zero dropped)
    eDiff<X><D>          (A + masked DNS) - (A + full-view DNS), mean (signed, 4 dp, the format of macros_hyp_forest.tex)
    eCI<X><D>            95% paired t-interval (df 4);  ePos<X><D> seeds with masked > full (k/5)
    eP<X><D>             two-sided paired t-test p (3 dp, '$<$0.001' below);  eT<X><D> paired t (2 dp)
    eSig<X> / eNSig<X> / eSig<X>Holm   datasets with p < 0.05 (phrase / word / after Holm over the datasets)
    eSigUp<X> / eSigDown<X>            significant datasets with masked above / below full view
    eXreDiff<X><D>, eXreCI<X><D>, eXreP<X><D>, eXrePos<X><D>   XRE - (A + full-view DNS), same format
    eNDs, eDsPhrase
Usage (repository root): python -s tables/src/collect_e1.py               (ssh to HOST_A, then macros)
                         python -s tables/src/collect_e1.py --no-collect  (macros from the CSV already in tables/)
Then: python -s tables/src/build_table_xmask.py, python -s tables/src/collect_hyp_forest.py, figures/src/fig_h_forest.py.
"""
import csv
import json
import math
import re
import statistics as st
import subprocess
import sys
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / 'tables'
OUT = TABLES / 'e1_perseed.csv'
MAC = TABLES / 'macros_e1.tex'
REMOTE = {d: '<RUN_ROOT_B>/P04_e1e2/runs_e1/xmask' for d in ('baby', 'pantry', 'scientific')}
REF_REMOTE = {d: '<RUN_ROOT_B>/P04_xmask/runs_xmask/xmask' for d in ('baby', 'pantry', 'scientific')}
REMOTE['sports'] = '<RUN_ROOT_B>/P04_sports_ctrl_HOST_B/runs_ctrl/xmask'
REF_REMOTE['sports'] = '<RUN_ROOT_B>/P04_sports_abl_HOST_B/runs_abl/xmask'
DS = [('baby', 'Baby'), ('pantry', 'Pantry'), ('scientific', 'Scientific'), ('sports', 'Sports')]
SEEDS = [1, 2, 3, 4, 5]
ALPHA = 0.05
WORDS = 'none one two three four'.split()
PY = ("import json,sys,os\n"
      "for f in sys.argv[1:]:\n"
      "  j=json.load(open(f)); b=j.get('best_test') or {}\n"
      "  print(json.dumps(dict(cfg=j.get('config'),ds=str(j.get('dataset')).split('@')[0],seed=j.get('seed'),"
      "status=j.get('status'),mode=j.get('mode'),te=j.get('test_eval'),r20=b.get('recall@20'),n20=b.get('ndcg@20'),"
      "ep=j.get('best_epoch'),gpu=(j.get('gpu') or '').replace('NVIDIA ',''),run=os.path.dirname(f))))\n")


def fetch(pattern):
    """result.json of every run dir matching pattern on HOST_A, eligible runs only, keyed by (dataset, seed)."""
    cmd = f"python3 -c \"$(cat)\" $(ls {pattern}/result.json 2>/dev/null)"
    out = subprocess.run(['ssh', '-o', 'BatchMode=yes', 'HOST_A', cmd], input=PY, capture_output=True, text=True,
                         check=True).stdout
    got = {}
    for line in out.splitlines():
        j = json.loads(line)
        if not (j['status'] == 'SUCCESS' and j['mode'] == 'train' and j['te'] == 'best_primed'):
            print('skip (not eligible):', j['run'], j['status'], j['mode'], j['te'])
            continue
        key = (j['ds'], int(j['seed']))
        assert key not in got, f'two eligible runs for {key} (no replicates): {got[key]["run"]} {j["run"]}'
        got[key] = j
    return got


def collect():
    ref_csv = {(r['dataset'], int(r['seed'])): (float(r['r20']), float(r['n20']))
               for r in csv.DictReader(open(TABLES / 'xmask_perseed.csv', encoding='utf-8')) if r['row'] == 'mdns'}
    xre_csv = {(r['dataset'], int(r['seed'])): (float(r['r20']), float(r['n20']))
               for r in csv.DictReader(open(TABLES / 'xre_perseed.csv', encoding='utf-8'))
               if r['backbone'] == 'MSCA' and r['row'] == 'XRE'}
    rows, bad = [], []
    for d, _ in DS:
        e1 = fetch(f'{REMOTE[d]}/xmask__{d}_xmask_mdnsfv__{d}*__s*__train__*')
        ref = fetch(f'{REF_REMOTE[d]}/xmask__{d}_xmask_mdns__{d}*__s*__train__*')
        for s in SEEDS:
            assert (d, s) in e1, f'missing full-view DNS run {d} s{s}'
            assert (d, s) in ref and (d, s) in ref_csv and (d, s) in xre_csv, f'missing reference run {d} s{s}'
            j, a = e1[(d, s)], ref[(d, s)]
            if (round(a['r20'], 6), round(a['n20'], 6)) != tuple(round(x, 6) for x in ref_csv[(d, s)]):
                bad.append(f'reference {d} s{s}: HOST_A {a["r20"]}/{a["n20"]} vs xmask_perseed.csv {ref_csv[(d, s)]}')
            assert j['cfg'] == f'{d}_xmask_mdnsfv', j
            if j['gpu'] != a['gpu']:   # device check (non-fatal): both runs of a pair should share one device
                print(f'WARNING {d} s{s}: the two runs of the pair report different devices', file=sys.stderr)
            rows.append(dict(row='mdnsfv', dataset=d, seed=s, r20=f"{j['r20']:.6f}", n20=f"{j['n20']:.6f}",
                             best_epoch=j['ep'], gpu=j['gpu'], config=j['cfg'], run_dir=j['run'],
                             ref_r20=f'{ref_csv[(d, s)][0]:.6f}', ref_n20=f'{ref_csv[(d, s)][1]:.6f}',
                             xre_r20=f'{xre_csv[(d, s)][0]:.6f}', xre_n20=f'{xre_csv[(d, s)][1]:.6f}'))
        extra = sorted(set(e1) - {(d, s) for s in SEEDS})
        assert not extra, f'unexpected full-view DNS runs {extra}'
    if bad:
        print('MISMATCH:', *bad, sep='\n  ')
        sys.exit(1)
    assert len({(r['dataset'], r['seed']) for r in rows}) == len(rows) == 5 * len(DS)
    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    print('wrote', OUT, len(rows), 'rows; reference values equal xmask_perseed.csv (row mdns)')


def fmt(v):
    return f'{v:+.4f}'.replace('-', '$-$')


def phrase(names):
    if not names:
        return 'none of the datasets'
    return names[0] if len(names) == 1 else ', '.join(names[:-1]) + (' and ' if len(names) == 2 else ', and ') + names[-1]


def holm(p):
    order = sorted(range(len(p)), key=lambda i: p[i])
    rej = set()
    for k, i in enumerate(order):
        if p[i] <= ALPHA / (len(p) - k):
            rej.add(i)
        else:
            break
    return rej


def paired(a, b):
    dd = [x - y for x, y in zip(a, b)]
    mu, sd = st.mean(dd), st.stdev(dd)
    h = stats.t.ppf(0.975, len(dd) - 1) * sd / math.sqrt(len(dd))
    tt = stats.ttest_rel(a, b)
    return dict(mean=mu, lo=mu - h, hi=mu + h, pos=sum(x > 0 for x in dd), p=float(tt.pvalue), t=float(tt.statistic),
                a=st.mean(a), b=st.mean(b), d=dd)


def macros():
    V = {}
    for r in csv.DictReader(open(OUT, encoding='utf-8')):
        key = (r['dataset'], int(r['seed']))
        assert key not in V, f'duplicate {key}'
        V[key] = r
    out = ['% Generated by tables/src/collect_e1.py from tables/e1_perseed.csv (A + full-view DNS, reviewer control E1),',
           '% the mdns rows of tables/xmask_perseed.csv and the MSCA XRE rows of tables/xre_perseed.csv. Do not edit by hand.',
           '% Diff = (A + masked DNS) - (A + full-view DNS), XreDiff = XRE - (A + full-view DNS), same seed, seeds 1-5, one',
           '% run per seed; CI = 95% paired t-interval (df 4); P = two-sided paired t-test; Pos = seeds with a positive difference.']

    def mac(n, v):
        assert re.fullmatch(r'[A-Za-z]+', n), n
        out.append(f'\\newcommand{{\\{n}}}{{{v}}}')
    res = {}
    for X, col in (('R', 'r20'), ('N', 'n20')):
        for d, D in DS:
            full = [float(V[(d, s)][col]) for s in SEEDS]
            masked = [float(V[(d, s)][f'ref_{col}']) for s in SEEDS]
            xre = [float(V[(d, s)][f'xre_{col}']) for s in SEEDS]
            s = res[(X, d)] = paired(masked, full)
            x = res[('xre', X, d)] = paired(xre, full)
            mac(f'eMean{X}Full{D}', f"{s['b']:.4f}".replace('0.', '.', 1))
            mac(f'eDiff{X}{D}', fmt(s['mean']))
            mac(f'eCI{X}{D}', f"[{fmt(s['lo'])}, {fmt(s['hi'])}]")
            mac(f'ePos{X}{D}', f"{s['pos']}/5")
            mac(f'eP{X}{D}', f"{s['p']:.3f}" if s['p'] >= 0.001 else '$<$0.001')
            mac(f'eT{X}{D}', f"{s['t']:.2f}".replace('-', '$-$'))
            mac(f'eXreDiff{X}{D}', fmt(x['mean']))
            mac(f'eXreCI{X}{D}', f"[{fmt(x['lo'])}, {fmt(x['hi'])}]")
            mac(f'eXrePos{X}{D}', f"{x['pos']}/5")
            mac(f'eXreP{X}{D}', f"{x['p']:.3f}" if x['p'] >= 0.001 else '$<$0.001')
        sig = [D for d, D in DS if res[(X, d)]['p'] < ALPHA]
        sig_holm = [DS[i][1] for i in sorted(holm([res[(X, d)]['p'] for d, _ in DS]))]
        mac(f'eSig{X}', phrase(sig))
        mac(f'eNSig{X}', WORDS[len(sig)])
        mac(f'eSig{X}Holm', phrase(sig_holm))
        mac(f'eSigUp{X}', phrase([D for d, D in DS if res[(X, d)]['p'] < ALPHA and res[(X, d)]['mean'] > 0]))
        mac(f'eSigDown{X}', phrase([D for d, D in DS if res[(X, d)]['p'] < ALPHA and res[(X, d)]['mean'] < 0]))
        res[X] = (sig, sig_holm)
    mac('eNDs', WORDS[len(DS)])
    mac('eDsPhrase', phrase([D for _, D in DS]))
    MAC.write_text('\n'.join(out) + '\n', encoding='utf-8', newline='\n')
    print('wrote', MAC)
    for key, s in res.items():
        if isinstance(key, tuple):
            lab = 'masked-full' if len(key) == 2 else 'XRE-full   '
            X, d = key[-2], key[-1]
            print(f"{lab} {X} {d:10s} a {s['a']:.4f} b {s['b']:.4f} diff {s['mean']:+.5f} CI [{s['lo']:+.5f}, "
                  f"{s['hi']:+.5f}] pos {s['pos']}/5 t {s['t']:+.2f} p {s['p']:.4f} d {[round(x, 4) for x in s['d']]}")
    print('significant masked-full (p<0.05): R', res['R'][0], 'N', res['N'][0], '| Holm: R', res['R'][1], 'N', res['N'][1])


def main():
    if '--no-collect' not in sys.argv:
        collect()
    macros()


if __name__ == '__main__':
    main()
