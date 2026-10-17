"""Collect the single-probe control of the anchor term (reviewer control "M = 1") and write its paired macros.

Claim under test: the anchor term (A) helps because each co-occurrence neighbor of a training positive is ranked above the
HIGHEST-scoring of the M = 50 uniform probes. Control: the identical anchor term (J = 10 neighbors, count weights,
lambda 0.1, everything else equal) whose partner is probe 0 of the SAME 50 draws (the 50 probes are still drawn, so the
CUDA RNG stream is unchanged; equivalent to one uniform probe, M = 1).
Runs: HOST_A <RUN_ROOT_B>/P04_m1probe/runs_m1a/xres/*/result.json, config <ds>_xres_a1first_l010 (repo
<HOME>/P04_xres_m1_wt, branch xres-m1probe 203e5e2; README.txt and collect_m1.py in the work dir), MSCA,
Baby / Pantry / Scientific, seeds 1-5. Sports (added 2026-10-07): HOST_B runs synced to HOST_A
<RUN_ROOT_B>/P04_sports_ctrl_HOST_B/runs_ctrl/xres (branch xres-m1probe-sports b259d9a = 203e5e2 on the HOST_B Sports
ablation code 6003cba; <TOOLS>/hyp_figs_20261007/E1E2_RUNS.md 'Sports 대조 2행'), reference = the Sports A1 runs of the
same machine (<RUN_ROOT_B>/P04_sports_abl_HOST_B/runs_abl/xres), so both runs of every pair share one machine. Eligible run: status SUCCESS, mode train, test_eval best_primed. One run per
(dataset, seed): a second eligible run is an error (no replicates), the script stops.
Reference: Table III row "Anchor term (A)" = MSCA row A1 of tables/xre_perseed.csv (<ds>_xres_a1_l010, the same seeds). The reference runs on HOST_A (<RUN_ROOT_B>/P04_main_runs/qxs2/A/xres) are read as well, and their values
must equal xre_perseed.csv digit for digit (cross-check, the script stops otherwise).
Outputs
  tables/m1probe_perseed.csv   row A1first, dataset, seed, r20, n20, best_epoch, gpu, config, run_dir (test metrics of the
                               best-validation checkpoint, best_test of result.json), plus ref_r20 / ref_n20 (row A1).
  tables/macros_m1probe.tex    paired differences strongest probe - single probe (A1 - A1first, same seed, seeds 1-5)
                               (macro names mp..., no digits), for D in Baby, Pantry, Scientific, Sports and X in R (Recall@20),
                               N (NDCG@20):
    mpMean<X>Single<D>   mean of the single-probe row (4 dp, the format of Table III, leading zero dropped)
    mpDiff<X><D>         mean difference (signed, 4 dp, the format of macros_hyp_forest.tex)
    mpCI<X><D>           95% paired t-interval, df 4
    mpPos<X><D>          seeds on which the strongest probe is higher (k/5)
    mpP<X><D>            two-sided paired t-test p (3 dp, '$<$0.001' below)
    mpT<X><D>            paired t statistic (2 dp)
    mpSig<X> / mpNSig<X> datasets with p < 0.05 (phrase / word); mpSig<X>Holm the same after Holm over the datasets
    mpPhrase             one data-driven clause on what the strongest probe adds, mpNDs, mpDsPhrase
Usage (repository root): python -s tables/src/collect_m1probe.py          (ssh to HOST_A, then macros)
                         python -s tables/src/collect_m1probe.py --no-collect   (macros from the CSV already in tables/)
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
OUT = TABLES / 'm1probe_perseed.csv'
MAC = TABLES / 'macros_m1probe.tex'
REMOTE = {d: '<RUN_ROOT_B>/P04_m1probe/runs_m1a/xres' for d in ('baby', 'pantry', 'scientific')}
REF_REMOTE = {d: '<RUN_ROOT_B>/P04_main_runs/qxs2/A/xres' for d in ('baby', 'pantry', 'scientific')}
REMOTE['sports'] = '<RUN_ROOT_B>/P04_sports_ctrl_HOST_B/runs_ctrl/xres'
REF_REMOTE['sports'] = '<RUN_ROOT_B>/P04_sports_abl_HOST_B/runs_abl/xres'
DS = [('baby', 'Baby'), ('pantry', 'Pantry'), ('scientific', 'Scientific'), ('sports', 'Sports')]
SEEDS = [1, 2, 3, 4, 5]
ALPHA = 0.05
WORDS = 'none one two three four'.split()
PY = ("import json,sys,os\n"
      "for f in sys.argv[1:]:\n"
      "  j=json.load(open(f)); b=j.get('best_test') or {}\n"
      "  print(json.dumps(dict(cfg=j.get('config'),ds=j.get('dataset'),seed=j.get('seed'),status=j.get('status'),"
      "mode=j.get('mode'),te=j.get('test_eval'),r20=b.get('recall@20'),n20=b.get('ndcg@20'),ep=j.get('best_epoch'),"
      "gpu=(j.get('gpu') or '').replace('NVIDIA ',''),run=os.path.dirname(f))))\n")


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
               for r in csv.DictReader(open(TABLES / 'xre_perseed.csv', encoding='utf-8'))
               if r['backbone'] == 'MSCA' and r['row'] == 'A1'}
    rows, bad = [], []
    for d, _ in DS:
        m1 = fetch(f'{REMOTE[d]}/xres__{d}_xres_a1first_l010__{d}__s*__train__*')
        ref = fetch(f'{REF_REMOTE[d]}/xres__{d}_xres_a1_l010__{d}__s*__train__*')
        for s in SEEDS:
            assert (d, s) in m1, f'missing single-probe run {d} s{s}'
            assert (d, s) in ref and (d, s) in ref_csv, f'missing reference run {d} s{s}'
            j, a = m1[(d, s)], ref[(d, s)]
            if (round(a['r20'], 6), round(a['n20'], 6)) != tuple(round(x, 6) for x in ref_csv[(d, s)]):
                bad.append(f'reference {d} s{s}: HOST_A {a["r20"]}/{a["n20"]} vs xre_perseed.csv {ref_csv[(d, s)]}')
            assert j['cfg'] == f'{d}_xres_a1first_l010', j
            if j['gpu'] != a['gpu']:   # device check (non-fatal): both runs of a pair should share one device
                print(f'WARNING {d} s{s}: the two runs of the pair report different devices', file=sys.stderr)
            rows.append(dict(row='A1first', dataset=d, seed=s, r20=f"{j['r20']:.6f}", n20=f"{j['n20']:.6f}",
                             best_epoch=j['ep'], gpu=j['gpu'], config=j['cfg'], run_dir=j['run'],
                             ref_r20=f'{ref_csv[(d, s)][0]:.6f}', ref_n20=f'{ref_csv[(d, s)][1]:.6f}'))
        extra = sorted(set(m1) - {(d, s) for s in SEEDS})
        assert not extra, f'unexpected single-probe runs {extra}'
    if bad:
        print('MISMATCH:', *bad, sep='\n  ')
        sys.exit(1)
    assert len({(r['dataset'], r['seed']) for r in rows}) == len(rows) == 5 * len(DS)
    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    print('wrote', OUT, len(rows), 'rows; reference values equal xre_perseed.csv (MSCA, A1)')


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


def macros():
    V = {}
    for r in csv.DictReader(open(OUT, encoding='utf-8')):
        key = (r['dataset'], int(r['seed']))
        assert key not in V, f'duplicate {key}'
        V[key] = r
    out = ['% Generated by tables/src/collect_m1probe.py from tables/m1probe_perseed.csv (single-probe anchor term, M = 1)',
           '% and the MSCA A1 rows of tables/xre_perseed.csv. Do not edit by hand. Differences are strongest probe (Table III',
           '% "Anchor term (A)") minus single probe (same seed, seeds 1-5, one run per seed; both rows of a pair on the same machine);'
           '% CI = 95% paired t-interval (df 4);',
           '% P = two-sided paired t-test; Pos = seeds on which the strongest probe is higher.']

    def mac(n, v):
        assert re.fullmatch(r'[A-Za-z]+', n), n
        out.append(f'\\newcommand{{\\{n}}}{{{v}}}')
    res = {}
    for X, col in (('R', 'r20'), ('N', 'n20')):
        for d, D in DS:
            strong = [float(V[(d, s)][f'ref_{col}']) for s in SEEDS]
            single = [float(V[(d, s)][col]) for s in SEEDS]
            dd = [a - b for a, b in zip(strong, single)]
            mu, sd = st.mean(dd), st.stdev(dd)
            h = stats.t.ppf(0.975, len(dd) - 1) * sd / math.sqrt(len(dd))
            tt = stats.ttest_rel(strong, single)
            res[(X, d)] = dict(mean=mu, lo=mu - h, hi=mu + h, pos=sum(x > 0 for x in dd), p=float(tt.pvalue),
                               t=float(tt.statistic), single=st.mean(single), strong=st.mean(strong), d=dd)
            s = res[(X, d)]
            mac(f'mpMean{X}Single{D}', f"{s['single']:.4f}".replace('0.', '.', 1))
            mac(f'mpDiff{X}{D}', fmt(s['mean']))
            mac(f'mpCI{X}{D}', f"[{fmt(s['lo'])}, {fmt(s['hi'])}]")
            mac(f'mpPos{X}{D}', f"{s['pos']}/5")
            mac(f'mpP{X}{D}', f"{s['p']:.3f}" if s['p'] >= 0.001 else '$<$0.001')
            mac(f'mpT{X}{D}', f"{s['t']:.2f}".replace('-', '$-$'))
        sig = [D for d, D in DS if res[(X, d)]['p'] < ALPHA]
        sig_holm = [DS[i][1] for i in sorted(holm([res[(X, d)]['p'] for d, _ in DS]))]
        res[X] = (sig, sig_holm)
        mac(f'mpSig{X}', phrase(sig))
        mac(f'mpNSig{X}', WORDS[len(sig)])
        mac(f'mpSig{X}Holm', phrase(sig_holm))
        # datasets with a significant difference whose sign favors the strongest probe
        mac(f'mpSigUp{X}', phrase([D for d, D in DS if res[(X, d)]['p'] < ALPHA and res[(X, d)]['mean'] > 0]))
    sigR, sigN = res['R'][0], res['N'][0]
    upN = [D for d, D in DS if res[('N', d)]['p'] < ALPHA and res[('N', d)]['mean'] > 0]
    if upN and not sigR:
        ph = f'the strongest probe raises NDCG@20 on {phrase(upN)} without a significant Recall@20 change on any dataset'
    else:
        ph = (f'the strongest probe changes Recall@20 significantly on {phrase(sigR)} and NDCG@20 on {phrase(sigN)}')
    mac('mpPhrase', ph)
    # R0500: c) cites \mpCINPantry and \mpCINScientific after \mpSigUpN, so the two must name the same datasets.
    assert phrase(upN) == 'Pantry and Scientific', ('c) NDCG sentence cites Pantry/Scientific CIs', upN)
    mac('mpNDs', WORDS[len(DS)])
    mac('mpDsPhrase', phrase([D for _, D in DS]))
    MAC.write_text('\n'.join(out) + '\n', encoding='utf-8', newline='\n')
    print('wrote', MAC)
    for (k, *rest), s in ((key if isinstance(key, tuple) else (key,), v) for key, v in res.items()):
        if rest:
            print(f"{k} {rest[0]:10s} strongest {s['strong']:.4f} single {s['single']:.4f} diff {s['mean']:+.5f} "
                  f"CI [{s['lo']:+.5f}, {s['hi']:+.5f}] pos {s['pos']}/5 t {s['t']:+.2f} p {s['p']:.4f}")
    print('significant (p<0.05): R', sigR, 'N', sigN, '| Holm: R', res['R'][1], 'N', res['N'][1])
    print('phrase:', ph)


def main():
    if '--no-collect' not in sys.argv:
        collect()
    macros()


if __name__ == '__main__':
    main()
