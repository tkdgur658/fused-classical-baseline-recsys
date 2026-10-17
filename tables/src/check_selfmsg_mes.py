"""PREREG_MESCKPT unit U6: consistency check of the MES self-message cells against tables/analysis_selfmsg.csv (arm mes) and
tables/mes_perseed.csv (MSCA, MES run directories). Nothing is regenerated; the verdict is written to
tables/src/data/selfmsg_mes_check.txt.

  server  python check_selfmsg_mes.py remote > out/selfmsg_raw.json
  local   python tables/src/check_selfmsg_mes.py --raw selfmsg_raw.json
Per cell <ds>_mes_s<k> (Baby, Pantry, Scientific: P04_newprin/prop2/out; Sports: P04_newprin/prop2_sports/out): repro_ok true,
run_dir equals the MES run_dir of mes_perseed.csv, re-evaluated test Recall@20 equals the test_r20 column of the bin 'all' row of
analysis_selfmsg.csv and the n_pairs agree. 20 cells; fig_selfmsg (b) and the smShare / gradLost macros read exactly these rows.
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DSS = ['baby', 'pantry', 'scientific', 'sports']
SEEDS = [1, 2, 3, 4, 5]
DIRS = {'sports': '<RUN_ROOT_B>/P04_newprin/prop2_sports/out'}
DEF = '<RUN_ROOT_B>/P04_newprin/prop2/out'


def remote():
    out = []
    for ds in DSS:
        for s in SEEDS:
            j = json.loads((Path(DIRS.get(ds, DEF)) / f'{ds}_mes_s{s}.json').read_text())
            out.append([ds, s, j['repro_ok'], j['run_dir'], j['reeval_test_recall20'], j['result_json_test_recall20'], j['n_pairs']])
    json.dump(out, sys.stdout)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'remote':
        return remote()
    cells = json.load(open(sys.argv[sys.argv.index('--raw') + 1], encoding='utf-8'))
    csvrows = {(r['dataset'], int(r['seed'])): r for r in csv.DictReader(open(ROOT / 'tables/analysis_selfmsg.csv', encoding='utf-8'))
               if r['arm'] == 'mes' and r['bin'] == 'all'}
    mes = {(r['dataset'], int(r['seed'])): r['run_dir'].rstrip('/') for r in csv.DictReader(open(ROOT / 'tables/mes_perseed.csv', encoding='utf-8'))
           if r['backbone'] == 'MSCA' and r['row'] == 'MES'}
    lines, bad = [], 0
    for ds, s, ok, rd, re_, rj, npairs in cells:
        c = csvrows[(ds, s)]
        checks = dict(repro_ok=bool(ok), run_dir=rd.rstrip('/') == mes[(ds, s)], r20=abs(re_ - float(c['test_r20'])) < 1e-9,
                      r20_json=abs(re_ - rj) < 1e-9, n_pairs=int(c['n_pairs']) == npairs)
        bad += not all(checks.values())
        lines.append(f'{ds} s{s}: ' + ' '.join(f'{k}={"ok" if v else "FAIL"}' for k, v in checks.items()))
    lines.append(f'cells {len(cells)}, failing {bad}')
    (ROOT / 'tables/src/data/selfmsg_mes_check.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines[-3:]))
    if bad:
        sys.exit('U6 check FAILED')


if __name__ == '__main__':
    main()
