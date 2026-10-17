"""Collect the list-composition statistics of the crowding figure (figures/src/fig_crowding.py) from HOST_A. Reads remote
CSV files only, trains nothing.

  tables/listana_stats.csv  (dataset, arm, metric, value, n_users, source)
  tables/listana_rank.csv   (dataset, arm, condition, metric, value, n, source)
Sources (written by <RUN_ROOT_B>/P04_listana/tools/list_stats_full.py --summarize; per cell --one <run_dir>, CPU,
inference only; the tool stops unless the restored checkpoint reproduces result.json test Recall@20 to 4 dp):
  Baby, Pantry, Scientific  <RUN_ROOT_B>/P04_listana/stats/list_stats.csv and train_vs_test_rank.csv
                            (tools/run_all_full.sh; seed-1 P04_listana reruns of MSCA Native and XRE, source 'listana').
                            These two files were copied verbatim into tables/ before this collector existed; the collector
                            stops if their rows differ from the current tables/ rows.
  Sports                    <RUN_ROOT_B>/P04_listana/stats_sports/list_stats.csv and train_vs_test_rank.csv
                            (stats_sports/run_sports.sh; seed-1 Table I runs of P04_sports/runs_xres: objx_s sports_native
                            and xres sports_xres_xre_l010_k40, repo P04_xres_wt 406a7a7; source label 'tableI' because the
                            runs are not under P04_listana; summary with --datasets sports).
Usage (repository root): python -s tables/src/collect_listana.py
"""
import csv
import io
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / 'tables'
SSH = ['ssh', '-o', 'BatchMode=yes', 'HOST_A']
BASE = '<RUN_ROOT_B>/P04_listana/stats'
SPORTS = '<RUN_ROOT_B>/P04_listana/stats_sports'
FILES = [('list_stats.csv', 'listana_stats.csv'), ('train_vs_test_rank.csv', 'listana_rank.csv')]
OLD_DSS = {'baby', 'pantry', 'scientific'}


def fetch(path):
    out = subprocess.run(SSH + [f'cat {path}'], capture_output=True, check=True).stdout.decode('utf-8')
    return out.replace('\r\n', '\n')


def main():
    for src, dst in FILES:
        base, sp = fetch(f'{BASE}/{src}'), fetch(f'{SPORTS}/{src}')
        hb, _, bb = base.partition('\n')
        hs, _, bs = sp.partition('\n')
        if hb != hs:
            sys.exit(f'column mismatch {src}: {hb} | {hs}')
        rows_sp = list(csv.DictReader(io.StringIO(sp)))
        if {r['dataset'] for r in rows_sp} != {'sports'}:
            sys.exit(f'{SPORTS}/{src}: datasets {sorted({r["dataset"] for r in rows_sp})}')
        old = T / dst
        if old.exists():
            cur = [ln for ln in old.read_text(encoding='utf-8').replace('\r\n', '\n').splitlines()[1:]
                   if ln.split(',')[0] in OLD_DSS]
            if cur != bb.rstrip('\n').splitlines():
                sys.exit(f'{BASE}/{src} differs from the Baby / Pantry / Scientific rows of tables/{dst}')
        text = hb + '\n' + bb.rstrip('\n') + '\n' + bs.rstrip('\n') + '\n'
        (T / dst).write_text(text, encoding='utf-8', newline='\n')
        print('wrote', dst, text.count('\n') - 1, 'rows')


if __name__ == '__main__':
    main()
