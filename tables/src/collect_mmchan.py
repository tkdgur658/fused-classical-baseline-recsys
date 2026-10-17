"""Collect the self-message channel analysis (problem statement, Fig. 1) from HOST_A. Reads remote files, trains nothing.

  tables/mmchan_perseed.csv        = <RUN_ROOT_B>/P04_newprin/mmchan/summary_long.csv (summ.py)
      MSCA Native / XRE (P04_listana checkpoints, seeds 1-5, mmchan2.py) and LightGCN (ID-only, MMRec_clean checkpoints:
      Baby s1, Pantry s1-s3, Scientific s1, lgcn_self.py), same split and training-positive sample (hash-checked).
      LightGCN Sports s1-s5 are appended from mmchan_sports (below).
  tables/mmchan_topk_perseed.csv   = <RUN_ROOT_B>/P04_newprin/mmchan/summary3_long.csv (summ3.py)
      MSCA Native (same checkpoints) and the Table I top-K loss checkpoints PSL / SL@K / SSM (P04_topk runs_topkE, chosen
      configs, seeds 1-5, mmchan3.py), with the EFLEC-style removal of the collaborative channel of the self-message.
  tables/mmchan_plg_perseed.csv    = <RUN_ROOT_B>/P04_newprin/mmchan/summary_plg_long.csv (summ_plg.py)
      the original best checkpoints of the Table I MSCA plug-in rows (MG, DA-MRS, INA, MDVT, VI-MMRec, UltraGCN-LI; Baby,
      Pantry, Scientific, seeds 1-5), fetched on 2026-10-05 from the pod volumes into <RUN_ROOT_B>/P04_podckpt and
      restored with the pod repo copies (prep_plg_cells.py). DA-MRS cells add its item-side offset (h_v + h_t + h_s)/3 to the
      item vector, as its loss and scoring do. Restored test Recall@20 equals result.json to four decimals in 73 cells and
      differs by at most 0.0005 in 17 (DA-MRS 15, UltraGCN-LI Baby s2/s3; meta repro_absdiff).
Sports (appended to the first two files; <RUN_ROOT_B>/P04_newprin/mmchan_sports, summ_sports.py, same metrics and
bootstrap as summ.py / summ3.py, imported unchanged):
  summary3_sports_long.csv       -> rows appended to mmchan_topk_perseed.csv: MSCA Native = the Table I Sports runs
      (P04_sports/runs_xres objx_s sports_native) and the Table I top-K checkpoints sports_psl_t005_dot / slk_t020_dot /
      ssm_t050_dot (P04_topk runs_topkE), seeds 1-5; pairs from prop2_sports/prop2_nl.py and cells from mmchan3_nl.py
      (copies of prop2.py / mmchan3.py generalized to the 3 LightGCN layers of the Sports MSCA config; their 2-layer path
      reproduces the original outputs bitwise, checked on baby_native_s1 / scientific_native_s1).
  summary_lgcn_sports_long.csv   -> rows appended to mmchan_perseed.csv when present: LightGCN Sports (lgcn_self_nl.py,
      written by mmchan_sports/add_lgcn_sports.sh from lgcn_sports_cells.txt; summ_sports.py stops unless the split hashes
      equal those of the MSCA Sports cells). Absent file = no LightGCN Sports rows (the teaser marks that bar n/a).
  Adding LightGCN Sports later (one command, repository root, after filling mmchan_sports/lgcn_sports_cells.txt):
      ssh HOST_A bash <RUN_ROOT_B>/P04_newprin/mmchan_sports/add_lgcn_sports.sh && python -s tables/src/collect_mmchan.py && python -s tables/src/build_mmchan_macros.py && python -s figures/src/fig_teaser.py
Columns: dataset, model, arm, seed, metric, value, lo, hi (user-bootstrap 95% CI, 1000 resamples, seed 0), n_pairs, n_users.
The checkpoints of the first two files reproduce their recorded test Recall@20 to four decimals (summ.py / summ3.py
assert repro_ok); the plug-in file follows the tolerance stated above.
Usage (repository root): python -s tables/src/collect_mmchan.py
"""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / 'tables'
REMOTE = '<RUN_ROOT_B>/P04_newprin/mmchan'
REMOTE_SP = '<RUN_ROOT_B>/P04_newprin/mmchan_sports'
# (remote file, local file, appended Sports file or None, Sports file required)
FILES = [('summary_long.csv', 'mmchan_perseed.csv', 'summary_lgcn_sports_long.csv', False),
         ('summary3_long.csv', 'mmchan_topk_perseed.csv', 'summary3_sports_long.csv', True),
         ('summary_plg_long.csv', 'mmchan_plg_perseed.csv', None, False)]


def fetch(path, required=True):
    out = subprocess.run(['ssh', '-o', 'BatchMode=yes', 'HOST_A', f'test -f {path} && cat {path} || echo __ABSENT__'],
                         capture_output=True, check=True).stdout.decode('utf-8').replace('\r\n', '\n')
    if out.strip() == '__ABSENT__':
        if required:
            raise SystemExit(f'missing remote file {path}')
        return None
    return out


def main():
    for src, dst, sp, req in FILES:
        out = fetch(f'{REMOTE}/{src}')
        if sp:
            add = fetch(f'{REMOTE_SP}/{sp}', req)
            if add is None:
                print('absent (no rows appended):', f'{REMOTE_SP}/{sp}')
            else:
                head, _, body = add.partition('\n')
                if head != out.partition('\n')[0]:
                    raise SystemExit(f'column mismatch: {sp} vs {src}')
                out = out.rstrip('\n') + '\n' + body
        (T / dst).write_text(out, encoding='utf-8', newline='\n')
        print('wrote', T / dst, out.count('\n') - 1, 'rows')


if __name__ == '__main__':
    main()
