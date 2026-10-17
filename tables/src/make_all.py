"""Re-run every collector and builder of the XRE manuscript in dependency order (local machine, repository root).

  python -s tables/src/make_all.py            collect from HOST_A, then rebuild every table, figure and macro file
  python -s tables/src/make_all.py --no-collect   rebuild from the CSV/TSV files already in tables/
collect_table1.py (Table II: Native and plug-in rows, seeds 1-5) runs first; every Native value of the manuscript
is read from its output (tables/src/native_rows.py). collect_topk.py (top-K loss rows SL@K / PSL / SSM of Table tab:main,
<RUN_ROOT_B>/P04_topk) runs right after it, because it checks its runs against the MSCA Native runs chosen in
tables/table1_runs.tsv. The Table I baseline collector (collect_table2.py, HOST_D access)
is owned by the author and is not re-run here; its CSV output is read as it is. fig_ablation.py runs before the
appendix tables and Table III because they read its tests (tables/ablation_tests.csv). Run with `python -s` (matplotlib on this machine).
Proposed method MES (revision 2026-10-08): the proposed rows and their macros read tables/src/mes_source.py. While
tables/mes_perseed.csv (or the file named by the environment variable MES_PERSEED) lacks a cell, the builders print a
provisional stand-in in \\tbd{} (red); dropping the file in and re-running `python -s tables/src/make_all.py --no-collect`
fills the real numbers. The status is printed at the end of the run.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COLLECT = ['tables/src/collect_table1.py', 'tables/src/collect_topk.py', 'tables/src/collect_xre.py', 'tables/src/collect_m1probe.py', 'tables/src/build_tau_boundary.py', 'tables/src/collect_dataset_stats.py',
           'tables/src/collect_hparams.py', 'tables/src/collect_table2_extra.py', 'tables/src/collect_analysis.py',
           'tables/src/collect_mmchan.py', 'tables/src/collect_xmask.py', 'tables/src/collect_e1.py', 'tables/src/collect_cohsmo.py',
           'tables/src/collect_cohsmo_topk.py', 'tables/src/collect_dotrank.py', 'tables/src/collect_listana.py',
           'tables/src/collect_recallk.py']
BUILD = ['tables/src/build_table_classic.py', 'tables/src/build_rev2.py',   # revision log 29: CLASSIC cells and the rev2 counts before Tables II, III
         'tables/src/build_table1_main.py', 'tables/src/build_table2_main.py', 'tables/src/build_table_fusion_added.py', 'figures/src/fig_ablation.py',
         'tables/src/build_table_xre_appendix.py',
         'tables/src/build_table_protocol.py', 'tables/src/build_table_datasets.py', 'tables/src/build_table_hparams.py', 'figures/src/fig_events.py', 'figures/src/fig_transfer.py',
         'figures/src/fig_method.py', 'figures/src/fig_selfmsg.py', 'figures/src/fig_gradsplit.py', 'figures/src/fig_recallk.py',
         'figures/src/fig_crowding.py',
         'tables/src/build_macros.py', 'tables/src/build_timing_macros.py', 'tables/src/build_analysis_macros.py',
         'tables/src/build_mmchan_macros.py', 'figures/src/fig_teaser.py', 'tables/src/build_table_xmask.py', 'tables/src/build_ctrl.py', 'figures/src/fig_ctrl_forest.py', 'figures/src/fig_nbrpop.py',
         'tables/src/build_table_cohsmo.py', 'tables/src/build_paired.py',
         'tables/src/build_fusion_test.py', 'tables/src/build_sigcounts.py',   # revision log 18-3: single test family (sigtest.py)
         'tables/src/build_stats_audit.py', 'tables/src/build_cosdot.py', 'tables/src/build_cost_dns.py',
         'tables/src/build_table_lamk.py', 'tables/src/build_power_macros.py',   # revision log 18-4 (inputs: tables/src/data/lamk_*.csv, power_fusion_s4.json)
         # revision log 18-2 (MES numbers only): N' with MES (tables/src/data/r2null3.json, n3p_mes.json), history-length check of
         # MES (tables/src/data/hyp_strat_hist_mes.csv), Fig. nbrpop from the MES checkpoint CSV
         'tables/src/build_nprime_mes.py', 'tables/src/build_diag_inst_macros.py', 'tables/src/build_strathist_mes_count.py', 'figures/src/fig_nbrpop_mes.py',
         # revision log 18-1: F and C specification, cold/routing, near-duplicate, and F prose macros
         'tables/src/build_table_fspec.py', 'tables/src/build_table_cfill.py', 'tables/src/build_coldroute_macros.py',
         'tables/src/build_neardup_macros.py', 'tables/src/build_fusion_prose_macros.py',
         # revision log 18-D: per-model near-duplicate table (w2_003.json) and the SOTAFILL gate macros (after build_sigcounts.py)
         'tables/src/build_table_neardup.py', 'tables/src/build_sotafill_macros.py',
         # revision log 19 (G-S5): stratified decomposition of the pre-fusion advantage (input tables/src/data/gs5_stats.json)
         'tables/src/build_gs5_macros.py', 'tables/src/build_table_complement.py',
         # revision log 29: MES10 macros, Definition 1 parameters, preregistration registry, Fig. 1
         # revision log 33 (review 3, minor): explicit generation order, see tables/src/BUILD_ORDER.md. build_macros_unfused.py reads macros_msca10.tex (xnMargin), thus it
         # follows build_macros_msca10.py; build_table_prereg.py reads macros_mes10, macros_classic, macros_mesf, macros_mesc, macros_babybuild, macros_fusionprose, macros_rev2,
         # and macros_msca10, thus it follows their generators; build_table_defparams.py reads macros_fspec and macros_sotafill, thus it follows build_table_fspec.py and build_sotafill_macros.py.
         # Before revision log 33 build_macros_unfused.py, build_macros_mesf.py, build_macros_mesc.py, and build_table_babybuild.py were not listed (their output files were read as they were).
         'tables/src/build_macros_mes10.py', 'tables/src/build_macros_msca10.py', 'tables/src/build_macros_unfused.py',
         'tables/src/build_macros_mesf.py', 'tables/src/build_macros_mesc.py', 'tables/src/build_table_babybuild.py',
         'tables/src/build_table_defparams.py', 'tables/src/build_table_prereg.py', 'figures/src/fig_teaser_b.py']


def main():
    steps = ([] if '--no-collect' in sys.argv else COLLECT) + BUILD
    for s in steps:
        cwd = ROOT / Path(s).parent if s.startswith('figures/') else ROOT
        print('>>', s, flush=True)
        subprocess.run([sys.executable, '-s', str(ROOT / s)], cwd=cwd, check=True)
    sys.path.insert(0, str(ROOT / 'tables' / 'src'))
    import mes_source
    print('>>', mes_source.source_text(['MSCA', 'COHESION', 'SMORE'], ['baby', 'pantry', 'scientific', 'sports']))


if __name__ == '__main__':
    main()
