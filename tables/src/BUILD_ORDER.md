# Generation order of the macro and table generators (revision log 33)

`tables/src/make_all.py` runs the generators in the order of its `COLLECT` and `BUILD` lists. A generator that reads the output file of another generator must follow it. The dependencies below were found by searching every `tables/src/*.py` and `figures/src/*.py` for reads of `tables/macros_*.tex` and for imports of other generators.

## Reads of another generator's macro file (must run after the generator in the second column)

| Reader | Reads | Writer (runs first) |
|---|---|---|
| build_macros_unfused.py | macros_msca10.tex (xnMargin) | build_macros_msca10.py |
| build_rev2.py | macros_classic.tex (clNComboPairs and others) | build_table_classic.py |
| build_table_defparams.py | macros_fspec.tex, macros_sotafill.tex | build_table_fspec.py, build_sotafill_macros.py |
| build_table_prereg.py | macros_mes10, macros_classic, macros_mesf, macros_mesc, macros_babybuild, macros_fusionprose, macros_rev2, macros_msca10 | build_macros_mes10.py, build_table_classic.py, build_macros_mesf.py, build_macros_mesc.py, build_table_babybuild.py, build_fusion_prose_macros.py, build_rev2.py, build_macros_msca10.py |
| build_ctrl.py | macros_xmask.tex | build_table_xmask.py |
| build_sigcounts.py | macros_xmask.tex | build_table_xmask.py |

## Imports of another generator as a module (read-only reuse)

build_rev2.py imports build_table_classic; build_table_classic.py imports build_table_protocol_restored (macro_lines() writes the px macros into macros_classic.tex); build_table2_main.py, build_table_fusion_added.py, build_table_defparams.py, and figures/src/fig_teaser_b.py import build_rev2; build_paired.py imports build_table_xmask; build_table_complement.py imports build_gs5_macros.

## Order that make_all.py now fixes

build_table_classic.py, then build_rev2.py, then the main tables; build_table_xmask.py before build_ctrl.py, build_fusion_test.py, and build_sigcounts.py; build_table_fspec.py and build_sotafill_macros.py before build_table_defparams.py; and at the end of BUILD

1. build_macros_mes10.py
2. build_macros_msca10.py
3. build_macros_unfused.py (new in the list; must follow 2)
4. build_macros_mesf.py (new in the list)
5. build_macros_mesc.py (new in the list)
6. build_table_babybuild.py (new in the list)
7. build_table_defparams.py
8. build_table_prereg.py (must follow 1 to 6 and build_rev2.py, build_fusion_prose_macros.py)
9. figures/src/fig_teaser_b.py

Run on a scratch copy with the old output files in place, items 1 to 6 and build_coldroute_macros.py and build_table_classic.py reproduced every file of `tables/` byte for byte.

## Not run by make_all.py (the output files are read as they are)

`macros_hyp_forest.tex` is written by collect_hyp_forest.py, which reads macros_paired.tex (build_paired.py), macros_xmask.tex (build_table_xmask.py), macros_m1probe.tex (collect_m1probe.py), and macros_e1.tex (collect_e1.py). build_ctrl.py reads macros_hyp_forest.tex, and `macros_hyp_strathist_mes.tex` (collect_hyp_strat_mes.py) is read by build_strathist_mes_count.py. After a change of the inputs of these chains, run collect_hyp_forest.py after build_paired.py, then build_ctrl.py again. The collect_* scripts that read HOST_A are run by `make_all.py` without `--no-collect` only.
