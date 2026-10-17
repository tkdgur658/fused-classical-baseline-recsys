## collection status
- cell plg2_s/baby_damrs_a1b2_k10_x2: complete (5 seeds, gate PASS)
- cell plg2_s/baby_ugcn_li_k10_l0007: complete (5 seeds, gate PASS)
- cell cohesion/COHESION_A5_baby: complete (5 seeds, gate PASS)

# CACHEFIX simulation (real CACHEFIX values), 2026-10-17 07:19:17

temporary copy: <TMP> (deleted by nothing; remove by hand)

- baseline rebuild of table1_marks.csv identical to the manuscript copy
- baseline rebuild of sigtests_table1.csv identical to the manuscript copy
- baseline rebuild of table2_marks.csv identical to the manuscript copy
- baseline rebuild of sigtests_table2.csv identical to the manuscript copy

### tab:main (real CACHEFIX values)
- printed mean / SD cells that change: 4
  - MSCA/DA-MRS/baby/r20: 0.1020 (0.000777) -> 0.1020 (0.001081)
  - MSCA/DA-MRS/baby/n20: 0.0460 (0.000472) -> 0.0459 (0.000602)
  - MSCA/UltraGCN-LI/baby/r20: 0.1040 (0.000939) -> 0.1040 (0.000336)
  - MSCA/UltraGCN-LI/baby/n20: 0.0467 (0.000396) -> 0.0467 (0.000351)
- bold (best) marks that change: 0
- significance marks (triangles) that change: 0
- unchanged marks whose Holm-adjusted p moves (MSCA DA-MRS / UltraGCN-LI rows, informational): 5
  - MSCA: DA-MRS - MES | baby | r20: p_holm 0.009486 -> 0.009723, diff -0.0028 -> -0.0028
  - MSCA: DA-MRS - MES | baby | n20: p_holm 0.0006883 -> 0.003613, diff -0.0009 -> -0.0009
  - MSCA: UltraGCN-LI - MES | baby | r20: p_holm 0.3207 -> 0.315, diff -0.0008 -> -0.0008
  - MSCA: UltraGCN-LI - MES | pantry | r20: p_holm 0.3207 -> 0.315, diff -0.0011 -> -0.0011
  - MSCA: UltraGCN-LI - MES | baby | n20: p_holm 0.2204 -> 0.1079, diff -0.0002 -> -0.0001
### tab:sota (real CACHEFIX values)
- printed mean / SD cells that change: 2
  - COHESION/baby/r20: 0.1020 -> 0.1021
  - COHESION/baby/n20: 0.0443 -> 0.0444
- best-mean (bold) marks that change: 0
- significance marks vs MSCA + MES that change: 0
- COHESION tests whose Holm-adjusted p or sign moves (informational): 2
  - COHESION - MSCA + MES | baby | r20: p_holm 0.04035 -> 0.01871, diff -0.0028 -> -0.0027
  - COHESION - MSCA + MES | baby | n20: p_holm 0.001805 -> 0.001557, diff -0.0026 -> -0.0025
