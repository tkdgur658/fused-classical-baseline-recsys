# How Much Gain Survives a Fused Classical Baseline? Measuring Learned Multimodal Recommenders

Code and data records for the manuscript of the same title, submitted to IEEE Transactions on Multimedia.
Authors: Sanghyuck Lee, Chaewoon Kim, Mohd Asyraf Zulkifley, and Jaesung Lee.
Corresponding author: Jaesung Lee (curseor@cau.ac.kr).

This repository holds the scripts that turn run results into every number of the manuscript, the statistical test, the
classical rankers and the fused baseline, the two supplementary-data CSV files, the per-run CSV files that the manuscript
says are public, and the preregistration documents of Table S9.

## Layout

| Path | Content |
|---|---|
| `tables/src/*.py` | Generators of every table and every number in the prose (`tables/macros_*.tex`). `make_all.py` runs them in dependency order (`BUILD_ORDER.md`). |
| `tables/src/sigtest.py` | The single test family: two-sided paired t-test over seeds, Holm correction over the datasets of a contrast. |
| `tables/src/data/` | Result records read by the generators (per-seed metrics, verdict files of the preregistered studies, classical-ranker results, fusion cells, the preregistration registry). |
| `tables/*.csv`, `tables/*.tsv` | Intermediate records that the collectors wrote from the run directories and that the generators read (80 files, 9.9 MB), among them `tables/ctrl_*.csv`, `tables/hyp_forest.csv`, and `tables/novctrl_*.csv`, the per-seed records and contrasts of the controlled-component diagnosis of the earlier full design (E + R + A). |
| `tables/*.tex`, `tables/claims_check.txt` | The table and macro files that the manuscript reads. The generators rewrite them; some generators also read macro files that earlier scripts wrote. |
| `figures/src/*.py` | Generators of the figures (matplotlib), run by `make_all.py` from `figures/src/`. |
| `tables/src/data/baby_cache_audit.csv`, `tables/src/data/baby_cache_audit_g2m.csv` | The item kNN cache build of every Baby run (Table S6). |
| `supplementary_data_protocol.csv`, `supplementary_data_classic_fusion.csv` | The supplementary-data files behind Tables D1-D6 of the supplementary data. |
| `server_code/table_fusion/` | `tf_fuse.py`, `tf_queue.py`: the fused baseline F (z-scored RP3beta + ItemKNN-CBF added to a learned score with a validation weight) and its reproduction gate. |
| `server_code/classic/` | `classic_common.py`, `classic_ds.py` (standalone RP3beta, ItemKNN-CBF, ItemKNN-CF, EASE, and their validation mix), `tf_fuse_ce.py`, `tf_queue_ce.py` (fusion of every learned model with each classical ranker and the mix), `run_e1a.sh`. |
| `server_code/mesf/` | `mesfres_sc.py`, `run_sc.sh`: the classical-score cache of the MES-F and MES-C variants. |
| `preregistration/` | The eleven preregistration documents of Table S9 (sanitized copies, see below). |
| `SPLIT_AND_PREPROCESSING.md` | Datasets, split, feature preprocessing, and evaluation. |

## Reproducing the numbers

Requirements: Python 3.9 or later, NumPy, SciPy, and matplotlib. Hardware of the runs: a single consumer GPU per run.

- `python -s tables/src/make_all.py --no-collect` rebuilds the tables, macro files, and figures from the files in
  `tables/` and `tables/src/data/` without any run directory. In a clean copy of this repository it runs the first 55 of
  its 57 steps (about one minute on a laptop CPU). Comparing the rebuilt `tables/` with the shipped one, 167 of
  the 170 files are identical apart from LaTeX comments: `tables/table1_marks.csv` differs in the 16th decimal of
  a floating-point helper column, `tables/claims_check.txt` lists the same lines in another order, and one prose macro of
  `tables/macros_results.tex` differs by one word. `figures/src/fig_teaser_b.py`, the last step, also runs on its own.
- Step 56, `tables/src/build_table_prereg.py`, reads the original preregistration documents and three result notes of the
  preregistered studies, which are not public (see below); it stops in this repository, and its output
  (`tables/appendix_prereg.tex`, `tables/table_factstatus.tex`, `tables/src/data/prereg_registry.csv`) is shipped as built. Run `python -s figures/src/fig_teaser_b.py` from
  `figures/src/` to finish the build.
- The `collect_*.py` scripts read run directories on the compute servers and are kept as a record of how each CSV was
  made. They cannot run outside those servers.
- Absolute paths and host names were replaced by placeholders: `<RUN_ROOT>` (root of the fusion, classical, and
  preregistered runs), `<RUN_ROOT_B>`, `<RUN_ROOT_C>`, `<POD_ROOT>` (roots of the earlier runs), `<DATA_ROOT>`, `<HOME>`,
  `<DISK>`, `<ENV_BIN>`, `<CONDA>`, `<REPO>` (this repository), `<PREREG_DIR>` (folder of the preregistration documents),
  `<TOOLS>`, `<TMP>`, `<LOCAL_PATH>`; `HOST_A` to `HOST_D` name the machines, `<SSH_KEY>` and `<PORT>` the access
  parameters, and `study/` the run folder of the preregistered studies. Replace them with your own paths before running.
  Numeric values were not changed by this replacement.
- Device names were replaced as well: the `gpu`, `gpus`, `ref_gpu`, and `arm_gpu` columns of the records hold the
  neutral token `GPU`, and run folders named after a device or a host carry neutral names (`P04_main_runs`, `hA_s45`,
  `spA`). No generator uses these columns for a result; the collectors and `build_cost_dns.py` only print a non-fatal
  warning when the two runs of a pair report different devices. The device-name filters of the collectors were removed;
  their other eligibility rules are unchanged.

## Preregistration documents (Table S9)

Table S9 of the supplementary material prints the first 16 hexadecimal digits of the SHA-256 of each archived document.
The copies in `preregistration/` are **sanitized**: internal file paths, server names, and internal process vocabulary were
replaced by the placeholders above or by neutral words (for example the name of the deciding role became "저자", the
author), and device names became `GPU` or were dropped. The plan, hypotheses, thresholds, script hashes, dates, and recorded outcomes were not changed. Because of these
replacements, the public copies do not reproduce the original hashes. The documents are written in Korean.

| Document | File | Date | SHA-256 of the original (Table S9) | SHA-256 of the public copy |
|---|---|---|---|---|
| MES9 | `PREREG_MES9.md` | 2026-10-15 | `b18d62336566f507bc74bf4552d86fa0efaf1098c0f6cb1e47d77eda3ab0ebc2` | `9f34439ebf38e4e82334ea60c1bfa82634a6699ef5d9936845425b17cc9d8b89` |
| R2NULL2 | `PREREG_R2NULL2.md` | 2026-10-15 | `07e998d996bc997cc61ae4afd3f47c7960cc9826062354048185349fc10b7b11` | `dd695a1b98ec0715456ad10edcc99e124e81b606e907b249f3aee594a8c68450` |
| R2NULL3 | `PREREG_R2NULL3.md` | 2026-10-15 | `4a41c91f8789a3114f17f775f9df8b10ddfac9f5d063bb3eb4689284c3f1b2e9` | `373158b1a883a27c696920b6a2d2d2ff7307d35e6cc0255267e0e439ae9ee4d6` |
| MESCKPT | `PREREG_MESCKPT.md` | 2026-10-16 | `6a78bc4df09edbea10442f159df44720cb2b42cfa5330ef626e555cff35551d7` | `a1eff360b16eddefabb8eff4ebb28df002d0e1ff7a0028c89f0bb1ddd7679366` |
| MESF | `PREREG_MESF.md` | 2026-10-16 | `f2493802bafb940a90ebfc98fec4da84bfed6e465e36f24baefe142fc9a54b67` | `61ddb0ac90d8fd8500ec399560d25c2d9eef1c2a1e9da70f975e02f1359d20fd` |
| MESC | `PREREG_MESC.md` | 2026-10-16 | `c7b1184772ab53e61e3e2c690e8972a3b999d8011b9a7c2c4e74083225966e2b` | `5a3dad33a3ca9ad2bdee073aca916da6331cdd652f09f4922591098751463aa9` |
| MES10 (A1) | `PREREG_MES10.md` | 2026-10-16 | `8a07ad2d42db0ed41dfc6a90cd5869b5c26722a5cb9852c050039225a2c79e88` | `a9969f096f33559b63d0d90ee2e7d7e489de28721b3c022274faf1240294ec8d` |
| CLASSIC | `PREREG_CLASSIC.md` | 2026-10-16 | `f299288c7cb917aa0aa0eb6802599651d4e473ed03ff96b406deb7b9bf5ce05c` | `34e5a6cf4a2496688f094ebd82a3c2b4dc6f818fffe330b9a89d656f120916c1` |
| CACHEFIX | `PREREG_CACHEFIX.md` | 2026-10-16 | `2ba8cc5f356be09e12ecbed22d65372e9068d56773e785ce58ed11b4c5251e19` | `5a8ffbb283dd40d39dd655b6333c35c2d924d7658dc15a8fe5dd92a34dcd2b42` |
| SOTAFILL | `PREREG_SOTAFILL.md` | 2026-10-16 | `3331011287020d409f8f9343c139eff4310d38ff689c28118670e05f10941119` | `8fa8d94d1c82d9796dcc5d33820214731aa505abf1131f8373b63f3a3bddd06b` |
| MSCA10 | `PREREG_MSCA10.md` | 2026-10-17 | `d54733a25212ad5dd877bceeb168ea40a48a3c6b0f9dd23cc65996f0881a6f18` | `4a297395fe64d758806f3a20e5850b20fffba25ca9345a50ac9c04188689ba7c` |

The dates are the dates written in each document. A document also holds its completion record, added after the runs.
`tables/src/build_table_prereg.py` computes the hashes of Table S9 from the original files; the public hashes above are
the SHA-256 of the files in `preregistration/`.

## What is not included

- The raw interaction and feature files of the datasets (public releases of MMRec, MISSRec, and MicroLens).
- Model checkpoints, score dumps, and caches of the runs.
- The training code of the learned recommenders, which follows the released code of each model in an MMRec-based pipeline.

## License

MIT, see `LICENSE`.
