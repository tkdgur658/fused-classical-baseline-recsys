"""One run set per Native value: every table that prints a Native row reads it from here.

Source: tables/table1_cells.csv and tables/table1_perseed.csv (tables/src/collect_table1.py), i.e. the Native row of
Table II (seeds 1-5, one run per seed).
  MSCA     Table I row "MSCA", Table II MSCA Native, Fig. ablation "MSCA (no auxiliary term)", prose macros, appendix
           protocol table (column C). The Table-I package run (msca MSCA_A5_<ds>) and objx_s <ds>_native are the same
           model configuration (the objx_s config = MSCA_A5_<ds> + inactive objx keys; first five batch losses bitwise
           equal at lambda = 0, ROUND10/12/13/14 unit checks), and objx_s <ds>_native seeds 1-5 are the official XRE
           baseline of the a14 judgement (<TOOLS>/thesis_share/OFFICIAL_CELLS_per_seed.csv), so the objx_s runs are used.
  FREEDOM, MMGCN   Table II Native and the appendix transfer table (the xret_f / xret_m native_l000 runs are bitwise
           identical to plg_f / plg2_m a4_native at lambda = 0 in the xre_t unit test, so the appendix shows the Table II
           runs and the rows of both tables differ only by the objective).
  Instruments: MSCA only (Table I Instruments column, user decision 2026-10-01; objx_s instruments_native under
           <RUN_ROOT_B>/P04_instr/runs*, see collect_table1.py). native('MSCA', 'instruments') feeds Table I and the
           Instruments gain macros; every other table and figure keeps baby / pantry / scientific.
A missing or incomplete cell (n < 5) returns None, and the callers print \\todo{pending}.
"""
import csv
from pathlib import Path

TABLES = Path(__file__).resolve().parents[1]
CELLS = TABLES / 'table1_cells.csv'
PERSEED = TABLES / 'table1_perseed.csv'
N_SEEDS = 5


def cells():
    if not CELLS.exists():
        return {}
    with open(CELLS, newline='', encoding='utf-8') as f:
        return {(r['backbone'], r['row'], r['dataset']): r for r in csv.DictReader(f)}


def complete(c):
    return c if c and c.get('n') and int(c['n']) == N_SEEDS else None


def native(backbone, dataset):
    """Native cell (row of table1_cells.csv) of a backbone, or None while fewer than five seeds exist."""
    return complete(cells().get((backbone, 'Native', dataset)))


def perseed(backbone, row='Native'):
    """{(dataset, metric): {seed: value}} with metric in r20, n20, r10, n10"""
    out = {}
    if not PERSEED.exists():
        return out
    with open(PERSEED, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if (r['backbone'], r['row']) == (backbone, row):
                for m in ('r20', 'n20', 'r10', 'n10'):
                    out.setdefault((r['dataset'], m), {})[int(r['seed'])] = float(r[m])
    return out
