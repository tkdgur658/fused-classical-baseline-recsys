"""PREREG_MESCKPT unit U1: rebuild the neighbor x popularity table of MSCA versus MES from the per-target cells, offline.

The server part is the unmodified `collect_nbrpop.py remote mes` (CPU, reads <RUN_ROOT_B>/P04_hyp_strat/cells/<ds>_{native,mes}_s<k>,
stops unless every cell passed repro_ok and recomputed_ok), its JSON goes to MESCKPT/out/nbrpop_raw.json. This script turns that JSON
into the CSV with the row format of collect_nbrpop.py and compares it with the committed tables/hyp_nbrpop.csv (U1 gate: byte-identical).

Usage (repository root):  python -s tables/src/collect_nbrpop_mes.py --raw <nbrpop_raw.json>
Writes tables/src/data/hyp_nbrpop_mes.csv and tables/src/data/nbrpop_check_mes.txt (sha256 of both files and the verdict).
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_nbrpop import COLS, LABEL   # noqa: E402  (unmodified module; only constants are used)

OUT = ROOT / 'tables' / 'src' / 'data' / 'hyp_nbrpop_mes.csv'
REF = ROOT / 'tables' / 'hyp_nbrpop.csv'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    rows = json.load(open(sys.argv[sys.argv.index('--raw') + 1], encoding='utf-8'))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(COLS)
        for ds, s, graph, link, q, n, lo, hi, a, b in rows:
            w.writerow([ds, s, 'MSCA', LABEL['mes'], graph, link, q, n, lo, hi, f'{a:.6f}', f'{b:.6f}', f'{100 * (b - a):.4f}'])
    same = OUT.read_bytes() == REF.read_bytes()
    txt = (f'rows {len(rows)}\nregenerated {OUT.name} sha256 {sha(OUT)}\ncommitted  {REF.name} sha256 {sha(REF)}\n'
           f'byte-identical: {same}\n')
    (OUT.parent / 'nbrpop_check_mes.txt').write_text(txt, encoding='utf-8')
    print(txt)
    if not same:
        sys.exit('U1 gate FAILED: regenerated CSV differs from tables/hyp_nbrpop.csv (stop and report, do not overwrite)')


if __name__ == '__main__':
    main()
