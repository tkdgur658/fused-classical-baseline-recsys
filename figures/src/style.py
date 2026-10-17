"""Shared matplotlib style for all figures (fonts, sizes for IEEE two-column: 3.5in / 7.16in).

Run the figure scripts with `python -s` on the Windows machine (the user-site numpy is incompatible with anaconda
matplotlib). Every figure script writes <name>.pdf and <name>.sources.json (input files with sha256) into figures/.
"""
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / 'figures'
TABLES = ROOT / 'tables'
COL = 3.5          # IEEE single column width (in)
DCOL = 7.16        # IEEE double column width (in)
C_OURS = '#1f4e8c'
C_CTRL = '#c0504d'
C_GREY = '#7f7f7f'


def setup():
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'], 'font.size': 7.5,
        'axes.titlesize': 7.5, 'axes.labelsize': 7.5, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
        'legend.fontsize': 6.5, 'axes.linewidth': 0.6, 'lines.linewidth': 1.0, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
    })


def save(fig, name, sources):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f'{name}.pdf')
    meta = {'figure': name, 'script': f'figures/src/{Path(__import__("sys").argv[0]).name}',
            'sources': [{'file': str(Path(s).relative_to(ROOT)).replace('\\', '/'),
                         'sha256': hashlib.sha256(Path(s).read_bytes()).hexdigest()} for s in sources]}
    (FIG / f'{name}.sources.json').write_text(json.dumps(meta, indent=1), encoding='utf-8')
    print('wrote', FIG / f'{name}.pdf')
