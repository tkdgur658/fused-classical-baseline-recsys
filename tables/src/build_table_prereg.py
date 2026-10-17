"""revision log 29 (item 6): preregistration registry of the manuscript, tables/appendix_prereg.tex with two floats:
  tab:prereg      one row per preregistration document: name, date fixed (first ISO date in the first lines of the document), SHA-256 of the
                  archived document (first 16 hex digits; the archived file includes the completion record appended after the runs), what it
                  fixed, and its outcome
  tab:factstatus  one row per fact of the manuscript with its status: preregistered (hypothesis and rule fixed before the runs), table
                  completion (runs and table fixed before the runs, no hypothesis), not registered (the planned comparison of the case study,
                  run before any of these documents), or post hoc (fixed after the results were read)
revision log 29: tab:factstatus is written to tables/table_factstatus.tex (a float of the manuscript).
Every outcome number is parsed from the result files named below or read from the generated macros; the role and status words are the
labels this generator assigns, each tied to a phrase that must occur in the document (asserted), so that a changed document fails the build.
Documents: <PREREG_DIR>/{PREREG_MESF.md, PREREG_MESC.md} and .../study/PREREG_*.md (read-only).
Usage: python -s tables/src/build_table_prereg.py
"""
import hashlib
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from numwords import num_word  # noqa: E402

ROOT = HERE.parent
NOV = Path(r'<PREREG_DIR>')
study = NOV / 'study'
OUT = ROOT / 'appendix_prereg.tex'


def macro(name, files=('macros_mes10.tex', 'macros_classic.tex', 'macros_mesf.tex', 'macros_mesc.tex', 'macros_babybuild.tex',
                       'macros_fusionprose.tex', 'macros_rev2.tex', 'macros_msca10.tex')):
    for f in files:
        m = re.search(r'\\newcommand\{\\' + name + r'\}\{([^}]*)\}', (ROOT / f).read_text(encoding='utf-8'))
        if m:
            return m.group(1)
    raise KeyError(name)


def noise(path):
    m = re.search(r'N_noise = (\d+)/(\d+)', path.read_text(encoding='utf-8'))
    return num_word(m.group(1)) + ' of ' + num_word(m.group(2))


def doc_row(path, name, must, role, outcome):
    t = path.read_text(encoding='utf-8')
    for phrase in must:
        assert phrase in t, (name, phrase)
    head = '\n'.join(t.splitlines()[:6])
    date = re.search(r'20\d\d-\d\d-\d\d', head).group(0)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(name=name, date=date, sha=sha, role=role, outcome=outcome)


def main():
    mes9 = (study / 'results' / 'mes9_verdict.md').read_text(encoding='utf-8')
    m9 = re.search(r'이득 셀 (\d+) / 평가 셀 (\d+)', mes9)
    docs = [
        doc_row(study / 'PREREG_MES9.md', 'MES9', ['H_F9', '0.9'], 'F raises at least 90\\% of the gated cells, eight datasets with MES',
                f'passed, {m9.group(1)} of {m9.group(2)} cells raised'),
        doc_row(study / 'PREREG_R2NULL2.md', 'R2NULL2', ['N2', 'N3'], 'two noise nulls of the headroom N$\'$, earlier design',
                f'noise reading on {noise(study / "results" / "r2null2.md")} datasets'),
        doc_row(study / 'PREREG_R2NULL3.md', 'R2NULL3', ['R2NULL2'], 'the same nulls with MES as the learned ranker',
                f'noise reading on {noise(study / "results" / "r2null3.md")} datasets'),
        doc_row(study / 'PREREG_MESCKPT.md', 'MESCKPT', ['U3', 'U7'], 'checkpoint analyses of MES, rules written after the earlier analyses',
                'descriptive, reported as post hoc'),
        doc_row(NOV / 'PREREG_MESF.md', 'MESF', ['**H_F = FAIL**'], 'fused offset inside training above F on at least three datasets',
                f"failed, above F on {macro('mfNSig')} of four"),
        doc_row(NOV / 'PREREG_MESC.md', 'MESC', ['**H_C = FAIL**'], 'complementarity weighting above F on at least three datasets',
                f"failed, above F on {macro('mcNSig')} of four"),
        doc_row(study / 'PREREG_MES10.md', 'MES10 (A1)', ['AMENDMENT A1'], 'masked view above the full-view control on at least two of four datasets',
                f"{macro('mtVerdict')}, {macro('mtMFRNSig')} of {macro('mtNDs')}"),
        doc_row(study / 'PREREG_CLASSIC.md', 'CLASSIC', ['완료 기록'], 'standalone rankers, mix, and mix fusion as a table completion without hypothesis',
                f"completed, {macro('clNFused')} of {macro('clNUniverse')} fused cells"),
        doc_row(study / 'PREREG_CACHEFIX.md', 'CACHEFIX', ['canonical'], 'build-correction re-run of the Baby cells on the second cache build',
                f"{macro('bbNReplaced')} cells replaced, no mark changed"),
        doc_row(study / 'PREREG_SOTAFILL.md', 'SOTAFILL', ['gate'], 'diagnosis of the fused cells that failed the re-evaluation gate',
                'diagnostic, no new table value'),
    ]
    # revision log 32 (decision ): label unified with the text, 'preregistered, no hypothesis (table completion)'; before: 'measured without a hypothesis, after F ...'
    # revision log 31: MSCA10 outcome from the ten-seed verdict (macros_msca10.tex, build_macros_msca10.py); before: 'pending'
    p10 = study / 'PREREG_MSCA10.md'
    docs.append(doc_row(p10, 'MSCA10', ['TOST', 'Holm'], 'MSCA seeds 6--10 for the fusion contrast at ten seeds, with equivalence and power',
                        f"preregistered, no hypothesis (table completion): after F {macro('xnFNSig')} of {macro('xnNDs')} significant in Recall@20 and {macro('xnFNEquiv')} equivalent, "
                        f"after the mix {macro('xnMixNSigNeg')} below"))
    facts = [
        ('F raises every pair (eight datasets)', 'preregistered', 'MES9'),
        ('F on MicroLens and on further models of Arts and Office', 'table completion', 'CLASSIC'),
        ('Standalone rankers, mix above F, learned models against the mix', 'table completion', 'CLASSIC'),
        ('Mix fusion above F', 'table completion', 'CLASSIC'),
        ('Plug-in counts under the common protocol', 'table completion', 'CLASSIC'),
        ('Protocol sensitivity (supplementary data)', 'table completion', 'CLASSIC'),
        ('Released-code reruns of MG, MDVT, and DA-MRS', 'not registered', '--'),
        ('MES margin over MSCA without fusion', 'not registered', '--'),
        ('MES margin over MSCA without fusion at ten seeds', 'post hoc', '--'),
        ('MES margin over MSCA after F and the mix, power, equivalence at ten seeds', 'table completion', 'MSCA10'),
        ('MES-F and MES-C against F', 'preregistered', 'MESF, MESC'),
        ('Masked view against the full-view control, subtraction arm', 'preregistered', 'MES10 (A1)'),
        ('Weight decay, classical-hit split, neighbor and popularity split', 'post hoc', '--'),
        ('Checkpoint analyses of MES', 'post hoc', 'MESCKPT'),
        ('Headroom N$\'$: noise nulls', 'preregistered', 'R2NULL2, R2NULL3'),
        ('Headroom N$\'$: decoy test, Instruments diagnosis', 'post hoc', '--'),
        ('Cost of the classical fill C', 'post hoc', '--'),
        ('Baby cache build correction', 'preregistered', 'CACHEFIX'),
        ('Fusion of the stored seed-1 MSCA checkpoints of the unfused comparison', 'post hoc', '--'),
    ]
    L = ['% Generated by tables/src/build_table_prereg.py (revision log 29). Do not edit by hand.',
         '\\begin{table*}[t]',
         '\\caption{Preregistration registry. Date: the day the document was fixed before its runs. SHA-256: first 16 hexadecimal digits of the '
         'archived document, which also holds the completion record written after the runs. Each document fixed its runs and its analysis '
         'before any result of these runs was read, MESCKPT repeated on MES analyses whose rules were written after earlier results, '
         'and amendment A1 of MES10 was made after the launch and before any result of the amended arm.}',
         '\\label{tab:prereg}', '\\centering', '\\footnotesize', '\\setlength{\\tabcolsep}{3pt}',
         '\\begin{tabular}{@{}lll p{6.6cm} p{4.4cm}@{}}', '\\toprule',
         'Document & Date & SHA-256 & Fixed before the runs & Outcome \\\\', '\\midrule']
    for d in docs:
        L.append(f"{d['name']} & {d['date']} & \\texttt{{{d['sha'][:16]}}} & {d['role']} & {d['outcome']} \\\\")
    L += ['\\bottomrule', '\\end{tabular}', '\\end{table*}', '']
    OUT.write_text('\n'.join(L), encoding='utf-8', newline='\n')
    # revision log 29: the status table is a float of the manuscript (tables/table_factstatus.tex, cited in Section IV-A)
    L = ['% Generated by tables/src/build_table_prereg.py (revision log 29). Do not edit by hand.',
          '\\begin{table}[t]',
          '\\caption{Status of each fact of the manuscript. Preregistered: hypothesis and rule fixed before the runs. Table completion: runs and '
          'table fixed before the runs, without a hypothesis (preregistered, no hypothesis). Not registered: the planned comparisons, run before any document of '
          'supplementary Table~\\ref{tab:prereg}. Post hoc: fixed after the results were read.}',
          '\\label{tab:factstatus}', '\\centering', '\\footnotesize', '\\setlength{\\tabcolsep}{3pt}',
          '\\begin{tabular}{@{}p{4.6cm}p{1.9cm}p{1.9cm}@{}}', '\\toprule', 'Fact & Status & Document \\\\', '\\midrule']
    names = {d['name'] for d in docs}
    for f, stt, doc in facts:
        assert doc == '--' or all(x.strip() in names for x in doc.split(',')), doc
        L.append(f'{f} & {stt} & {doc} \\\\')
    L += ['\\bottomrule', '\\end{tabular}', '\\end{table}', '']
    (ROOT / 'table_factstatus.tex').write_text('\n'.join(L), encoding='utf-8', newline='\n')
    with open(HERE / 'data' / 'prereg_registry.csv', 'w', encoding='utf-8', newline='\n') as fh:
        fh.write('document,date,sha256\n' + ''.join(f"{d['name']},{d['date']},{d['sha']}\n" for d in docs))
    print('\n'.join(L))


if __name__ == '__main__':
    main()
