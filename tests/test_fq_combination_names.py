"""The names of the rule combinations in the figures of scripts/make_figures.py: one form,
math italic, the form of the manuscript's text, equations and tables.

  * comb_tex sets each of the eight names as mathtext ($N$ ... $LSW$) and refuses any other
    string, so a word or the pooled row cannot be set in italic by mistake;
  * the italic is the italic of the figure's face: a PDF with a name and a word embeds TeX Gyre
    Termes Italic and Regular only, and passes the font gate of the script;
  * the phrase of a policy with two combinations held out keeps its words upright and sets
    only the names in italic (the legend of fq2_heldout); the plain form is the printed one;
  * the codes beside the points of fq4 are drawn in the italic form, while the placement notes
    the run prints keep the plain names.
The tick and row labels of fq1, fq2_reuse and fq2_heldout are checked in the tests of those
figures.

Nothing is read from the repository's records; no display is needed.
Run from the repository root:  python -B -m pytest tests/test_fq_combination_names.py -q
"""
import os
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from conftest import module_environment  # noqa: E402
MODULE_ENVIRONMENT = module_environment(PACT_ENV=None, PACT_SEEDS=None, PACT_OUT=None,
                                        PACT_RESULTS=None)

import make_figures as M  # noqa: E402

NAMES = ['N', 'L', 'S', 'W', 'LS', 'LW', 'SW', 'LSW']


def test_the_eight_names_are_set_in_math_italic():
    assert list(M.COMBOS) == NAMES
    assert [M.comb_tex(c) for c in NAMES] == ['$N$', '$L$', '$S$', '$W$', '$LS$', '$LW$',
                                              '$SW$', '$LSW$']


@pytest.mark.parametrize('text', ['All eight', 'NL', 'lsw', 'LSW ', '', 'policy', '$LSW$'])
def test_any_other_text_is_refused(text):
    with pytest.raises(ValueError, match='is not a rule combination'):
        M.comb_tex(text)


def test_the_italic_is_the_italic_of_the_figure_face(tmp_path):
    if shutil.which('pdffonts') is None:
        pytest.skip('needs pdffonts')
    fig = M.plt.figure(figsize=(2.0, 0.6))
    fig.text(0.05, 0.6, M.comb_tex('LSW'), fontsize=M.FS_TICK)
    fig.text(0.05, 0.2, M.FQH_FILL_TEXT[1], fontsize=M.FS_TICK)
    pdf = str(tmp_path / 'names.pdf')
    fig.savefig(pdf)
    M.plt.close(fig)
    out = subprocess.run(['pdffonts', pdf], capture_output=True, text=True, check=True).stdout
    faces = sorted({ln.split()[0].split('+')[-1] for ln in out.splitlines()[2:] if ln.split()})
    assert faces == ['TeXGyreTermes-Italic', 'TeXGyreTermes-Regular']
    M.assert_times_only([pdf])


def test_the_held_out_phrase_sets_only_the_names_in_italic():
    assert M._fqh_held_text(('LS', 'LSW')) == 'policy, LS and LSW held out'
    assert M._fqh_held_text(('LW', 'LSW'), M.comb_tex) == 'policy, $LW$ and $LSW$ held out'


def test_the_codes_of_fq4_are_drawn_in_math_italic(capsys):
    fig = M.plt.figure(figsize=(3.0, 2.4), dpi=300)
    ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    marks = [dict(x=15.0, y=20.0, ms=4.6, style='public', label='N'),
             dict(x=50.0, y=55.0, ms=4.6, style='public', label='LS'),
             dict(x=75.0, y=80.0, ms=4.6, style='public', label='LSW'),
             dict(x=60.0, y=25.0, ms=4.2, style='factory', label='SW'),
             dict(x=30.0, y=70.0, ms=3.0, style='sens')]          # a point without a code
    for m in marks:
        M._fq4_point(ax, m)
    fig.canvas.draw()
    placed, notes = M._fq4_place_labels(fig, ax, marks, 'point')
    assert sorted(t.get_text() for t in ax.texts) == ['$LS$', '$LSW$', '$N$', '$SW$']
    assert sorted(L['it']['m']['label'] for L in placed) == ['LS', 'LSW', 'N', 'SW']
    assert not any('$' in n for n in notes)
    assert M._fq4_check_marks(fig, ((ax, marks),)) == 4
    M.plt.close(fig)
