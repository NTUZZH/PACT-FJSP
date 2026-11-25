"""scripts/p116_convergence.py on synthetic training logs.

  * the three statistics of one run (late_change, best_at, gain_second_half) equal the values
    computed by hand for a curve with a tie at its minimum;
  * a last validation line that appears twice in identical form is read once, and two different
    lines for the last update are refused;
  * each refusal names the run: an absent log, no completion line and too few validation lines
    are incomplete; another run in the header, another number of updates, an irregular update
    grid and a reported best that is not the minimum of the curve are defective, also under
    --partial;
  * without --partial an incomplete set is refused and nothing is written; with --partial the
    complete runs are reported, the others named, both files go to --out only and the first line
    of the macro file says that it is partial;
  * the macro file holds the expected names, all with the prefix Cvg, signed values with two
    decimals and their sign, shares in percent, and every value follows from the JSON;
  * the specialists trained for 2000 updates (arm long) are read from the logs that
    scripts/longspec_runs.py names; a log of this arm that declares 1000 updates is refused;
    the change after update 1000 equals the value computed by hand for a curve with values at
    the window edges, and the macros CvgLong follow their definitions on a small set of logs
    with known windows; the macros of the three other arms are the same with and without the
    logs of arm long; the raw curves of the JSON are the median, minimum and maximum of the
    runs' scores and the means of the per-combination medians;
  * the figure fs_convergence of scripts/make_figures.py draws the raw curves against the
    updates: panel (a) the eight specialists of 2000 updates as thin lines of one style, their
    mean and the mean of the specialists of 1000 updates (ending at update 1000) as two bold
    lines under a legend of three entries; panel (b) the policy and the control with their
    bands; in both panels the x axis from 0 to 2000 and one dashed line at update 1000; it
    refuses a record without arm long, a partial JSON unless the test-only review switch is set
    (then the file names end in _review and the folder must lie outside the repository) and a
    macro file that disagrees.

Synthetic logs only, written under the pytest temporary folder; nothing is trained, evaluated or
written in the repository.
Run from the repository root:  python -m pytest tests/test_p116_convergence.py -q
"""
import json
import os
import re
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from conftest import module_environment  # noqa: E402
MODULE_ENVIRONMENT = module_environment(PACT_ENV=None, PACT_SEEDS=None, PACT_OUT=None,
                                        PACT_RESULTS=None)

import arm_registry as REG  # noqa: E402
import longspec_runs as LR  # noqa: E402
import p116_convergence as P  # noqa: E402

SEEDS = '301-303'
PY = sys.executable

# A specialist curve of 40 validations (1000 updates) computed by hand:
#   updates   25 ... 500   the first half; its best is 0.9500 at update 250
#   updates  525 ... 600   0.9400
#   updates  625 ... 800   0.9200 (the fifth before the last), mean 0.92
#   updates  825 ... 1000  0.9000, except 0.8900 at 900 and at 950, mean 7.18 / 8 = 0.8975
HAND = ([0.9700 - 0.0010 * k for k in range(9)] + [0.9500]
        + [0.9600 - 0.0005 * k for k in range(10)] + [0.9400] * 4 + [0.9200] * 8
        + [0.9000, 0.9000, 0.9000, 0.8900, 0.9000, 0.8900, 0.9000, 0.9000])
HAND_LATE = 100.0 * (0.8975 / 0.92 - 1.0)            # -2.4456...
HAND_BEST_AT = 900 / 1000
HAND_GAIN = 100.0 * (0.89 / 0.95 - 1.0)              # -6.3157...

# A curve of 80 validations (2000 updates) for the change after update 1000, computed by hand:
#   update  800            0.5000 (left of the window 800 < u <= 1000, so not counted)
#   updates 825 ... 975    0.9000, and 0.9080 at update 1000: mean 7.208 / 8 = 0.9010
#   update  1600           0.6000 (left of the window 1600 < u <= 2000, so not counted)
#   updates 1625 ... 1975  0.8900, and 0.9060 at update 2000: mean 14.256 / 16 = 0.8910
#   every other update     0.9500
HAND_2000 = {800: 0.5000, 1000: 0.9080, 1600: 0.6000, 2000: 0.9060}
HAND_2000.update({u: 0.9000 for u in range(825, 1000, 25)})
HAND_2000.update({u: 0.8900 for u in range(1625, 2000, 25)})
HAND_LONG = [HAND_2000.get(25 * (k + 1), 0.9500) for k in range(80)]
HAND_AFTER = 100.0 * (0.8910 / 0.9010 - 1.0)          # -1.1098...


@pytest.fixture
def setting(monkeypatch, tmp_path):
    monkeypatch.setenv('PACT_ENV', 'exact')
    monkeypatch.setenv('PACT_SEEDS', SEEDS)
    monkeypatch.setenv('PACT_OUT', str(tmp_path / 'out'))
    return tmp_path


def log_text(arm, combo, seed, scores, done=True, duplicate_last=False, best=None,
             header_updates=None, header_model=None, updates=None):
    """A training log in the format of train_family.py."""
    name = P.run_name(arm, combo, seed)
    U = P.UPDATES[arm] if updates is None else updates
    regimes = P.trained_combos(arm, combo)
    out = ['[train_family] MASK: first_mask=exact mask_ties=exact',
           '[vali] MWKR norms: ' + '  '.join(f'{r}=200.0' for r in regimes),
           '[model] params total=32786  film=0 (0.00%)',
           f'[train_family] model={header_model or REG.PREFIX + name} '
           f'regimes={[str(r) for r in regimes]} updates={header_updates or U}']
    run_best = float('inf')
    for k, s in enumerate(scores):
        u = P.VALI_EVERY * (k + 1) if not isinstance(s, tuple) else s[0]
        s = s if not isinstance(s, tuple) else s[1]
        run_best = min(run_best, s)
        out.append(f'Ep {u}\t reward -15.000\t ms 210.0\t loss 0.010000\t 7.0s')
        out.append(f'[vali @{u}] rel={s:.4f} (best {run_best:.4f})  '
                   + '  '.join(f'{r}={200 * s:.1f}' for r in regimes))
    if duplicate_last:
        out.append(out[-1])
    if done:
        b = min(s if not isinstance(s, tuple) else s[1] for s in scores) if best is None else best
        out.append(f'[train_family] done in 1.84 h; best rel={b:.4f}')
    return '\n'.join(out) + '\n'


def write_log(root, arm, combo, seed, scores, **kw):
    p = os.path.join(str(root), P.run_log(arm, combo, seed))
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'w') as f:
        f.write(log_text(arm, combo, seed, scores, **kw))
    return p


def synthetic(arm, combo, seed):
    """A decreasing curve with a late tail whose direction depends on the run; four decimals."""
    U = P.UPDATES[arm]
    n = U // P.VALI_EVERY
    rng = np.random.default_rng(1000 * seed + 17 * (P.R8.index(combo) if combo else 9)
                                + {'control': 3, 'long': 5}.get(arm, 0))
    x = np.arange(1, n + 1) / n
    base = 0.85 + 0.10 * np.exp(-5.0 * x)
    tail = (0.03 if (seed + len(combo or 'xx')) % 2 else -0.03) * np.clip(x - 0.6, 0, None)
    s = base + tail + rng.normal(0, 0.0015, n)
    return [float(f'{v:.4f}') for v in s]


def write_set(root, seeds=(301, 302, 303), skip=(), arms=None):
    for arm, combo in P.arm_combos():
        if arms is not None and arm not in arms:
            continue
        for s in seeds:
            if (arm, combo, s) in skip:
                continue
            write_log(root, arm, combo, s, synthetic(arm, combo, s))


def macros(path):
    return dict(re.findall(r'\\newcommand\{\\(\w+)\}\{([^}]*)\}', open(path).read()))


# ------------------------------------------------------------------ statistics
def test_statistics_on_a_hand_computed_curve():
    assert len(HAND) == 40
    curve = [(25 * (k + 1), s) for k, s in enumerate(HAND)]
    st = P.run_statistics(curve, 1000)
    assert st['late_change'] == pytest.approx(HAND_LATE, abs=1e-9)
    assert st['best_at'] == pytest.approx(HAND_BEST_AT)          # the first of the two minima
    assert st['best_update'] == 900
    assert st['gain_second_half'] == pytest.approx(HAND_GAIN, abs=1e-9)
    assert st['share'][0] == pytest.approx(2.5) and st['share'][-1] == pytest.approx(100.0)
    assert min(st['above_best']) == 0.0
    assert st['above_best'][0] == pytest.approx(100.0 * (0.97 / 0.89 - 1.0))


def test_gain_is_zero_when_the_best_lies_in_the_first_half():
    curve = [(25 * (k + 1), 0.90 if k == 3 else 0.95) for k in range(40)]
    st = P.run_statistics(curve, 1000)
    assert st['gain_second_half'] == 0.0 and st['best_at'] == pytest.approx(0.1)
    assert st['late_change'] == 0.0


def test_log_with_the_hand_curve_and_a_duplicated_last_line(setting):
    p = write_log(setting, 'specialist', 'LS', 302, HAND, duplicate_last=True)
    U, curve, best = P.read_curve(p, 'specialist', 'LS', REG.model('specialist', 'LS', 302))
    assert U == 1000 and len(curve) == 40 and best == pytest.approx(0.89)
    assert curve[-1] == (1000, pytest.approx(0.90))
    st = P.run_statistics(curve, U)
    assert st['late_change'] == pytest.approx(HAND_LATE, abs=1e-9)
    assert st['gain_second_half'] == pytest.approx(HAND_GAIN, abs=1e-9)


def test_two_different_lines_for_the_last_update_are_refused(setting):
    scores = HAND + [(1000, 0.9100)]
    p = write_log(setting, 'specialist', 'LS', 302, scores)
    name = REG.model('specialist', 'LS', 302)
    with pytest.raises(P.Refused, match=f'{name}: .*two different lines for update 1000') as e:
        P.read_curve(p, 'specialist', 'LS', name)
    assert not isinstance(e.value, P.Incomplete)


# ------------------------------------------------------------------ refusals
def _read(root, arm, combo, seed):
    name = P.run_name(arm, combo, seed)
    return P.read_curve(os.path.join(str(root), P.run_log(arm, combo, seed)), arm, combo, name)


def test_absent_log_is_incomplete(setting):
    with pytest.raises(P.Incomplete, match=r'xmask-solo-W-s301: no training log'):
        _read(setting, 'specialist', 'W', 301)


def test_log_without_completion_line_is_incomplete(setting):
    write_log(setting, 'released', None, 303, synthetic('released', None, 303), done=False)
    with pytest.raises(P.Incomplete, match=r'xmask-v1-s303: .* holds no completion line'):
        _read(setting, 'released', None, 303)


def test_log_with_too_few_validation_lines_is_incomplete(setting):
    write_log(setting, 'control', None, 302, synthetic('control', None, 302)[:79])
    with pytest.raises(P.Incomplete, match=r'xmask-ctl-s302: .* holds 79 validation lines, '
                                           r'want 80'):
        _read(setting, 'control', None, 302)


@pytest.mark.parametrize('kind,kw,message', [
    ('best', dict(best=0.8000), r'reports best rel=0\.8000, the minimum of its curve is'),
    ('updates', dict(header_updates=1200), r'declares 1200 updates, the arm trains 1000'),
    ('model', dict(header_model='10x25+family+xmask-solo-S-s301'),
     r'is the log of 10x25\+family\+xmask-solo-S-s301'),
])
def test_defective_log_is_refused_and_named(setting, kind, kw, message):
    write_log(setting, 'specialist', 'N', 301, synthetic('specialist', 'N', 301), **kw)
    with pytest.raises(P.Refused, match=r'xmask-solo-N-s301: .*' + message) as e:
        _read(setting, 'specialist', 'N', 301)
    assert not isinstance(e.value, P.Incomplete)


def test_irregular_update_grid_is_refused(setting):
    s = synthetic('specialist', 'L', 301)
    scores = [(25 * (k + 1) + (5 if k == 10 else 0), v) for k, v in enumerate(s)]
    write_log(setting, 'specialist', 'L', 301, scores)
    with pytest.raises(P.Refused, match=r'xmask-solo-L-s301: .* validates at updates'):
        _read(setting, 'specialist', 'L', 301)


def test_defective_log_is_refused_under_partial_too(setting):
    write_set(setting)
    write_log(setting, 'specialist', 'SW', 302, synthetic('specialist', 'SW', 302), best=0.7)
    out = setting / 'part'
    with pytest.raises(P.Refused, match='xmask-solo-SW-s302'):
        P.run(root=str(setting), partial=True, out_dir=str(out))
    assert not (out / 'macros_convergence.tex').exists()
    assert not (out / 'convergence.json').exists()


def test_wrong_environment_is_refused(setting, monkeypatch):
    monkeypatch.setenv('PACT_ENV', 'legacy')
    with pytest.raises(P.Refused, match='PACT_ENV=exact'):
        P.run(root=str(setting))


# ------------------------------------------------------------------ complete and partial sets
def test_incomplete_set_is_refused_without_partial(setting):
    write_set(setting, skip={('specialist', 'W', 303), ('control', None, 301)})
    write_log(setting, 'control', None, 301, synthetic('control', None, 301), done=False)
    with pytest.raises(P.Refused, match=r'2 of 54 runs are incomplete') as e:
        P.run(root=str(setting))
    assert 'xmask-solo-W-s303' in str(e.value) and 'xmask-ctl-s301' in str(e.value)
    out = setting / 'out'
    assert not (out / P.OUT_FILES['macros']).exists() and not (out / P.OUT_FILES['json']).exists()
    assert P.check(str(setting)) == 1


def test_partial_reports_the_complete_runs_and_names_the_others(setting):
    write_set(setting, skip={('specialist', 'W', 303), ('released', None, 302)})
    out = setting / 'part'
    res, names, outs = P.run(root=str(setting), partial=True, out_dir=str(out))
    assert outs == {'macros': str(out / 'macros_convergence.tex'),
                    'json': str(out / 'convergence.json')}
    assert res['present'] == 52 and res['expected'] == 54 and res['partial']
    assert sorted(m['name'] for m in res['missing']) == ['xmask-solo-W-s303', 'xmask-v1-s302']
    assert res['arms']['released']['n'] == 2 and res['arms']['specialist']['pooled']['n'] == 23
    assert res['arms']['specialist']['combos']['W']['n'] == 2
    first = open(outs['macros']).readline()
    assert first.startswith('% PARTIAL: 52 of 54 runs')
    j = json.load(open(outs['json']))
    assert j['partial'] is True and len(j['runs']) == 52
    assert not (setting / 'out').exists()             # nothing under PACT_OUT
    assert macros(outs['macros'])['CvgSpecRuns'] == '23'


def test_partial_needs_its_own_folder(setting):
    write_set(setting)
    with pytest.raises(P.Refused, match='give --out'):
        P.run(root=str(setting), partial=True)
    with pytest.raises(P.Refused, match='folder of the real outputs'):
        P.run(root=str(setting), partial=True, out_dir=str(setting / 'out' / 'manuscript'))
    with pytest.raises(P.Refused, match='folder of the real outputs'):
        P.run(root=str(setting), partial=True, out_dir=os.path.join(ROOT, 'manuscript'))
    with pytest.raises(P.Refused, match='--out is for --partial only'):
        P.run(root=str(setting), out_dir=str(setting / 'part'))


def test_partial_on_the_command_line(setting):
    write_set(setting, skip={('specialist', 'LSW', 301)})
    out = setting / 'cli'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    r = subprocess.run([PY, '-B', os.path.join(ROOT, 'scripts', 'p116_convergence.py'),
                        '--partial', '--out', str(out), '--root', str(setting)],
                       capture_output=True, text=True, timeout=120, env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert 'p116: 53 of 54 runs' in r.stdout
    assert 'missing: xmask-solo-LSW-s301: no training log' in r.stdout
    assert (out / 'macros_convergence.tex').exists()
    r = subprocess.run([PY, '-B', os.path.join(ROOT, 'scripts', 'p116_convergence.py'),
                        '--root', str(setting)], capture_output=True, text=True, timeout=120,
                       env=env)
    assert r.returncode != 0 and 'REFUSING' in r.stderr and 'xmask-solo-LSW-s301' in r.stderr


# ------------------------------------------------------------------ arm long
def test_arm_long_reads_the_logs_that_longspec_runs_names(setting):
    for c in P.R8:
        for s in (301, 303):
            assert P.run_name('long', c, s) == LR.name(c, s)
            assert LR.name(c, s).endswith(f'-solo-{c}-s{s}') and LR.name(c, s) != \
                REG.model('specialist', c, s)
            assert P.run_log('long', c, s) == LR.train_log(c, s) \
                == f'train_log/FAMILY_{LR.name(c, s)}.log'
    assert P.UPDATES['long'] == 2000 and P.trained_combos('long', 'LS') == ('LS',)
    assert ('long', 'W') in P.arm_combos() and len(P.arm_combos()) == 18
    write_set(setting)
    got, missing = P.collect(str(setting))
    assert not missing and len(got) == 54
    long = [r for r in got if r['arm'] == 'long']
    assert len(long) == 24 and {r['updates'] for r in long} == {2000}
    assert {len(r['curve']) for r in long} == {80}
    assert all(r['log'] == LR.train_log(r['combo'], r['seed']) for r in long)


def test_arm_long_with_1000_updates_is_refused(setting):
    write_log(setting, 'long', 'W', 302, synthetic('specialist', 'W', 302), updates=1000)
    with pytest.raises(P.Refused, match=re.escape(LR.name('W', 302)) + r': .*declares 1000 '
                                        r'updates, the arm trains 2000') as e:
        _read(setting, 'long', 'W', 302)
    assert not isinstance(e.value, P.Incomplete)
    write_set(setting, skip={('long', 'W', 302)})
    out = setting / 'part'
    with pytest.raises(P.Refused, match=re.escape(LR.name('W', 302))):
        P.run(root=str(setting), partial=True, out_dir=str(out))
    assert not (out / 'macros_convergence.tex').exists()


def test_arm_long_unfinished_is_incomplete_and_named(setting):
    write_set(setting, skip={('long', 'SW', 303)})
    write_log(setting, 'long', 'SW', 303, synthetic('long', 'SW', 303)[:50], done=False)
    with pytest.raises(P.Refused, match=r'1 of 54 runs are incomplete') as e:
        P.run(root=str(setting))
    assert LR.name('SW', 303) in str(e.value)
    assert P.check(str(setting)) == 1
    res, _, outs = P.run(root=str(setting), partial=True, out_dir=str(setting / 'part'))
    assert [m['name'] for m in res['missing']] == [LR.name('SW', 303)]
    assert res['arms']['long']['n'] == 23 and res['arms']['long']['combos']['SW']['n'] == 2
    assert res['raw_curves']['long']['mean']['combos'] == list(P.R8)


def test_check_holds_only_when_all_40_runs_of_arm_long_are_complete(setting, monkeypatch, capsys):
    monkeypatch.setenv('PACT_SEEDS', '301-305')
    seeds = (301, 302, 303, 304, 305)
    write_set(setting, seeds=seeds, skip={('long', 'LSW', 305)})
    write_log(setting, 'long', 'LSW', 305, synthetic('long', 'LSW', 305)[:70], done=False)
    assert P.check(str(setting)) == 1
    out = capsys.readouterr().out
    assert 'p116:   specialist of 2000 updates: 39 of 40' in out
    assert LR.name('LSW', 305) in out
    os.remove(os.path.join(str(setting), P.run_log('long', 'LSW', 305)))
    assert P.check(str(setting)) == 1
    assert f'{LR.name("LSW", 305)}: no training log' in capsys.readouterr().out
    write_log(setting, 'long', 'LSW', 305, synthetic('long', 'LSW', 305))
    assert P.check(str(setting)) == 0
    out = capsys.readouterr().out
    assert 'p116: training logs of seeds 301,302,303,304,305: 90 of 90 complete' in out
    assert 'p116:   specialist of 2000 updates: 40 of 40' in out


def test_change_after_1000_on_a_hand_computed_curve(setting):
    assert len(HAND_LONG) == 80
    curve = [(25 * (k + 1), s) for k, s in enumerate(HAND_LONG)]
    assert P.change_after_1000(curve) == pytest.approx(HAND_AFTER, abs=1e-9)
    assert (P.AFTER_REF, P.AFTER_LATE) == ((800, 1000), (1600, 2000))
    p = write_log(setting, 'long', 'N', 301, HAND_LONG)
    U, cv, best = P.read_curve(p, 'long', 'N', P.run_name('long', 'N', 301))
    assert U == 2000 and best == pytest.approx(0.5)
    assert P.change_after_1000(cv) == pytest.approx(HAND_AFTER, abs=1e-9)


# the change after update 1000 of each run of a small set: the mean over 800 < u <= 1000 is 0.9000
# and the mean over 1600 < u <= 2000 is the value below, so the change is 100 (value / 0.9 - 1)
LATE_MEANS = {'N': (0.8820, 0.8910, 0.9036), 'L': (0.9045, 0.9018, 0.9090),
              'S': (0.8955, 0.9018, 0.8964), 'W': (0.8730, 0.8892, 0.8973),
              'LS': (0.9054, 0.8982, 0.9009), 'LW': (0.9036, 0.9027, 0.8991),
              'SW': (0.8946, 0.8937, 0.9063), 'LSW': (0.9009, 0.8991, 0.9072)}


def long_scores(late):
    """80 validations: 0.9500 up to update 800, 0.9000 over 800 < u <= 1000, 0.9200 up to update
    1600 and `late` over 1600 < u <= 2000."""
    return [0.9500 if u <= 800 else 0.9000 if u <= 1000 else 0.9200 if u <= 1600 else late
            for u in range(25, 2001, 25)]


def test_long_macros_follow_their_definitions(setting):
    write_set(setting, arms=('released', 'control', 'specialist'))
    for c, vals in LATE_MEANS.items():
        for s, v in zip((301, 302, 303), vals):
            write_log(setting, 'long', c, s, long_scores(v))
    res, names, outs = P.run(root=str(setting))
    m = macros(outs['macros'])
    chg = {(c, s): 100.0 * (v / 0.9 - 1.0) for c, vals in LATE_MEANS.items()
           for s, v in zip((301, 302, 303), vals)}
    for r in res['runs']:
        if r['arm'] == 'long':
            assert r['change_after_1000'] == pytest.approx(chg[(r['combo'], r['seed'])], abs=1e-9)
    per = {c: float(np.median([chg[(c, s)] for s in (301, 302, 303)])) for c in P.R8}
    for c in P.R8:
        assert m[f'CvgLongMed{c}'] == P.F88.f2(per[c])
    allv = np.array(list(chg.values()))
    assert m['CvgLongMed'] == P.F88.f2(np.median(allv))
    assert m['CvgLongMin'] == P.F88.f2(allv.min()) == '-3.00'          # W, seed 301
    assert m['CvgLongMax'] == P.F88.f2(allv.max()) == '+1.00'          # L, seed 303
    assert (m['CvgLongMinCombo'], m['CvgLongMaxCombo']) == ('W', 'L')
    neg = [c for c in P.R8 if per[c] < 0]
    assert neg == ['N', 'S', 'W', 'SW']        # S: median -0.40; LS +0.10, LW +0.30, LSW +0.10
    assert m['CvgLongNegN'] == '4' and m['CvgLongNegCombos'] == 'N, S, W, SW'
    assert m['CvgLongNegList'] == '$N$, $S$, $W$ and $SW$'
    assert m['CvgLongRuns'] == '24' and m['CvgLongSeeds'] == '3'
    text = open(outs['macros']).read().splitlines()
    assert all('%' in ln for ln in text if 'CvgLong' in ln)            # every line commented
    # the raw curves: per combination the median, minimum and maximum over the seeds, and the
    # mean over the eight combinations of the medians
    g = res['raw_curves']['long']
    assert g['combos']['W']['update'] == list(range(25, 2001, 25))
    assert g['combos']['W']['median'][-1] == pytest.approx(0.8892)
    assert g['combos']['W']['min'][-1] == pytest.approx(0.8730)
    assert g['combos']['W']['max'][-1] == pytest.approx(0.8973)
    assert g['mean']['mean'][-1] == pytest.approx(np.mean([sorted(v)[1]
                                                          for v in LATE_MEANS.values()]))
    assert g['mean']['mean'][0] == pytest.approx(0.95)
    sp = res['raw_curves']['specialist']
    assert sp['mean']['update'][-1] == 1000 and sp['mean']['n'] == 8
    rows = [r for r in res['runs'] if r['arm'] == 'specialist' and r['combo'] == 'LS']
    a = np.array([[s for _, s in r['curve']] for r in rows])
    assert np.allclose(sp['combos']['LS']['median'], np.median(a, axis=0), rtol=0, atol=1e-12)
    pol = [r for r in res['runs'] if r['arm'] == 'released']
    a = np.array([[s for _, s in r['curve']] for r in pol])
    assert np.allclose(res['raw_curves']['released']['max'], a.max(axis=0), rtol=0, atol=1e-12)


def test_other_macros_do_not_depend_on_arm_long(setting, monkeypatch):
    write_set(setting)
    _, _, outs = P.run(root=str(setting))
    full = [ln for ln in open(outs['macros']).read().splitlines()
            if ln.startswith('\\newcommand') and 'CvgLong' not in ln]
    three = [x for x in P.arm_combos() if x[0] != 'long']
    monkeypatch.setattr(P, 'arm_combos', lambda: three)
    _, _, o3 = P.run(root=str(setting), partial=True, out_dir=str(setting / 'three'))
    alone = [ln for ln in open(o3['macros']).read().splitlines() if ln.startswith('\\newcommand')]
    assert full == alone and len(full) == 8


# ------------------------------------------------------------------ macro file
EXPECTED_NAMES = {
    'CvgSpecRuns', 'CvgSeeds',
    'CvgSpecLateMed', 'CvgSpecLateMin', 'CvgSpecLateMax', 'CvgSpecGainMin', 'CvgSpecGainMax',
    'CvgSpecBestAtMed',
    'CvgPolLateMed', 'CvgPolLateMin', 'CvgPolLateMax', 'CvgPolGainMin', 'CvgPolGainMax',
    'CvgPolBestAtMed',
    'CvgCtlLateMed', 'CvgCtlLateMin', 'CvgCtlLateMax', 'CvgCtlGainMin', 'CvgCtlGainMax',
    'CvgCtlBestAtMed',
    'CvgSpecLateMinCombo', 'CvgSpecLateMedLow', 'CvgSpecLateMedLowCombo'}
LONG_NAMES = ({'CvgLongRuns', 'CvgLongSeeds', 'CvgLongMed', 'CvgLongMin', 'CvgLongMax',
               'CvgLongMinCombo', 'CvgLongMaxCombo', 'CvgLongNegN', 'CvgLongNegCombos',
               'CvgLongNegList'} | {f'CvgLongMed{c}' for c in REG.COMBOS})
SIGNED = re.compile(r'[+-]\d+\.\d\d')


def test_macro_file_names_signs_and_values(setting):
    write_set(setting)
    res, names, outs = P.run(root=str(setting))
    assert outs['macros'] == str(setting / 'out' / P.OUT_FILES['macros'])
    text = open(outs['macros']).read().splitlines()
    assert text[0] == '% Generated by scripts/p116_convergence.py, training seeds 301, 302, 303'
    assert not any(x.startswith('% PARTIAL') for x in text)
    m = macros(outs['macros'])
    assert set(m) == EXPECTED_NAMES | LONG_NAMES == set(names)
    assert all(re.fullmatch(r'Cvg[A-Za-z]+', k) for k in m)
    assert m['CvgSpecRuns'] == '24' and m['CvgSeeds'] == '3'
    j = json.load(open(outs['json']))
    for w, key in (('Spec', None), ('Pol', 'released'), ('Ctl', 'control')):
        S = j['arms']['specialist']['pooled'] if key is None else j['arms'][key]
        for stat, word in (('late_change', 'Late'), ('gain_second_half', 'Gain')):
            for q, qw in (('median', 'Med'), ('min', 'Min'), ('max', 'Max')):
                name = f'Cvg{w}{word}{qw}'
                if name not in m:
                    continue
                assert SIGNED.fullmatch(m[name]), (name, m[name])
                assert float(m[name]) == pytest.approx(S[stat][q], abs=0.005 + 1e-12)
                assert m[name][0] == ('-' if S[stat][q] < 0 else '+')
        assert float(m[f'Cvg{w}GainMax']) <= 0.0
        assert float(m[f'Cvg{w}BestAtMed']) == pytest.approx(100 * S['best_at']['median'])
    # every summary is recomputable from the per-run values of the JSON
    spec = [r for r in j['runs'] if r['arm'] == 'specialist']
    late = [r['late_change'] for r in spec]
    assert j['arms']['specialist']['pooled']['late_change']['median'] == \
        pytest.approx(float(np.median(late)))
    low = min(spec, key=lambda r: r['late_change'])
    assert m['CvgSpecLateMinCombo'] == low['combo']
    assert float(m['CvgSpecLateMin']) == pytest.approx(low['late_change'], abs=0.005)
    med = {c: float(np.median([r['late_change'] for r in spec if r['combo'] == c]))
           for c in P.R8}
    c = min(med, key=med.get)
    assert m['CvgSpecLateMedLowCombo'] == c
    assert float(m['CvgSpecLateMedLow']) == pytest.approx(med[c], abs=0.005)
    for r in j['runs']:
        st = P.run_statistics([tuple(x) for x in r['curve']], r['updates'])
        assert st['late_change'] == pytest.approx(r['late_change'])
    assert any(r['late_change'] < 0 for r in spec) and any(r['late_change'] > 0 for r in spec)
    assert P.check(str(setting)) == 0


def test_signed_format_of_small_values():
    assert P.F88.f2(-0.004) == '-0.00' and P.F88.f2(0.0) == '+0.00' and P.F88.f2(0.7) == '+0.70'
    assert P.pct(0.725) == '72.5' and P.pct(0.7125) == '71.25' and P.pct(0.5) == '50'


# ------------------------------------------------------------------ figure
def _lines(ax, M):
    """The data lines of a panel: thin (one specialist), bold (zorder 4 and 5) and dashed."""
    data = [ln for ln in ax.get_lines() if ln.get_zorder() in (3, 4, 5)]
    thin = [ln for ln in data if ln.get_linewidth() == M.FS_CONV_ONE_LW]
    bold = [ln for ln in data if ln.get_linewidth() != M.FS_CONV_ONE_LW]
    dashed = [ln for ln in ax.get_lines() if ln.get_linestyle() == '--']
    return thin, bold, dashed


def test_figure_draws_the_json_and_refuses_disagreement(setting, monkeypatch):
    import make_figures as M
    write_set(setting)
    res, names, outs = P.run(root=str(setting))
    figs = setting / 'figs'
    kept = []
    with monkeypatch.context() as mp:
        mp.setattr(M.plt, 'close', kept.append)
        w = M.fs_convergence(outs['json'], outs['macros'], str(figs))
    assert sorted(os.path.basename(x) for x in w) == ['fs_convergence.pdf', 'fs_convergence.png']
    j = json.load(open(outs['json']))
    R = j['raw_curves']
    ax, bx = kept[0].axes
    # panel (a): the eight specialists of 2000 updates as thin lines of one style, their mean
    # and the mean of the specialists of 1000 updates as two bold lines, a legend of three entries
    assert [t.get_text() for t in ax.get_legend().get_texts()] == [
        'one specialist, 2,000 updates', 'mean of the eight, 2,000 updates',
        'mean of the eight, 1,000 updates']
    thin, bold, dashed = _lines(ax, M)
    assert len(thin) == 8 and len(bold) == 2 and len(dashed) == 1
    assert {(ln.get_color(), ln.get_linestyle()) for ln in thin} == {(M.FS_CONV_ONE_COLOR, '-')}
    for ln, c in zip(thin, [c for grp in M.FS_CONV_SPEC for c in grp]):
        assert np.allclose(ln.get_ydata(), R['long']['combos'][c]['median'], rtol=0, atol=1e-12)
        assert list(ln.get_xdata()) == list(range(25, 2001, 25))
    med = [ln for ln in bold if ln.get_color() == M.FS_CONV_MED_COLOR]
    std = [ln for ln in bold if ln.get_color() == M.FS_CONV_STD_COLOR]
    assert len(med) == 1 and len(std) == 1 and M.FS_CONV_MED_COLOR != M.FS_CONV_STD_COLOR
    long_med = np.array([R['long']['combos'][c]['median'] for c in P.R8])
    assert np.allclose(med[0].get_ydata(), long_med.mean(axis=0), rtol=0, atol=1e-12)
    assert med[0].get_xdata()[-1] == 2000
    std_med = np.array([R['specialist']['combos'][c]['median'] for c in P.R8])
    assert np.allclose(std[0].get_ydata(), std_med.mean(axis=0), rtol=0, atol=1e-12)
    assert std[0].get_xdata()[0] == 25 and std[0].get_xdata()[-1] == 1000
    # panel (b): the policy and the control, median over the seeds and the range as a band
    assert [t.get_text() for t in bx.get_legend().get_texts()] == ['policy',
                                                                   'single-rule control']
    thin_b, bold_b, dashed_b = _lines(bx, M)
    assert not thin_b and len(bold_b) == 2 and len(dashed_b) == 1 and len(bx.collections) == 2
    for ln, arm in zip(bold_b, ('released', 'control')):
        assert np.allclose(ln.get_ydata(), R[arm]['median'], rtol=0, atol=1e-12)
        assert ln.get_xdata()[-1] == 2000
    # both panels: updates 0 to 2000 on the x axis, one dashed line at update 1000, text sizes
    for a in (ax, bx):
        assert a.get_xlim() == (0, pytest.approx(2020))
        assert list(a.get_xticks()) == [0, 500, 1000, 1500, 2000]
        assert [t.get_text() for t in a.get_xticklabels()] == ['0', '500', '1,000', '1,500',
                                                              '2,000']
        d = _lines(a, M)[2][0]
        assert list(d.get_xdata()) == [1000, 1000]
        assert d.get_zorder() < 3
        assert min(t.get_fontsize() for t in a.get_xticklabels() + a.get_yticklabels()) >= 8.0
        top = a.get_ylim()[1]
        assert all(ln.get_ydata()[np.asarray(ln.get_xdata()) >= M.FS_CONV_FROM].max() <= top
                   for ln in _lines(a, M)[0] + _lines(a, M)[1])
    assert 'most-work-remaining' in ax.get_ylabel() and 'MWKR' not in ax.get_ylabel()
    # a disagreeing macro file and a record without the specialists of 2000 updates are refused
    for name in ('CvgPolLateMed', 'CvgLongMedW', 'CvgLongNegList'):
        bad = setting / f'bad_{name}.tex'
        txt = open(outs['macros']).read()
        v = macros(outs['macros'])[name]
        bad.write_text(txt.replace(f'{{\\{name}}}{{{v}}}', f'{{\\{name}}}{{+9.99}}'))
        with pytest.raises(SystemExit, match=name):
            M.fs_convergence(outs['json'], str(bad), str(figs))
    old = dict(j, runs=[r for r in j['runs'] if r['arm'] != 'long'])
    old['raw_curves'] = {k: v for k, v in R.items() if k != 'long'}
    old['arms'] = {k: v for k, v in j['arms'].items() if k != 'long'}
    p_old = setting / 'old.json'
    p_old.write_text(json.dumps(old))
    with pytest.raises(SystemExit, match='trained for 2000 updates'):
        M.fs_convergence(str(p_old), outs['macros'], str(figs))
    no_raw = {k: v for k, v in j.items() if k != 'raw_curves'}
    p_nr = setting / 'no_raw.json'
    p_nr.write_text(json.dumps(no_raw))
    with pytest.raises(SystemExit, match="no key 'raw_curves'"):
        M.fs_convergence(str(p_nr), outs['macros'], str(figs))
    changed = json.loads(json.dumps(j))
    changed['raw_curves']['long']['mean']['mean'][10] += 0.001
    p_ch = setting / 'changed.json'
    p_ch.write_text(json.dumps(changed))
    with pytest.raises(SystemExit, match='mean of arm long'):
        M.fs_convergence(str(p_ch), outs['macros'], str(figs))


def test_figure_of_a_partial_record_is_for_review_only(setting):
    import make_figures as M
    write_set(setting, skip={('long', 'LSW', 303), ('long', 'N', 302)})
    part = setting / 'part'
    _, _, po = P.run(root=str(setting), partial=True, out_dir=str(part))
    figs = setting / 'figs'
    with pytest.raises(SystemExit, match='partial'):
        M.fs_convergence(po['json'], po['macros'], str(figs))
    with pytest.raises(TypeError):
        M.fs_convergence(po['json'], po['macros'], str(figs), True)
    for inside in (os.path.join(ROOT, 'manuscript', 'sections_new', 'figures'),
                   os.path.join(ROOT, 'out', 'review')):
        with pytest.raises(SystemExit, match='outside the repository'):
            M.fs_convergence(po['json'], po['macros'], inside, test_only_partial_review=True)
        assert not os.path.exists(os.path.join(inside, 'fs_convergence_review.png'))
    w = M.fs_convergence(po['json'], po['macros'], str(figs), test_only_partial_review=True)
    assert sorted(os.path.basename(x) for x in w) == ['fs_convergence_review.pdf',
                                                     'fs_convergence_review.png']
    assert not (figs / 'fs_convergence.png').exists()
