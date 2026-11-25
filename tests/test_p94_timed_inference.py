"""Inference of the timed campaign (scripts/p94_timed_inference.py): the reading of a
trajectory at a budget, the sign of each comparison, the matching time, the pooled
row's clustering, the refusals, partial mode and the macro names; on the request clock
the offsets of the rule and policy records of scripts/p97_policy_anytime.py (written by
p97's own record builders and writer), the missing numbers before an offset, the held
values of the sampled policy and comparisons (vii) to (ix); the pooled rows of the
combinations without setups and with setups beside Pool, and that they leave every other
output unchanged; the bold markers of the decision table over its six columns (the two
searches started from the policy's schedule equal to comparison (ii)) and the size of the
solver's model, against a recomputation from the macros and the records, and that they too
leave every other output unchanged; the refusal of a decision row whose columns are means
over different instances, and the macro file of a partial run (--partial-macros). Synthetic
record trees and arrays only; no solver runs and no policy is loaded."""
import csv
import json
import math
import os
import re
import shutil
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'scripts'))
sys.path.insert(0, os.path.join(HERE, '..'))
import p94_timed_inference as M  # noqa: E402
import stats_family as SF  # noqa: E402
from baselines.memetic.records import incumbent_at  # noqa: E402

P97 = M.P97

SIZE = '50x25n30'
N_DESC = 30
EVID = list(range(3, 30))
SEEDS = (301, 302, 303)
SEL = 302
ARMS4 = 'memetic_rule@600 memetic_policy@600 cpsat_rule@600 cpsat_policy@600'
SLOT_A = list(range(8, 16))


def r0(i, j):
    """Best-rule makespan of instance i in combination j."""
    return 500.0 + i + 3 * j


def pol(s, i, j):
    """Greedy makespan of training run s: 20 below the best rule, the selected run exactly."""
    return r0(i, j) - 20 + (s - SEL)


# ---------------------------------------------------------------------------
# a synthetic campaign

def write_world(root):
    """Descriptors, rule and policy arrays and the records of the four arms at 50x25n30."""
    data, res, rr = root / 'data', root / 'res', root / 'rr'
    for j, r in enumerate(M.COMBOS):
        d = data / SIZE / 'desc' / r
        d.mkdir(parents=True)
        for i in range(N_DESC):
            json.dump({'base_instance': f'base/instance_{i:03d}', 'regime': r},
                      open(d / f'instance_{i:03d}.desc.json', 'w'))
    out = res / f'{SIZE}_xmask'
    out.mkdir(parents=True)
    base = np.array([[r0(i, j) for j in range(8)] for i in range(N_DESC)])
    for j, r in enumerate(M.COMBOS):
        for q, off in (('FIFO', 40), ('MOR', 60), ('SPT', 0), ('MWKR', 20)):
            np.save(out / f'Result_{q}_{r}.npy', np.stack([base[:, j] + off,
                                                          np.full(N_DESC, 0.1)], axis=1))
        for s in SEEDS:
            np.save(out / f'Result_greedy+10x25+family+xmask-v1-s{s}_{r}.npy',
                    np.stack([[pol(s, i, j) for i in range(N_DESC)], np.full(N_DESC, 0.5)], axis=1))
    a = M.build_parser().parse_args(['--result-root', str(rr), '--bars', str(root / 'bars.json')])
    camp = M.make_campaign([SIZE], ARMS4.split(), a, SEL, 'ref_fullhint')
    for seg in camp.segments:
        for c in camp.cells(seg):
            write_record(camp, c, M.COMBOS.index(c.combo))
    return root


def n_vars(i, j, start='rule'):
    """Variables of the initial model of instance i in combination j: the sequence variables
    of the setups make the model large; a policy-started record states one more, so that a
    count read from the wrong arm shows."""
    return ((560000 if M.COMBOS[j] in M.P.SETUP else 19000) + 1000 * i + 17 * j
            + (0 if start == 'rule' else 1))


def peak_gb(i, j, start='rule'):
    """Peak memory of the solve in GB, larger with setups."""
    return round((19.0 if M.COMBOS[j] in M.P.SETUP else 1.0) + 0.1 * i + 0.01 * j
                 + (0 if start == 'rule' else 0.5), 3)


def solver_log(n, presolved=None):
    """The head of a CP-SAT log as p85 stores it: the initial model's variable count after the
    line that starts 'Initial optimization model', then the presolved model's count."""
    q = lambda x: f'{x:,}'.replace(',', "'")  # noqa: E731
    presolved = n // 3 if presolved is None else presolved
    return ['', 'Starting CP-SAT solver v9.14.6206',
            'Parameters: random_seed: 1 max_time_in_seconds: 600 num_search_workers: 8', '',
            "Initial optimization model '': (model_fingerprint: 0x8936fd9b3bdca7d1)",
            f'#Variables: {q(n)} (#ints: 1 in objective) ({q(n - 3301)} primary variables)',
            f'  - {q(n - 1200)} Booleans in [0,1]', '#kCircuit: 25', '',
            'Starting presolve at 0.22s', '',
            "Presolved optimization model '': (model_fingerprint: 0x2c1d0a7e5f3b9d11)",
            f'#Variables: {q(presolved)} (#ints: 1 in objective) ({q(presolved)} primary '
            f'variables)']


def write_record(camp, c, j, **kw):
    R0, PS = r0(c.idx, j), pol(SEL, c.idx, j)
    rec = dict(camp.expected_fields(c))
    if c.seg.runner == 'memetic':
        rec.update(validated=True, threads=1, workers=1, cpu_share=0.999, affinity=[SLOT_A[0]],
                   search_start_wall_s=1.0, search_start_cpu_s=1.0)
        if c.seg.start == 'rule':
            rec.update(seeds=[dict(source='warm_cache', makespan=R0)], makespan=R0 - 40,
                       anytime=[[0.0, 1.0, 1.0, R0], [1.0, 2.0, 2.0, R0 - 10],
                                [29.0, 30.0, 30.0, R0 - 30], [49.0, 50.0, 50.0, R0 - 40]])
        else:
            rec.update(seeds=[dict(source='warm_cache', makespan=R0),
                              dict(source='seed_from', makespan=PS)], makespan=PS - 10,
                       anytime=[[0.0, 1.0, 1.0, PS], [4.0, 5.0, 5.0, PS - 10]])
    else:
        rec.update(hint='complete', validated=True, status='FEASIBLE', cpu_share=0.95,
                   affinity=SLOT_A, build_s=5.0, peak_rss_gb=peak_gb(c.idx, j, c.seg.start),
                   solver_log_head=solver_log(n_vars(c.idx, j, c.seg.start)))
        if c.seg.start == 'rule':
            rec.update(warm_makespan=R0, objective=R0 - 25,
                       anytime=[[3.0, R0 - 5, 100.0], [50.0, R0 - 25, 100.0]])
        else:
            rec.update(warm_makespan=PS, objective=PS - 2, anytime=[[10.0, PS - 2, 100.0]])
    rec.update(kw)
    p = camp.record(c)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(rec, open(p, 'w'))
    return p


# ---------------------------------------------------------------------------
# the records of p97 (request clock): rule arm and policy arm of every cell

POL_BUDGET, POL_BATCH, POL_WORKERS = 30.0, 2, 2
POL_TOKEN = f'{M.POLICY_ARM}@30'
LATE = ('LS', 7)                    # the cell whose greedy schedule comes at 12 s
RULE_OFF = {'FIFO': 40, 'MOR': 60, 'SPT': 0, 'MWKR': 20}       # as the rule arrays
IDLE = {'idle': True, 'busy_processes': []}


def o_rule(i, j):
    """t_all of the rule record: the MWKR rollout ends last."""
    return 1.5 + 0.05 * j


def o_pol(i, j):
    """t of the greedy schedule in the policy record."""
    return 12.0 if (M.COMBOS[j], i) == LATE else 6.0 + 0.1 * (i % 5)


def sampled_at(i, j, t):
    """Best schedule so far of the policy record at time t (None before the greedy one),
    held after the budget: the greedy G at o_pol, G - 3 two seconds later, G - 8 at 20 s,
    G - 9 at 25 s."""
    g, og = pol(SEL, i, j), o_pol(i, j)
    vals = [v for tt, v in ((og, g), (og + 2, g - 3), (20.0, g - 8), (25.0, g - 9))
            if tt <= t + 1e-9]
    return min(vals) if vals else None


def stub_validator(spec, amch, ct):
    ct = np.asarray(ct, dtype=float)
    return [], (np.asarray(amch), ct - 1, ct)


def with_reports(msgs, build, spec):
    """p97's record builder on reports in place of its worker processes."""
    real = P97.start_workers
    P97.start_workers = lambda *a, **k: (msgs, [], [m['core'] for m in msgs.values()])
    try:
        return build(spec, validator=stub_validator)
    finally:
        P97.start_workers = real


def policy_reports(i, j, share=0.97):
    """Reports of the two workers of a policy cell: worker 0 the greedy schedule, then a
    sampled batch at 20 s; worker 1 sampled batches at o_pol + 2 and at 25 s (batches of
    POL_BATCH rows)."""
    g, og = pol(SEL, i, j), o_pol(i, j)
    plan = {0: [(0, og, [g], 'greedy'), (1, 20.0, [g - 8, g - 5], 'sampled')],
            1: [(0, og + 2, [g - 1, g - 3], 'sampled'), (1, 25.0, [g - 9, g - 6], 'sampled')]}
    msgs = {}
    for w, batches in plan.items():
        ready = round(og - 4 + 0.1 * w, 4)
        rows, improving, best = [], [], np.inf
        for k, t, ms, kind in batches:
            seed = None if kind == 'greedy' else P97.rollout_seed(P97.DEFAULT_BASE_SEED, w, k)
            rows += [[w, k, kind, seed, t, round(t - ready, 4), float(m), round(t, 3), r]
                     for r, m in enumerate(ms)]
            r = int(np.argmin(ms))
            if ms[r] < best:
                best = ms[r]
                improving.append(dict(index=k, row=r, makespan=float(ms[r]),
                                      amch=np.zeros(3, dtype=int), ct=np.full(3, float(ms[r]))))
        msgs[w] = dict(worker=w, core=8 + w, affinity=[8 + w], cpu_share=share, ready_s=ready,
                       stop_s=POL_BUDGET, peak_rss_mb=500.0, n_batches=len(batches),
                       n_schedules=len(rows), rows=rows, improving=improving)
    return msgs


def rule_reports(i, j):
    """Reports of the four workers of a rule cell, one rule each; MWKR ends at o_rule."""
    msgs = {}
    for w, q in enumerate(P97.RULE_ORDER):
        t = (0.9, 1.1, 1.3, o_rule(i, j))[w]
        ms = float(r0(i, j) + RULE_OFF[q])
        msgs[w] = dict(worker=w, core=8 + w, affinity=[8 + w], cpu_share=0.97, ready_s=0.5,
                       stop_s=t, peak_rss_mb=200.0, check_s=0.01,
                       rules=[dict(rule=q, makespan=ms, t=t, t_resident=round(t - 0.5, 4),
                                   cpu_s=round(t - 0.5, 4), wall_s=round(t - 0.5, 4), share=0.99,
                                   worker=w, core=8 + w, cpu_s_worker=t,
                                   amch=np.zeros(3, dtype=int), ct=np.full(3, ms), eligible=True,
                                   c6='pass')])
    return msgs


def rule_cache(i, j):
    """The stored rule start schedule p97.classify_rules compares with."""
    return dict(file='synthetic rule cache', rule='SPT', ms=float(r0(i, j)),
                rule_ms={q: float(r0(i, j) + RULE_OFF[q]) for q in P97.RULE_ORDER})


MODEL = M.REG.PREFIX + M.REG.model('released', None, SEL, environment='exact')


def write_policy_record(root, i, j, share=0.97, greedy=None, overwrite=False):
    """One policy record by p97.run_cell and p97.write_record (validity by p97.classify
    against the stored greedy value, as p97's cell process does)."""
    r = M.COMBOS[j]
    spec = dict(model=MODEL, checkpoint='-', config='-', training_seed=SEL, environment='exact',
                mask={'first_mask': 'exact', 'mask_ties': 'exact'}, mask_check='-', size=SIZE,
                combination=r, instance=i, budget=POL_BUDGET, workers=POL_WORKERS,
                batch=POL_BATCH, base_seed=P97.DEFAULT_BASE_SEED,
                affinity=[8 + w for w in range(POL_WORKERS)], development=False,
                idle_before=IDLE)
    msgs = policy_reports(i, j, share)
    if greedy is not None:
        for m in msgs.values():
            for row in m['rows']:
                if row[2] == 'greedy':
                    row[6] = greedy
            for s in m['improving']:
                if s['index'] == 0 and m['worker'] == 0:
                    s.update(makespan=greedy, ct=np.full(3, greedy))
    rec, sched, errors = with_reports(msgs, P97.run_cell, spec)
    assert not errors, errors
    stored = {'synthetic greedy array': pol(SEL, i, j)}
    rec['greedy_stored'], rec['greedy_stored_notes'] = stored, []
    rec['valid'], rec['valid_reason'] = P97.classify(rec, stored=stored)
    p = P97.record_path(str(root / 'rr'), SIZE, POL_BUDGET, POL_BATCH, f's{SEL}', r, i)
    if overwrite and os.path.exists(p):
        os.remove(p)
    assert P97.write_record(p, rec)
    return p


def write_rule_record(root, i, j, overwrite=False):
    """One rule record by p97.run_rule_cell and p97.write_record."""
    r = M.COMBOS[j]
    spec = dict(size=SIZE, combination=r, instance=i, workers=P97.RULE_WORKERS,
                rules=P97.RULE_ORDER, t0=0.0, budget=0.0, environment='exact',
                affinity=[8, 9, 10, 11], development=False, idle_before=IDLE)
    rec, sched, errors, raw = with_reports(rule_reports(i, j), P97.run_rule_cell, spec)
    assert not errors, errors
    cache = rule_cache(i, j)
    rec['stored'] = {k: v for k, v in cache.items()}
    rec['stored_schedule_equal'] = None
    rec['valid'], rec['valid_reason'] = P97.classify_rules(rec, cache)
    p = P97.rule_record_path(str(root / 'rr'), SIZE, r, i)
    if overwrite and os.path.exists(p):
        os.remove(p)
    assert P97.write_record(p, rec)
    return p


def write_request_records(root):
    for i in EVID:
        for j in range(len(M.COMBOS)):
            write_rule_record(root, i, j)
            write_policy_record(root, i, j)
    return root


@pytest.fixture(scope='module')
def base_world(tmp_path_factory):
    return write_world(tmp_path_factory.mktemp('world'))


@pytest.fixture(scope='module')
def base_request_world(base_world, tmp_path_factory):
    root = tmp_path_factory.mktemp('rworld') / 'w'
    shutil.copytree(base_world, root)
    return write_request_records(root)


def enter(src, dst, monkeypatch):
    shutil.copytree(src, dst)
    monkeypatch.chdir(M.ROOT)
    monkeypatch.setenv('PACT_ENV', 'exact')
    monkeypatch.setenv('PACT_SEEDS', '301-303')
    monkeypatch.setenv('PACT_RESULTS', str(dst / 'res'))
    monkeypatch.setattr(M, 'select_policy_seed',
                        lambda: (SEL, {301: (0.9, 10), 302: (0.8, 12), 303: (0.85, 5)}))
    monkeypatch.setattr(M, 'select_cpsat_arm', lambda *a: ('ref_fullhint', []))
    # the stored values p97's validity compares with, here the synthetic ones
    monkeypatch.setattr(P97, 'stored_greedy', lambda model, seed, size, combo, idx: (
        {'synthetic greedy array': pol(SEL, idx, M.COMBOS.index(combo))}, []))
    monkeypatch.setattr(P97, 'rule_cache_entry',
                        lambda size, combo, idx: rule_cache(idx, M.COMBOS.index(combo)))
    return dst


@pytest.fixture
def world(base_world, tmp_path, monkeypatch):
    return enter(base_world, tmp_path / 'w', monkeypatch)


@pytest.fixture
def rworld(base_request_world, tmp_path, monkeypatch):
    return enter(base_request_world, tmp_path / 'w', monkeypatch)


def args(root, *extra, arms=ARMS4, partial=False, origin='start'):
    out = ['--campaign', f'{SIZE} = {arms}', '--budgets', '1', '10', '60', '--primary', '60',
           '--boot', '200',
           '--result-root', str(root / 'rr'), '--data-root', str(root / 'data'),
           '--bars', str(root / 'bars.json')]
    if origin is not None:
        out += ['--origin', origin]
    if partial:
        out += ['--partial', '--out', str(root / 'partial')]
    else:
        out += ['--out-root', str(root / 'out')]
    return out + list(extra)


def rargs(root, *extra, arms=f'{ARMS4} {POL_TOKEN}', partial=False, origin='request'):
    """The request clock: the policy arm's token and the plan of its records."""
    return args(root, '--policy-batch', str(POL_BATCH), '--policy-workers', str(POL_WORKERS),
                *extra, arms=arms, partial=partial, origin=origin)


def macros(root):
    txt = open(root / 'out' / 'manuscript' / 'macros_timed.tex').read()
    return dict(re.findall(r'\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}', txt)), txt


def camp_of(root, arms=ARMS4):
    a = M.build_parser().parse_args(['--result-root', str(root / 'rr'),
                                     '--bars', str(root / 'bars.json')])
    return M.make_campaign([SIZE], arms.split(), a, SEL, 'ref_fullhint')


def cell(camp, arm, combo, idx):
    seg = next(s for s in camp.segments if s.arm == arm)
    return next(c for c in camp.cells(seg) if c.combo == combo and c.idx == idx)


# ---------------------------------------------------------------------------
# reading a trajectory at a budget

def test_reading_before_between_and_after_the_incumbents():
    mem = dict(budget_s=600.0, seeds=[dict(source='warm_cache', makespan=520)], makespan=500,
               search_start_wall_s=1.0, search_start_cpu_s=1.0,
               anytime=[[0.0, 1.0, 1.0, 520], [2.0, 3.0, 3.0, 510], [6.0, 7.0, 7.0, 500]])
    g = (0.5, 5.0, 100.0)
    assert M.read_record(mem, 'memetic', g, 'start') == {0.5: 520.0, 5.0: 510.0, 100.0: 500.0}
    # from the search start the same events come one second earlier
    assert M.read_record(mem, 'memetic', (1.9, 2.0, 6.5), 'search') == \
        {1.9: 520.0, 2.0: 510.0, 6.5: 500.0}
    cp = dict(budget_s=600.0, arm='ref_fullhint', warm_makespan=520.0, build_s=30.0,
              objective=512.0, anytime=[[45.0, 521.0, 300.0], [70.0, 512.0, 300.0]])
    # start: the model build comes first; before the first incumbent the hint counts, an
    # incumbent above the hint does not replace it, after the last one it holds
    assert M.read_record(cp, 'cpsat', (10.0, 60.0, 80.0, 100.0, 500.0), 'start') == \
        {10.0: 520.0, 60.0: 520.0, 80.0: 520.0, 100.0: 512.0, 500.0: 512.0}
    assert M.read_record(cp, 'cpsat', (44.0, 69.0, 70.0), 'search') == \
        {44.0: 520.0, 69.0: 520.0, 70.0: 512.0}
    # the neighbourhood-search arm builds inside its budget: no offset from the start
    lns = dict(cp, arm='lns', anytime=[[0.0, 520.0, None], [40.0, 505.0, None]])
    assert M.read_record(lns, 'cpsat', (39.0, 40.0), 'start') == {39.0: 520.0, 40.0: 505.0}
    # beyond the budget there is no reading
    assert 700.0 not in M.read_record(dict(cp, budget_s=600.0), 'cpsat', (700.0,), 'search')


# ---------------------------------------------------------------------------
# matching time and clustering

def test_matching_time_against_a_hand_computed_case():
    grid = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    traj = np.array([[110, 120, 130],      # t = 0
                     [105, 120, 130],
                     [100, 118, 130],      # cell 0 reaches 100 at t = 2
                     [100, 110, 130],      # cell 1 reaches 110 at t = 3
                     [100, 110, 130],
                     [100, 110, 125]], dtype=float)
    pol = np.array([100.0, 110.0, 120.0])  # cell 2 never reaches 120
    x = M.match_one(traj, pol, grid)
    # means: 120, 118.3, 116, 113.3, 113.3, 111.7 against the policy's 110: never
    assert x['mean_s'] is None and x['never_pct'] == pytest.approx(100 / 3)
    assert x['per_cell_s'] == [2.0, 3.0, None] and x['median_s'] == 3.0
    x = M.match_one(traj, np.array([100.0, 110.0, 135.0]), grid)
    # means 120, 118.3, 116, 113.3 ... against 115: first at t = 3; cell 2 at t = 0
    assert x['mean_s'] == 3.0 and x['per_cell_s'] == [2.0, 3.0, 0.0] and x['median_s'] == 2.0
    x = M.match_one(traj[:, :2], np.array([90.0, 100.0]), grid)
    assert x['mean_s'] is None and x['median_s'] is None and x['never_pct'] == 100.0


def test_pooled_row_keeps_the_combinations_of_a_base_instance_together():
    rng = np.random.default_rng(3)
    n, c = 12, 8
    inst = rng.normal(0, 2, n)[:, None]                 # a shared instance effect
    first = 100 + inst + rng.normal(0, 0.1, (n, c))
    second = np.full((n, c), 100.0)
    pool = M.compare(first, second, list(M.COMBOS), 500, 0.05)['Pool']
    d = M.gap(first, second)
    want = SF.instance_boot(d.mean(axis=1), B=500, seed=7)
    assert (pool['lo'], pool['hi'], pool['n'], pool['c']) == (want['lo'], want['hi'], n, c)
    cells = SF.instance_boot(d.ravel(), B=500, seed=7)  # cells treated as independent
    assert cells['hi'] - cells['lo'] < 0.7 * (pool['hi'] - pool['lo'])
    k = 3
    first3 = first[None] + rng.normal(0, 0.5, (k, 1, 1))
    hier = M.compare(first3, second, list(M.COMBOS), 300, 0.05, seeds=True)['Pool']
    want = SF.hier_boot(M.gap(first3, second), B=300, seed=11)
    assert (hier['lo'], hier['hi'], hier['k'], hier['c']) == (want['lo'], want['hi'], k, c)
    # a missing value removes the instance from the pooled row only
    first[0, M.COMBOS.index('LS')] = np.nan
    rows = M.compare(first, second, list(M.COMBOS), 200, 0.05)
    assert rows['Pool']['n'] == n - 1 and rows['LS']['n'] == n - 1 and rows['N']['n'] == n


# ---------------------------------------------------------------------------
# the full run on a synthetic campaign

def test_full_run_signs_matching_times_and_macro_names(world):
    assert M.main(args(world)) == 0
    mac, txt = macros(world)
    # (i) policy minus rule-started arm: ahead at 1 s (arm still at the rule schedule),
    # behind at 60 s (memetic at R0 - 40 against the policy at R0 - 20)
    assert float(mac['TmdFiftyPolVsMemOneNSel']) < 0 and mac['TmdFiftyPolVsMemOneNSelDir'] == 'ahead'
    assert float(mac['TmdFiftyPolVsMemSixtyPoolSel']) > 0
    assert mac['TmdFiftyPolVsMemSixtyPoolSelDir'] == 'behind'
    assert mac['TmdFiftyPolVsCpOnePoolAllDir'] == 'ahead'
    assert mac['TmdFiftyPolVsMemOnePoolSelShorter'] == '100.0'
    # (ii) policy-started arm minus the policy it started from
    assert mac['TmdFiftyMemPolStartVsPolOnePool'] == '+0.00'
    assert mac['TmdFiftyMemPolStartVsPolOnePoolDir'] == 'neither'
    assert float(mac['TmdFiftyMemPolStartVsPolTenPool']) < 0
    assert mac['TmdFiftyCpPolStartVsPolSixtyPoolDir'] == 'ahead'
    # (iii) policy-started minus rule-started: ahead at 10 s, behind at 60 s
    assert mac['TmdFiftyMemPolStartVsRuleStartTenPoolDir'] == 'ahead'
    assert mac['TmdFiftyMemPolStartVsRuleStartSixtyPoolDir'] == 'behind'
    # (iv) best rule minus policy: the rule is longer on every cell
    assert float(mac['TmdFiftyRuleVsPolPoolSel']) > 0 and mac['TmdFiftyRuleVsPolPoolSelDir'] == 'behind'
    assert mac['TmdFiftyRuleVsPolPoolSelLonger'] == '100.0'
    # (v) memetic below the policy from wall 30 s of the process, CP-SAT from 5 + 50 s
    assert mac['TmdFiftyMemMatchMeanN'] == '30' and mac['TmdFiftyMemMatchMedianPool'] == '30'
    assert mac['TmdFiftyCpMatchMeanLSW'] == '55' and mac['TmdFiftyCpMatchNeverPool'] == '0.0'
    assert mac['TmdSelSeed'] == str(SEL) and mac['TmdFiftyInstN'] == '27'
    # names: valid LaTeX control words, unique, prefix Tmd; signed numbers as p89's f2
    names = re.findall(r'\\newcommand\{\\([^}]*)\}', txt)
    assert len(names) == len(set(names)) > 500
    assert all(re.fullmatch(r'Tmd[A-Za-z]+', n) for n in names)
    signed = [(n, v) for n, v, cmt in re.findall(r'\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}  % (.*)', txt)
              if 'mean difference' in cmt or 'end of the 95% interval' in cmt]
    assert len(signed) > 300 and all(re.fullmatch(r'[+-]\d+\.\d\d', v) for _, v in signed)
    assert all(v == M.f2(v) for _, v in signed)
    assert txt.startswith('% Generated by scripts/p94_timed_inference.py -- do not hand-edit')
    assert 'Clock origin start' in txt
    out = json.load(open(world / 'out' / 'results' / 'timed_inference.json'))
    assert out['run'] == 'full' and out['partial'] is False and out['origin'] == 'start'
    assert out['counts'][f'cpsat_rule|{SIZE}|N'] == {'valid': 27}
    assert len(out['records']) == 4 * 27 * 8
    csv_txt = open(world / 'out' / 'results' / 'timed_curves.csv').read().splitlines()
    assert csv_txt[0].startswith('run,row,origin,policy,size') and csv_txt[0].endswith(',n_runs')
    assert csv_txt[1].startswith('full,Sel,start,')


def test_search_origin_moves_the_matching_times(world):
    assert M.main(args(world, origin='search')) == 0
    mac, txt = macros(world)
    assert mac['TmdFiftyMemMatchMeanN'] == '29' and mac['TmdFiftyCpMatchMeanN'] == '50'
    assert 'Clock origin search' in txt


# ---------------------------------------------------------------------------
# refusals

def refused(world, capsys, *extra, arms='memetic_rule@600 cpsat_rule@600', partial=False):
    mac = world / 'out' / 'manuscript' / 'macros_timed.tex'
    mac.parent.mkdir(parents=True, exist_ok=True)
    mac.write_text('% stale\n')
    rc = M.main(args(world, *extra, arms=arms, partial=partial))
    err = capsys.readouterr().err
    assert rc == 1 and 'REFUSING' in err
    if not partial:
        assert not mac.exists()                          # no stale macro survives
    return err


def test_refuses_a_cell_without_a_record(world, capsys):
    camp = camp_of(world)
    os.remove(camp.record(cell(camp, 'memetic_rule', 'LS', 7)))
    err = refused(world, capsys)
    assert 'memetic_rule@600/50x25n30/LS/007: missing' in err


def test_refuses_a_pending_cpsat_record(world, capsys):
    camp = camp_of(world)
    c = cell(camp, 'cpsat_rule', 'S', 4)
    write_record(camp, c, M.COMBOS.index('S'), cpu_share=0.5)
    err = refused(world, capsys)
    assert 'cpsat_rule@600/50x25n30/S/004: pending' in err


def test_refuses_missing_and_misshapen_arrays(world, capsys):
    os.remove(world / 'res' / f'{SIZE}_xmask' / 'Result_greedy+10x25+family+xmask-v1-s303_W.npy')
    err = refused(world, capsys)
    assert 'missing policy array of seed 303' in err
    np.save(world / 'res' / f'{SIZE}_xmask' / 'Result_SPT_N.npy', np.zeros((29, 2)) + 1)
    err = refused(world, capsys, partial=True)            # a wrong shape refuses in partial mode too
    assert 'has shape (29, 2), want (30, 2)' in err


def test_refuses_records_and_arrays_of_different_instances(world, capsys):
    camp = camp_of(world)
    c = cell(camp, 'cpsat_rule', 'SW', 12)
    write_record(camp, c, M.COMBOS.index('SW'), warm_makespan=r0(13, M.COMBOS.index('SW')))
    err = refused(world, capsys, partial=True)
    assert 'rule start schedule' in err and 'instance 12, SW' in err


def test_refuses_a_policy_start_that_is_not_the_selected_run(world, capsys):
    camp = camp_of(world)
    c = cell(camp, 'memetic_policy', 'N', 3)
    j = M.COMBOS.index('N')
    write_record(camp, c, j, seeds=[dict(source='warm_cache', makespan=r0(3, j)),
                                    dict(source='seed_from', makespan=pol(303, 3, j))])
    err = refused(world, capsys, arms=ARMS4)
    assert 'policy start schedule' in err


def test_refuses_combinations_of_different_base_instances(world, capsys):
    p = world / 'data' / SIZE / 'desc' / 'LW' / 'instance_009.desc.json'
    json.dump({'base_instance': 'base/instance_010', 'regime': 'LW'}, open(p, 'w'))
    err = refused(world, capsys)
    assert 'instance_009 is not one base instance' in err


def test_an_instance_range_is_the_design_of_its_arm(world, capsys):
    """A token with an instance range plans its arm on those instances: the full mode reads
    the arm there and states the design (JSON, Markdown header, count macros); a token
    without a range plans the evidence instances; the check that every cell of the plan has a
    valid record stays, and a record outside the range is not read."""
    camp = camp_of(world)
    os.remove(camp.record(cell(camp, 'memetic_rule', 'LS', 25)))       # outside 3-20
    assert M.main(args(world, arms='memetic_rule@600:3-20 cpsat_rule@600')) == 0
    J = json.load(open(world / 'out' / 'results' / 'timed_inference.json'))
    assert J['gaps'] == [] and J['cells_valid'] == J['cells_total'] == 18 * 8 + 27 * 8
    assert J['results'][SIZE]['design'] == {
        'memetic_rule': dict(instances=list(range(3, 21)), range_given=True),
        'cpsat_rule': dict(instances=EVID, range_given=False)}
    assert J['results'][SIZE]['i']['memetic']['60']['selected']['Pool']['n'] == 18
    assert J['results'][SIZE]['i']['cpsat']['60']['selected']['Pool']['n'] == 27
    mac, txt = macros(world)
    assert (mac['TmdFiftyInstN'], mac['TmdFiftyMemRuleStartInstN'],
            mac['TmdFiftyCpRuleStartInstN']) == ('27', '18', '27')
    line = next(x for x in txt.splitlines() if '{\\TmdFiftyMemRuleStartInstN}' in x)
    assert '(3-20, the range of its token)' in line and 'the same on every clock' in line
    md = open(world / 'out' / 'results' / 'timed_inference.md').read()
    assert f'{SIZE} memetic_rule 3-20 (18), cpsat_rule 3-29 (27)' in md
    # inside the range, a missing record still refuses
    os.remove(camp.record(cell(camp, 'memetic_rule', 'LS', 7)))
    err = refused(world, capsys, arms='memetic_rule@600:3-20 cpsat_rule@600')
    assert 'memetic_rule@600/50x25n30/LS/007: missing' in err
    assert 'the plan covers instances' not in err


def test_refuses_without_the_selection(world, capsys, monkeypatch):
    def missing():
        raise M.REG.MissingResult('missing training record for a requested seed: x')
    monkeypatch.setattr(M, 'select_policy_seed', missing)
    err = refused(world, capsys)
    assert 'training-run selection unavailable' in err


def test_refuses_a_cpsat_arm_that_is_not_the_pilots_choice(world, capsys):
    err = refused(world, capsys, '--cpsat-arm', 'compact')
    assert "differs from the pilot's choice ref_fullhint" in err


# ---------------------------------------------------------------------------
# partial mode

def test_partial_mode_option_restrictions(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(M.ROOT)
    for argv in (['--partial'], ['--out', str(tmp_path)], ['--rule-as-policy'],
                 ['--cpsat-folder', 'pilot'], ['--partial', '--out', str(tmp_path),
                                               '--record-policy-seed', '999'],
                 ['--partial', '--out', os.path.join(M.ROOT, 'results', 'x')],
                 ['--partial', '--out', os.path.join(M.ROOT, 'manuscript')],
                 ['--partial', '--out', os.path.join(M.ROOT, 'test_results', 'FAMILY', 'x')]):
        assert M.main(argv) == 2, argv
    assert 'REFUSING' in capsys.readouterr().err


def test_partial_mode_writes_marked_json_markdown_and_csv_only(world, capsys):
    camp = camp_of(world)
    os.remove(camp.record(cell(camp, 'cpsat_policy', 'LSW', 20)))
    assert M.main(args(world, partial=True)) == 0
    files = sorted(os.listdir(world / 'partial'))
    assert files == ['timed_curves.csv', 'timed_inference.json', 'timed_inference.md']
    assert not (world / 'out').exists()
    out = json.load(open(world / 'partial' / 'timed_inference.json'))
    assert out['partial'] is True and out['run'] == 'partial' and out['stand_in'] is False
    assert any('cpsat_policy@600/50x25n30/LSW/020: missing' in g for g in out['gaps'])
    pooled = out['results'][SIZE]['ii']['cpsat']['60']['Pool']
    assert pooled['n'] == 26 and pooled['c'] == 8          # instance 20 leaves the pooled row
    assert out['results'][SIZE]['ii']['cpsat']['60']['LSW']['n'] == 26
    assert open(world / 'partial' / 'timed_inference.md').readline().startswith(
        '# Timed campaign against the policy (partial)')
    rows = open(world / 'partial' / 'timed_curves.csv').read().splitlines()
    assert all(r.startswith('partial,') for r in rows[1:])


def test_stand_in_marks_every_output(world):
    assert M.main(args(world, '--rule-as-policy', '--record-policy-seed', str(SEL),
                       arms='memetic_rule@600 cpsat_rule@600', partial=True)) == 0
    out = json.load(open(world / 'partial' / 'timed_inference.json'))
    assert out['stand_in'] is True and out['run'] == 'partial, stand-in policy'
    assert out['policy'].startswith('stand-in')
    iv = out['results'][SIZE]['iv']['selected']['Pool']
    assert iv['mean'] == 0.0 and iv['equal'] == 100.0          # the rule against itself
    assert 'all_seeds' not in out['results'][SIZE]['i']['memetic']['1']
    assert 'stand-in' in open(world / 'partial' / 'timed_inference.md').readline()
    rows = open(world / 'partial' / 'timed_curves.csv').read().splitlines()
    assert all(r.startswith('"partial, stand-in policy"') for r in rows[1:])
    assert not (world / 'out').exists()


def test_macro_names_are_unique_over_every_size_arm_and_budget():
    match = [float(t) for t in range(601)]
    curve = M.log_grid(600, extra=M.BUDGETS)
    grid = sorted(set(match) | set(curve))
    n, C = 4, len(M.COMBOS)
    rng = np.random.default_rng(0)
    data = {}
    for size in M.SIZE_WORD:
        rule = 500 + rng.integers(0, 50, (n, C)).astype(float)
        p = rule - 10
        data[size] = dict(instances=list(range(n)), arms=list(M.P.ARMS), n_descriptors=n,
                          A={arm: np.broadcast_to(rule - 5, (len(grid), n, C)).copy()
                             for arm in M.P.ARMS},
                          rule=rule, pol_sel=p, all_pol=np.stack([p, p + 1, p - 1]),
                          seeds_with_arrays=list(SEEDS))
    R = M.analyse(data, list(M.COMBOS), list(M.BUDGETS), 30, 0.05, grid, match, curve, 'sel')
    info = dict(setting='exact', selection_text='s302 0.8', cpsat_arm='ref', boot=30,
                alpha=0.05, origin='start', campaigns=['all'], sel=SEL, k=3, horizon=600,
                primary=600)
    lines = M.macro_lines(R, info)
    names = M.macro_names(lines)
    M.check_macro_names(names, os.path.join(M.ROOT, 'manuscript', 'macros_timed.tex'))
    # per size, 11 entries (8 combinations, Pool and the two groups): (i) 2 x 5 x 11 x (7 + 4),
    # its verdicts at 600 s 2 x 2 x (8 + 3), (ii) and (iii) 2 x 5 x 11 x 4, (iv) 11 x 14, (v)
    # 2 x 11 x 3, (vi) the pooled rows only 2 x 5 x 2 x 4 x 3, 1
    assert len(names) == len(set(names)) == 3 * (1210 + 44 + 440 + 440 + 154 + 66 + 240 + 1) + 3
    # (vi) holds the numbers of the curve rows at the reported budgets, in their order
    # (arm minus policy), and is comparison (i) with the methods named the other way round
    for size, S in R.items():
        at = {(c['row'], c['method'], c['combination'], c['t_s']): c
              for c in S['curves'] + S['group_curves']}
        for m, arm in (('memetic', 'memetic_rule'), ('cpsat', 'cpsat_rule')):
            for b in M.BUDGETS:
                for key, rw in (('selected', 'Sel'), ('all_seeds', 'All')):
                    for r, x in S['vi'][m][str(b)][key].items():
                        c = at[(rw, arm, r, float(b))]
                        assert (x['mean'], x['lo'], x['hi']) == (c['mean_pct'], c['lo_pct'],
                                                                 c['hi_pct'])
                        assert x['mean'] * S['i'][m][str(b)][key][r]['mean'] <= 0
    assert {'TmdFiftyMemRuleStartVsPolSixHundredPoolSel', 'TmdEightyCpRuleStartVsPolOnePoolAllLo',
            'TmdTenCpRuleStartVsPolSixtyPoolAllDir'} <= set(names)
    verdicts = [x for x in names if 'Verdict' in x]
    assert len(verdicts) == 3 * 44 and all('SixHundred' in x for x in verdicts)
    assert {'TmdFiftyPolVsMemSixHundredLSWSelVerdict', 'TmdTenPolVsCpSixHundredAllVerdictBehindN',
            'TmdEightyPolVsMemSixHundredSelVerdictNotDistinguishableN'} <= set(verdicts)
    assert all(re.fullmatch(r'Tmd[A-Za-z]+', x) for x in names)
    assert {'TmdTenPolVsMemTenNSel', 'TmdEightyCpPolStartVsRuleStartThreeHundredLSW',
            'TmdFiftyRuleVsPolPoolAllLonger', 'TmdTenCpMatchNeverSW'} <= set(names)
    # the groups: NoSetup or Setup in the place of Pool, never a verdict
    assert {'TmdFiftyPolVsMemSixHundredNoSetupSel', 'TmdEightyCpRuleStartVsPolSixtySetupAllLo',
            'TmdTenRuleVsPolSetupSelLonger', 'TmdFiftyCpMatchMeanNoSetup',
            'TmdEightyMemPolStartVsRuleStartTenSetupDir'} <= set(names)
    assert not [x for x in verdicts if 'Setup' in x]
    head = [x for x in lines if x.startswith('%')]
    assert lines[:len(head)] == head and len(head) == 6      # header comments, then macros only
    assert 'NoSetup, the combinations without setups (N, L, W, LW)' in head[5]
    assert 'Setup, the combinations with setups (S, LS, SW, LSW)' in head[5]
    for line in lines[len(head):]:
        assert re.fullmatch(r'\\newcommand\{\\Tmd[A-Za-z]+\}\{[^{}]+\}  % .+', line), line
    with pytest.raises(M.Refusal):
        M.check_macro_names(names + [names[7]], 'none')
    with pytest.raises(M.Refusal):
        M.check_macro_names(['TmdFifty2'], 'none')


# ---------------------------------------------------------------------------
# verdicts at the stated budget

def verdict_case(k=5, n=40):
    """Differences d in % with a known verdict per combination: ahead on N, L, W, behind on
    LS, inside zero on LW, SW, LSW, and on S an interval just below zero whose one-sided p
    is below 0.05 before Holm and above it after."""
    e = np.random.default_rng(5).normal(0, 1, n // 2)
    e = np.concatenate([e, -e])
    e = (e - e.mean()) / e.std()                            # mean 0, sd 1 (standard error 1/sqrt(n))
    centre = dict(N=-3.0, L=-3.0, S=-2.1 / np.sqrt(n), W=-3.0, LS=3.0, LW=0.0, SW=0.2, LSW=-0.1)
    d = np.stack([centre[r] + e for r in M.COMBOS], axis=1)                         # [n, 8]
    run = np.array([-0.10, 0.0, 0.10, 0.05, -0.05])[:k, None, None]
    d_all = d[None] + run + 0.01 * np.random.default_rng(6).normal(0, 1, (k, n, 8))
    second = np.full((n, 8), 500.0)
    return second * (1 + d / 100), second * (1 + d_all / 100), second


def holm_is_monotone(rows, col):
    raw = [rows[r][col] for r in M.COMBOS]
    adj = [rows[r][col + '_holm'] for r in M.COMBOS]
    assert all(a >= p for a, p in zip(adj, raw))                   # never below the raw p
    assert all(adj[i] <= adj[j] for i in range(8) for j in range(8) if raw[i] <= raw[j])
    assert all(a <= 1.0 for a in adj)


def test_verdicts_ahead_behind_and_not_distinguishable():
    first, first_all, second = verdict_case()
    B = 8000
    sel = M.verdict_family(first, second, M.COMBOS, B, 0.05)
    want = dict(N='ahead', L='ahead', S='not distinguishable', W='ahead', LS='behind',
                LW='not distinguishable', SW='not distinguishable', LSW='not distinguishable')
    assert {r: x['verdict'] for r, x in sel['rows'].items()} == want
    assert sel['counts'] == {'ahead': 3, 'behind': 1, 'not distinguishable': 4}
    s = sel['rows']['S']                    # the interval lies below zero; Holm decides
    assert s['hi'] < 0 and s['p_boot_less'] < 0.05 <= s['p_boot_less_holm']
    for col in ('p_boot_less', 'p_boot_greater'):
        holm_is_monotone(sel['rows'], col)
    # the Sel row has no seed-level p: one run has no seed-level test
    assert all(not key.startswith('p_t') for x in sel['rows'].values() for key in x)
    assert 'seed-level' not in sel['scheme'] and 'No seed-level test' in sel['rule']
    # its interval is the one compare reports for the same row (same seed, same draws)
    rows = M.compare(first, second, M.COMBOS, B, 0.05)
    assert all((rows[r]['lo'], rows[r]['hi']) == (sel['rows'][r]['lo'], sel['rows'][r]['hi'])
               for r in M.COMBOS)
    allr = M.verdict_family(first_all, second, M.COMBOS, 2000, 0.05, seeds=True)
    assert {r: x['verdict'] for r, x in allr['rows'].items() if r != 'S'} == \
        {r: w for r, w in want.items() if r != 'S'}
    for col in ('p_t_less', 'p_t_greater', 'p_boot_less', 'p_boot_greater'):
        holm_is_monotone(allr['rows'], col)
    rows = M.compare(first_all, second, M.COMBOS, 2000, 0.05, seeds=True)
    assert all((rows[r]['lo'], rows[r]['hi']) == (allr['rows'][r]['lo'], allr['rows'][r]['hi'])
               for r in M.COMBOS)
    # the All row is p89's combined rule, applied to these numbers
    for x in allr['rows'].values():
        assert x['verdict'] == M.superiority_verdict(
            x['lo'], x['hi'], x['p_t_less_holm'], x['p_boot_less_holm'],
            x['p_t_greater_holm'], x['p_boot_greater_holm'])
    # a family with a combination without instances has no verdict
    gapped = first.copy()
    gapped[:, M.COMBOS.index('SW')] = np.nan
    assert M.verdict_family(gapped, second, M.COMBOS, 200, 0.05) is None


def test_verdict_macros_only_at_the_stated_budget(world, capsys):
    assert M.main(args(world)) == 0                          # budgets 1, 10, 60; stated 60
    mac, txt = macros(world)
    verdicts = {n: v for n, v in mac.items() if 'Verdict' in n}
    assert verdicts and all(re.match(r'TmdFiftyPolVs(Mem|Cp)Sixty', n) for n in verdicts)
    for m in ('Mem', 'Cp'):
        for rw in ('Sel', 'All'):
            counts = [int(mac[f'TmdFiftyPolVs{m}Sixty{rw}Verdict{w}'])
                      for w in ('AheadN', 'BehindN', 'NotDistinguishableN')]
            assert sum(counts) == 8
            assert all(mac[f'TmdFiftyPolVs{m}Sixty{r}{rw}Verdict'] in M.VERDICT_WORDS
                       for r in M.COMBOS)
            assert f'TmdFiftyPolVs{m}SixtyPool{rw}Verdict' not in mac        # no pooled verdict
    # at 60 s the memetic arm (R0 - 40) is shorter than the policy (R0 - 20) on every cell
    assert mac['TmdFiftyPolVsMemSixtySelVerdictBehindN'] == '8'
    # no verdict for (ii) to (iv)
    assert not any(n.startswith(('TmdFiftyMemPolStart', 'TmdFiftyCpPolStart', 'TmdFiftyRuleVsPol'))
                   for n in verdicts)
    out = json.load(open(world / 'out' / 'results' / 'timed_inference.json'))
    v = out['results'][SIZE]['verdict']
    assert sorted(v) == ['cpsat', 'memetic'] and v['memetic']['budget_s'] == 60
    assert not any(k.startswith('p_t') for x in v['memetic']['selected']['rows'].values() for k in x)
    assert 'p_t_less_holm' in v['memetic']['all_seeds']['rows']['N']
    assert out['primary_budget_s'] == 60 and out['verdicts']['family'] == list(M.COMBOS)
    md = open(world / 'out' / 'results' / 'timed_inference.md').read()
    assert '### (i) verdict at the stated budget of 60 s' in md
    # the stated budget must be one of the budgets; full mode keeps the eight combinations
    assert M.main(args(world, '--primary', '600')) == 2
    assert M.main(args(world, '--combos', 'N', 'L')) == 2
    assert 'is not one of --budgets' in capsys.readouterr().err


def test_curves_hold_the_selected_and_every_run(world):
    assert M.main(args(world)) == 0
    rows = list(csv.DictReader(open(world / 'out' / 'results' / 'timed_curves.csv')))
    assert {x['row'] for x in rows} == {'Sel', 'All'}
    assert {x['method'] for x in rows if x['row'] == 'All'} == {'memetic_rule', 'cpsat_rule'}
    assert {x['method'] for x in rows if x['row'] == 'Sel'} == {t.split('@')[0] for t in ARMS4.split()}
    assert all(x['n_runs'] == ('3' if x['row'] == 'All' else '1') for x in rows)
    assert all(x['policy'].startswith('every training run') for x in rows if x['row'] == 'All')
    # time zero, N, memetic_rule: the rule schedule against each run, averaged over runs and
    # instances; the interval is the hierarchical one of compare
    x = next(x for x in rows if x['row'] == 'All' and x['method'] == 'memetic_rule'
             and x['combination'] == 'N' and x['t_s'] == '0')
    d = np.array([[100 * (r0(i, 0) - pol(s, i, 0)) / pol(s, i, 0) for i in EVID] for s in SEEDS])
    assert float(x['mean_pct']) == pytest.approx(d.mean(), abs=1e-6)
    hb = SF.hier_boot(d, B=200, seed=M.HIER_SEED)
    assert (float(x['lo_pct']), float(x['hi_pct'])) == pytest.approx((hb['lo'], hb['hi']), abs=1e-6)
    sel = next(y for y in rows if y['row'] == 'Sel' and y['method'] == 'memetic_rule'
               and y['combination'] == 'N' and y['t_s'] == '0')
    assert float(sel['mean_pct']) == pytest.approx(d[SEEDS.index(SEL)].mean(), abs=1e-6)


def test_number_words_and_grid():
    assert [M.number_word(b) for b in M.BUDGETS] == ['One', 'Ten', 'Sixty', 'ThreeHundred',
                                                    'SixHundred']
    assert M.number_word(1800) == 'OneThousandEightHundred'
    g = M.log_grid(600, extra=M.BUDGETS)
    assert g[0] == 0.0 and g[1] == 0.1 and g[-1] == 600.0
    assert all(float(b) in g for b in M.BUDGETS) and len(g) > 40


# ---------------------------------------------------------------------------
# the request clock: offsets of the rule and policy records of p97

def rrefused(root, capsys, *extra, partial=False, **kw):
    mac = root / 'out' / 'manuscript' / 'macros_timed.tex'
    mac.parent.mkdir(parents=True, exist_ok=True)
    mac.write_text('% stale\n')
    rc = M.main(rargs(root, *extra, partial=partial, **kw))
    err = capsys.readouterr().err
    assert rc == 1 and 'REFUSING' in err
    if not partial:
        assert not mac.exists()
    return err


def p97_records(root, i, j):
    """(rule record, policy record) of one cell, as written."""
    r = M.COMBOS[j]
    rule = json.load(open(P97.rule_record_path(str(root / 'rr'), SIZE, r, i)))
    polr = json.load(open(P97.record_path(str(root / 'rr'), SIZE, POL_BUDGET, POL_BATCH,
                                          f's{SEL}', r, i)))
    return rule, polr


def run_value(rec, runner, t):
    """The start-clock value of a search record at run time t (the planner's reader)."""
    rows = M.P.record_rows(rec, runner, times=(t,), origin='start')
    return rows[0]['makespan'] if rows else 'beyond the budget'


def outputs(root):
    J = json.load(open(root / 'out' / 'results' / 'timed_inference.json'))
    mac, txt = macros(root)
    rows = list(csv.DictReader(open(root / 'out' / 'results' / 'timed_curves.csv')))
    md = open(root / 'out' / 'results' / 'timed_inference.md').read()
    return J, mac, txt, rows, md


def test_records_of_p97_carry_the_offsets(base_request_world):
    rule, polr = p97_records(base_request_world, 3, 0)
    assert rule['arm'] == M.RULE_ARM and rule['t_all'] == o_rule(3, 0) == 1.5
    assert rule['best_rule'] == 'SPT' and rule['best_makespan'] == r0(3, 0)
    assert polr['arm'] == M.POLICY_ARM and polr['greedy_t'] == o_pol(3, 0)
    assert polr['greedy_makespan'] == pol(SEL, 3, 0) and polr['greedy_t_resident'] == 4.0
    for t in (0.0, 6.0, 7.9, 8.0, 19.0, 20.0, 25.0, 30.0, 59.0):
        assert incumbent_at(polr, t, 'wall', 'process') == sampled_at(3, 0, t)


def test_request_clock_shifts_each_arm_by_its_offset(rworld):
    assert M.main(rargs(rworld)) == 0
    J, mac, txt, rows, md = outputs(rworld)
    assert J['origin'] == 'request' and 'Clock origin request' in txt
    assert J['budgets_per_size'] == {SIZE: [1, 10, 60]}
    camp = camp_of(rworld)
    runner = {seg.arm: seg.runner for seg in camp.segments}
    seen = 0
    for x in J['records']:
        i, j = x['instance'], M.COMBOS.index(x['combination'])
        off = o_rule(i, j) if x['arm'].endswith('_rule') else o_pol(i, j)
        assert x['offset_s'] == pytest.approx(off, abs=1e-12)
        rec = json.load(open(camp.record(cell(camp, x['arm'], x['combination'], i))))
        for b in (1, 10, 60):
            want = None if b < off else run_value(rec, runner[x['arm']], b - off)
            assert x['at'][str(b)] == want, (x['arm'], i, x['combination'], b)
            seen += 1
    assert seen == 4 * 27 * 8 * 3
    # by hand: memetic from the rules at 10 s on instance 3, N: run time 8.5 s, R0 - 10
    x = next(x for x in J['records'] if x['arm'] == 'memetic_rule' and x['instance'] == 3
             and x['combination'] == 'N')
    assert x['at'] == {'1': None, '10': r0(3, 0) - 10, '60': r0(3, 0) - 40}
    # the reader itself, on a fine grid of one record
    rec = json.load(open(camp.record(cell(camp, 'cpsat_rule', 'W', 5))))
    grid = [round(0.25 * k, 2) for k in range(0, 241)]
    got = M.read_record(rec, 'cpsat', grid, 'request', offset=o_rule(5, 3))
    for t in grid:
        if t < o_rule(5, 3):
            assert t not in got
        else:
            assert got[t] == run_value(rec, 'cpsat', t - o_rule(5, 3))
    # (v) on the request clock: the start-clock times (30 s, 55 s) plus o_rule, on the grid
    assert mac['TmdFiftyMemMatchMeanN'] == '32' and mac['TmdFiftyMemMatchMedianPool'] == '32'
    assert mac['TmdFiftyCpMatchMeanLSW'] == '57'


def test_no_number_before_an_offset(rworld):
    assert M.main(rargs(rworld)) == 0
    J, mac, txt, rows, md = outputs(rworld)
    S = J['results'][SIZE]
    # 1 s: neither the rules (from 1.5 s) nor the policy (from 6 s) has a schedule yet
    for blk in (S['i']['memetic']['1']['selected'], S['i']['cpsat']['1']['all_seeds'],
                S['ii']['memetic']['1'], S['iii']['cpsat']['1'], S['vii']['1'],
                S['viii']['memetic']['1']):
        assert set(blk) == set(M.COMBOS) | {'Pool', 'NoSetup', 'Setup'}
        assert all(x['no_schedule_yet'] == x['n_cells'] for x in blk.values())
    # no number at 1 s; the bold markers of the decision table exist for every row and are
    # empty in a row without a number
    bold = {n: v for n, v in mac.items() if n.endswith('Bold')}
    assert not [n for n in mac if re.search(r'One(?=[A-Z])', n) and n not in bold]
    assert [n for n in bold if 'One' in n] and all(bold[n] == '' for n in bold if 'One' in n)
    # 10 s: the greedy schedule of instance 7 in LS comes at 12 s, so LS, the pooled row and
    # the group with setups have no number for every comparison with the policy; the other
    # combinations and the group without setups do
    for blk in (S['i']['memetic']['10']['selected'], S['vi']['cpsat']['10']['all_seeds'],
                S['ii']['memetic']['10'], S['vii']['10'], S['viii']['cpsat']['10']):
        assert blk['LS']['no_schedule_yet'] == 1 and blk['LS']['n_cells'] == 27
        assert 'mean' not in blk['LS'] and 'mean' not in blk['Pool']
        assert blk['Pool']['no_schedule_yet'] == 1 and blk['Pool']['n_cells'] == 216
        assert all('mean' in blk[r] and blk[r]['n'] == 27 for r in M.COMBOS if r != 'LS')
        assert {k: blk['Setup'][k] for k in ('no_schedule_yet', 'n_cells', 'combinations')} \
            == dict(no_schedule_yet=1, n_cells=108, combinations=['S', 'LS', 'SW', 'LSW'])
        assert 'mean' not in blk['Setup']
        assert blk['NoSetup']['n'] == 27 and blk['NoSetup']['c'] == 4
        assert blk['NoSetup']['combinations'] == ['N', 'L', 'W', 'LW']
    assert 'TmdFiftyPolVsMemTenNSel' in mac and 'TmdFiftyPolVsMemTenLSWSel' in mac
    assert 'TmdFiftyPolVsMemTenNoSetupSel' in mac and 'TmdFiftyPolSmpVsPolTenNoSetup' in mac
    for name in ('TmdFiftyPolVsMemTenLSSel', 'TmdFiftyPolVsMemTenPoolSel',
                 'TmdFiftyMemPolStartVsPolTenLS', 'TmdFiftyPolSmpVsPolTenPool',
                 'TmdFiftyPolSmpVsCpTenLSSel', 'TmdFiftyCpRuleStartVsPolTenPoolAll',
                 'TmdFiftyPolVsMemTenSetupSel', 'TmdFiftyCpRuleStartVsPolTenSetupAll'):
        assert name not in mac
    # the policy-started arm starts at o_pol too, so (iii) has no LS number at 10 s
    assert S['iii']['memetic']['10']['LS']['no_schedule_yet'] == 1
    assert 'no schedule yet (1 of 216 cells)' in md and 'no schedule yet (27 of 27 cells)' in md
    # every macro holds a number or a word, never a placeholder (a bold marker is \bfseries
    # or empty)
    assert all(v not in ('', 'nan', '-', 'None') for n, v in mac.items() if n not in bold)
    assert {bold[n] for n in bold} == {'', M.BOLD}
    # the verdicts at the stated budget exist: every cell has both schedules at 60 s
    assert mac['TmdFiftyPolVsMemSixtySelVerdictBehindN'] == '8'


def test_held_values_are_flagged(rworld):
    assert M.main(rargs(rworld)) == 0
    J, mac, txt, rows, md = outputs(rworld)
    S = J['results'][SIZE]
    assert S['policy_budget_s'] == POL_BUDGET and mac['TmdFiftyPolSmpBudget'] == '30'
    assert all(x['held'] is True and x['held_after_s'] == POL_BUDGET
               for x in S['vii']['60'].values())
    assert all(x['held'] is False for x in S['vii']['10'].values())
    assert all(x['held'] is True for blk in S['viii'].values() for x in blk['60'].values())
    assert S['verdict_viii']['cpsat']['held'] is True
    lines = {n: line for line in txt.splitlines()
             for n in re.findall(r'\\newcommand\{\\([A-Za-z]+)\}', line)}
    note = 'held after its budget of 30 s'
    assert note in lines['TmdFiftyPolSmpVsPolSixtyPool']
    assert note in lines['TmdFiftyPolSmpVsMemSixtyNSel'] and note in lines['TmdFiftyPolSmpVsCpSixtyNSelVerdict']
    assert note not in lines['TmdFiftyPolSmpVsPolTenN']
    # (v) against the sampled curve: memetic reaches it at 32 s, after the record's budget
    assert mac['TmdFiftyMemMatchSmpMeanN'] == '32' and note in lines['TmdFiftyMemMatchSmpMeanN']
    assert mac['TmdFiftyCpMatchSmpMeanN'] == M.NOT_REACHED and mac['TmdFiftyCpMatchSmpNeverPool'] == '100.0'
    assert note in lines['TmdFiftyCpMatchSmpNeverPool']
    smp = [x for x in rows if x['method'] == M.POLICY_ARM]
    assert smp and all(x['held'] == ('yes' if float(x['t_s']) > POL_BUDGET else 'no') for x in smp)
    assert {x['held'] for x in rows if x['method'] != M.POLICY_ARM} == {'no'}
    assert all(x['policy_budget_s'] == '30' for x in rows)
    assert '(held)' in md and 'budget 30 s (values after it are held)' in md


def test_sampling_never_lengthens_the_greedy_schedule(rworld):
    assert M.main(rargs(rworld)) == 0
    J, mac, txt, rows, md = outputs(rworld)
    vii = J['results'][SIZE]['vii']
    written = [x for blk in vii.values() for x in blk.values() if 'mean' in x]
    # 60 s: the eight, Pool and both groups; 10 s: all but LS, Pool and the group with setups
    assert len(written) == 11 + 8
    assert all(x['mean'] <= 0 and x['hi'] <= 0 for x in written)
    assert all(x['equal'] == 0.0 and x['shorter'] == 100.0 for x in written)
    want = np.mean([100 * (-9) / pol(SEL, i, j) for i in EVID for j in range(8)])
    assert vii['60']['Pool']['mean'] == pytest.approx(want, abs=1e-9)
    assert float(mac['TmdFiftyPolSmpVsPolSixtyPool']) == pytest.approx(want, abs=0.006)
    smp = [x for x in rows if x['method'] == M.POLICY_ARM]
    assert smp and all(float(x['mean_pct']) <= 0 and float(x['hi_pct']) <= 0 for x in smp)
    assert {x['row'] for x in smp} == {'Sel'}


def test_equal_effort_comparison_has_the_sign_opposite_to_arm_minus_sampled(rworld):
    assert M.main(rargs(rworld)) == 0
    J, mac, txt, rows, md = outputs(rworld)
    viii = J['results'][SIZE]['viii']
    camp = camp_of(rworld)
    members = dict({g[0]: g[2] for g in M.GROUPS}, Pool=M.COMBOS)
    checked = 0
    for m, arm, runner in (('memetic', 'memetic_rule', 'memetic'), ('cpsat', 'cpsat_rule', 'cpsat')):
        for b in (10, 60):
            for r, x in viii[m][str(b)].items():
                if 'mean' not in x:
                    continue
                d, rev = [], []
                for j in range(8):
                    if M.COMBOS[j] not in members.get(r, (r,)):
                        continue
                    for i in EVID:
                        rule, polr = p97_records(rworld, i, j)
                        rec = json.load(open(camp.record(cell(camp, arm, M.COMBOS[j], i))))
                        a = run_value(rec, runner, b - rule['t_all'])
                        s = incumbent_at(polr, b, 'wall', 'process')
                        d.append(100 * (s - a) / a)
                        rev.append(100 * (a - s) / s)
                assert x['mean'] == pytest.approx(np.mean(d), abs=1e-9)
                assert x['mean'] * np.mean(rev) < 0
                checked += 1
    assert checked == 2 * (11 + 8)
    # 60 s: memetic (R0 - 40) shorter than the held R0 - 29, CP-SAT (R0 - 25) longer
    assert mac['TmdFiftyPolSmpVsMemSixtySelVerdictBehindN'] == '8'
    assert mac['TmdFiftyPolSmpVsCpSixtySelVerdictAheadN'] == '8'
    assert all(mac[f'TmdFiftyPolSmpVsCpSixty{r}SelVerdict'] == 'ahead' for r in M.COMBOS)
    v = J['results'][SIZE]['verdict_viii']['memetic']['selected']
    assert not any(k.startswith('p_t') for x in v['rows'].values() for k in x)
    assert '(viii) verdict at the stated budget of 60 s' in md


def test_first_schedule_times_and_the_start_of_each_curve(rworld):
    assert M.main(rargs(rworld)) == 0
    J, mac, txt, rows, md = outputs(rworld)
    S = J['results'][SIZE]
    t_all = [o_rule(i, j) for i in EVID for j in range(8)]
    t_pol = [o_pol(i, j) for i in EVID for j in range(8)]
    for word, vals in (('FirstRule', t_all), ('FirstPol', t_pol), ('FirstPolResident', [4.0])):
        assert mac[f'TmdFifty{word}Median'] == f'{np.median(vals):.2f}'
        assert mac[f'TmdFifty{word}Min'] == f'{min(vals):.2f}'
        assert mac[f'TmdFifty{word}Max'] == f'{max(vals):.2f}'
    assert S['ix']['policy']['t']['n_cells'] == 216 and S['ix']['policy']['t']['max_s'] == 12.0
    every = [x for x in rows if x['group'] == 'all']
    assert {x['policy_first_s'] for x in every} == {f'{np.median(t_pol):.6f}'}
    assert {x['rule_first_s'] for x in every} == {f'{np.median(t_all):.6f}'}
    head = open(rworld / 'out' / 'results' / 'timed_curves.csv').readline().strip().split(',')
    assert tuple(head) == M.CSV_COLUMNS_REQUEST and {x['origin'] for x in rows} == {'request'}
    # each arm's rows start at the first grid time at which every cell has a schedule
    curve = J['grid']['curve']
    first = {arm: min(float(x['t_s']) for x in every if x['method'] == arm)
             for arm in {x['method'] for x in every}}
    assert first['memetic_rule'] == first['cpsat_rule'] == min(t for t in curve if t >= max(t_all))
    assert first['memetic_policy'] == first[M.POLICY_ARM] == min(t for t in curve if t >= 12.0)
    assert S['curve_start_s'] == first
    # per group: the times over the group's cells, and its rows from the first grid time at
    # which every cell of its own combinations has a schedule (the late cell is LS, a
    # combination with setups, so the policy's rows without setups start earlier)
    for key, _, combos, grp in M.GROUPS:
        cols = [M.COMBOS.index(r) for r in combos]
        g_all = [o_rule(i, j) for i in EVID for j in cols]
        g_pol = [o_pol(i, j) for i in EVID for j in cols]
        mine = [x for x in rows if x['group'] == grp]
        assert mine and {x['combination'] for x in mine} == {key}
        assert {x['n_combinations'] for x in mine} == {'4'}
        assert {x['policy_first_s'] for x in mine} == {f'{np.median(g_pol):.6f}'}
        assert {x['rule_first_s'] for x in mine} == {f'{np.median(g_all):.6f}'}
        assert mac[f'TmdFiftyFirstPolMax{key}'] == f'{max(g_pol):.2f}'
        assert mac[f'TmdFiftyFirstRuleMedian{key}'] == f'{np.median(g_all):.2f}'
        assert S['ix'][key]['policy']['t']['n_cells'] == 108
        starts = {arm: min(float(x['t_s']) for x in mine if x['method'] == arm)
                  for arm in {x['method'] for x in mine}}
        assert S['group_curve_start_s'][key] == starts
        assert starts['memetic_rule'] == min(t for t in curve if t >= max(g_all))
        assert starts[M.POLICY_ARM] == min(t for t in curve if t >= max(g_pol))
    assert S['group_curve_start_s']['NoSetup'][M.POLICY_ARM] < first[M.POLICY_ARM]
    assert S['group_curve_start_s']['Setup'] == first
    # the rows of the groups come after every row of group all
    groups = [x['group'] for x in rows]
    assert groups == sorted(groups, key=lambda g: g != 'all')
    # at a reported budget the curve and comparison (vi) hold the same numbers
    for key, rw in (('selected', 'Sel'), ('all_seeds', 'All')):
        x = S['vi']['memetic']['60'][key]['Pool']
        c = next(c for c in rows if c['row'] == rw and c['method'] == 'memetic_rule'
                 and c['combination'] == 'Pool' and c['t_s'] == '60')
        assert (float(c['mean_pct']), float(c['lo_pct'])) == pytest.approx((x['mean'], x['lo']),
                                                                           abs=1e-6)


def test_request_clock_macro_names_are_unique(rworld):
    assert M.main(rargs(rworld)) == 0
    J, mac, txt, rows, md = outputs(rworld)
    names = re.findall(r'\\newcommand\{\\([^}]*)\}', txt)
    assert len(names) == len(set(names)) and all(re.fullmatch(r'Tmd[A-Za-z]+', n) for n in names)
    assert {'TmdFiftyPolSmpBudget', 'TmdFiftyFirstRuleMedian', 'TmdFiftyFirstPolResidentMax',
            'TmdFiftyPolSmpVsPolSixtyPool', 'TmdFiftyPolSmpVsMemSixtyNSelLo',
            'TmdFiftyPolSmpVsCpSixtySelVerdictAheadN', 'TmdFiftyMemMatchSmpMedianPool',
            'TmdFiftyCpPolStartVsRuleStartSixtyPool'} <= set(names)
    head = [x for x in txt.splitlines() if x.startswith('%')]
    assert len(head) == 7 and 'Request clock' in head[5] and 'Pooled rows per group' in head[6]


def test_request_macro_names_are_unique_over_every_size():
    match = [float(t) for t in range(601)]
    budgets_of = {s: list(M.TRAINING_BUDGETS if s == M.P.TRAINING_SIZE else M.REQUEST_BUDGETS)
                  for s in M.SIZE_WORD}
    curve = M.log_grid(600, extra=M.TRAINING_BUDGETS)
    grid = sorted(set(match) | set(curve))
    n, C = 4, len(M.COMBOS)
    rng = np.random.default_rng(0)
    g = np.array(grid)[:, None, None]
    data = {}
    for size in M.SIZE_WORD:
        rule = 500 + rng.integers(0, 50, (n, C)).astype(float)
        p = rule - 10
        o_r, o_p = np.full((n, C), 1.0), np.full((n, C), 2.0)
        later = np.where(g >= 1.0, rule - 5 - (g > 100), np.nan)
        data[size] = dict(instances=list(range(n)), arms=list(M.P.ARMS), n_descriptors=n,
                          A={arm: later.copy() for arm in M.P.ARMS},
                          A_present={arm: np.ones((n, C), dtype=bool) for arm in M.P.ARMS},
                          rule=rule, pol_sel=p, all_pol=np.stack([p, p + 1, p - 1]),
                          seeds_with_arrays=list(SEEDS), o_rule=o_r, o_pol=o_p,
                          o_res=o_p - 1, S=np.where(g >= 2.0, p - 3 + 0 * g, np.nan),
                          S_present=np.ones((n, C), dtype=bool), policy_budget_s=120.0,
                          budgets=budgets_of[size])
    R = M.analyse_request(data, list(M.COMBOS), 30, 0.05, grid, match, curve, 'sel')
    info = dict(setting='exact', selection_text='s302 0.8', cpsat_arm='ref', boot=30,
                alpha=0.05, origin='request', campaigns=['all'], sel=SEL, k=3, horizon=600,
                primary=600)
    names = M.macro_names(M.macro_lines(R, info))
    M.check_macro_names(names, os.path.join(M.ROOT, 'manuscript', 'macros_timed.tex'))
    assert len(names) == len(set(names)) > 3 * 1500
    assert {'TmdTenPolSmpVsCpTenNSel', 'TmdEightyPolSmpVsMemSixHundredSelVerdictBehindN',
            'TmdFiftyFirstPolMedian', 'TmdTenPolSmpBudget', 'TmdEightyCpMatchSmpNeverLSW',
            'TmdTenPolVsMemTenPoolAll'} <= set(names)
    # the groups beside Pool, and (ix) per group with the group at the end of the name
    assert {'TmdFiftyFirstPolMedianSetup', 'TmdTenFirstPolResidentMaxNoSetup',
            'TmdTenPolSmpVsPolTenNoSetup', 'TmdEightyMemMatchSmpNeverSetup',
            'TmdEightyPolSmpVsCpSixHundredNoSetupSel', 'TmdFiftyMemRuleStartVsPolSixtySetupAll'
            } <= set(names)
    assert not [x for x in names if 'Verdict' in x and 'Setup' in x]
    assert not any(x.startswith('TmdFifty') and 'TenN' in x and 'PolSmpVs' in x for x in names)
    # every value held after 120 s says so; the stated budget of 600 s is one of them
    assert R['10x25']['verdict_viii']['memetic']['held'] is True
    assert R['10x25']['vii']['60']['Pool']['held'] is False


def test_request_is_the_default_clock_and_needs_the_policy_token(rworld, capsys):
    assert M.main(rargs(rworld, origin=None)) == 0
    J, mac, txt, rows, md = outputs(rworld)
    assert J['origin'] == 'request' and J['policy_sample']['batch'] == POL_BATCH
    assert J['new_records_valid'] == {M.RULE_ARM: 216, M.POLICY_ARM: 216}
    assert len(J['policy_records']) == len(J['rule_records']) == 216
    err = rrefused(rworld, capsys, arms=ARMS4)
    assert f'no {M.POLICY_ARM} token in the campaign' in err


def test_start_clock_refuses_the_request_options(world, capsys):
    err = refused(world, capsys, arms=f'memetic_rule@600 {POL_TOKEN}')
    assert 'the sampled schedules are read on the request clock only' in err
    assert M.main(args(world, '--policy-batch', '2')) == 2
    assert M.main(args(world, '--training-budgets', '10', '60')) == 2
    assert M.main(rargs(world, '--rule-as-policy', partial=True)) == 2
    assert '--rule-as-policy reads the start and search clocks only' in capsys.readouterr().err


def test_start_clock_ignores_the_new_records(rworld):
    """--origin start writes the same bytes whether or not the rule and policy records
    exist (the proof that the request clock leaves the start clock unchanged)."""
    out = {}
    for label in ('with', 'without'):
        if label == 'without':
            shutil.rmtree(P97.folder(str(rworld / 'rr'), SIZE, POL_BUDGET, POL_BATCH, f's{SEL}'))
            shutil.rmtree(P97.rule_folder(str(rworld / 'rr'), SIZE))
        assert M.main(args(rworld)) == 0
        out[label] = {p: open(rworld / 'out' / p, 'rb').read() for p in
                      ('manuscript/macros_timed.tex', 'results/timed_inference.json',
                       'results/timed_inference.md', 'results/timed_curves.csv')}
    assert out['with'] == out['without']
    assert b'"origin": "start"' in out['with']['results/timed_inference.json']


def test_request_refuses_missing_invalid_and_mismatched_records(rworld, capsys, monkeypatch):
    os.remove(P97.rule_record_path(str(rworld / 'rr'), SIZE, 'LS', 7))
    err = rrefused(rworld, capsys)
    assert f'{M.RULE_ARM}/{SIZE}/LS/007: missing' in err
    write_rule_record(rworld, 7, M.COMBOS.index('LS'))
    # a worker below the share bar: p97.classify calls the record invalid
    write_policy_record(rworld, 9, M.COMBOS.index('W'), share=0.5, overwrite=True)
    err = rrefused(rworld, capsys)
    assert f'{M.POLICY_ARM}@30/{SIZE}/W/009: invalid (cpu_share 0.5' in err
    write_policy_record(rworld, 9, M.COMBOS.index('W'), overwrite=True)
    # a greedy schedule that is not the selected run's array (no stored value to compare,
    # so p97 calls it valid; p94's own check refuses)
    monkeypatch.setattr(P97, 'stored_greedy', lambda *a: ({}, []))
    write_policy_record(rworld, 4, 0, greedy=pol(303, 4, 0), overwrite=True)
    err = rrefused(rworld, capsys)
    assert 'greedy makespan' in err and 'policy array of the selected run' in err
    assert 'instance 4, N' in err
    # a record of another plan (another number of workers) is foreign; another batch size
    # names another folder
    write_policy_record(rworld, 4, 0, overwrite=True)
    err = rrefused(rworld, capsys, '--policy-workers', '3')
    assert f'{M.POLICY_ARM}@30/{SIZE}/N/004: foreign (workers=2, plan 3)' in err
    err = rrefused(rworld, capsys, '--policy-batch', '4')
    assert ': missing' in err and '_polsample30s_b4_s302' in err


def test_request_partial_mode_leaves_out_a_cell_without_its_policy_record(rworld):
    os.remove(P97.record_path(str(rworld / 'rr'), SIZE, POL_BUDGET, POL_BATCH, f's{SEL}', 'N', 3))
    assert M.main(rargs(rworld, partial=True)) == 0
    out = json.load(open(rworld / 'partial' / 'timed_inference.json'))
    assert any(f'{M.POLICY_ARM}@30/{SIZE}/N/003: missing' in g for g in out['gaps'])
    S = out['results'][SIZE]
    assert S['i']['memetic']['60']['selected']['N']['n'] == 26
    assert S['i']['memetic']['60']['selected']['Pool']['n'] == 26
    assert S['vii']['60']['N']['n'] == 26 and S['viii']['cpsat']['60']['L']['n'] == 27
    # the rule-started arms against each other keep the cell
    assert S['iii']['memetic']['60']['N']['n'] == 26        # its policy-started arm has no offset
    assert S['iv']['selected']['N']['n'] == 27


# ---------------------------------------------------------------------------
# the groups: combinations without setups and combinations with setups

GROUP_OF = {g[0]: g[2] for g in M.GROUPS}
NEW_KEYS = {'NoSetup', 'Setup', 'groups', 'group_curve_start_s'}   # what the groups add


def without_groups(x):
    """The JSON with every entry the groups add removed."""
    if isinstance(x, dict):
        return {k: without_groups(v) for k, v in x.items() if k not in NEW_KEYS}
    if isinstance(x, list):
        return [without_groups(v) for v in x]
    return x


def test_group_entries_equal_a_recomputation_from_their_cells(rworld):
    """(i) and (vii) of each group: the mean of the differences on the cells of the group's
    combinations, and the interval of the same statistic on those cells with the script's
    generator rule (a fresh default_rng(7) for one run, default_rng(11) over every run)."""
    assert M.main(rargs(rworld)) == 0
    J, mac, txt, rows, md = outputs(rworld)
    S = J['results'][SIZE]
    camp = camp_of(rworld)
    C8 = range(len(M.COMBOS))
    # every cell from the records and the arrays, [instance, combination] (and [run, ...])
    sel = np.array([[pol(SEL, i, j) for j in C8] for i in EVID], dtype=float)
    every = np.array([[[pol(s, i, j) for j in C8] for i in EVID] for s in SEEDS], dtype=float)
    arm_at = {arm: np.array([[run_value(json.load(open(camp.record(cell(camp, arm, M.COMBOS[j], i)))),
                                        runner, 60 - o_rule(i, j)) for j in C8] for i in EVID],
                            dtype=float)
              for arm, runner in (('memetic_rule', 'memetic'), ('cpsat_rule', 'cpsat'))}
    smp_at = {b: np.array([[incumbent_at(p97_records(rworld, i, j)[1], float(b), 'wall',
                                         'process') or np.nan for j in C8] for i in EVID],
                          dtype=float) for b in (10, 60)}

    def same(x, d, stat):
        """The entry against the group's cells d: the mean equal to the correctly rounded
        mean of the cells up to the order of summation (numpy sums in memory order, so two
        layouts of the same cells can differ in the last bit: at most 4 units in the last
        place), the interval bit for bit that of the same statistic on d with the script's
        seed."""
        want = stat(d)
        exact = math.fsum(d.ravel()) / d.size
        assert abs(x['mean'] - exact) <= 4 * np.spacing(abs(exact))
        assert abs(x['mean'] - want['mean']) <= 4 * np.spacing(abs(exact))
        assert (x['lo'], x['hi']) == (want['lo'], want['hi'])

    one = lambda d: SF.instance_boot(d, B=200, seed=M.INST_SEED)   # noqa: E731
    runs = lambda d: SF.hier_boot(d, B=200, seed=M.HIER_SEED)      # noqa: E731
    checked = 0
    for key, combos in GROUP_OF.items():
        cols = [M.COMBOS.index(r) for r in combos]
        # (i) at 60 s since the request: the policy's greedy schedule minus each rule-started arm
        for m, arm in (('memetic', 'memetic_rule'), ('cpsat', 'cpsat_rule')):
            a = arm_at[arm]
            for row, d, stat in (('selected', (100.0 * (sel - a) / a)[:, cols], one),
                                 ('all_seeds', (100.0 * (every - a) / a)[:, :, cols], runs)):
                x = S['i'][m]['60'][row][key]
                same(x, d, stat)
                assert (x['n'], x['c'], x['combinations']) == (27, 4, list(combos))
                checked += 1
        # (vii): the sampled schedules minus the greedy schedule, at 60 s (held) and at 10 s
        for b in (10, 60):
            x = S['vii'][str(b)][key]
            if not np.isfinite(smp_at[b][:, cols]).all():
                # 10 s: the greedy schedule of instance 7 in LS (with setups) comes at 12 s
                assert (key, b) == ('Setup', 10)
                assert x == dict(no_schedule_yet=1, n_cells=108, combinations=list(combos),
                                 held=False, held_after_s=POL_BUDGET)
                continue
            same(x, (100.0 * (smp_at[b] - sel) / sel)[:, cols], one)
            assert x['held'] is (b > POL_BUDGET) and x['combinations'] == list(combos)
            checked += 1
    assert checked == 2 * (2 * 2 + 1) + 1
    # the macros of a group carry its number and name it in words
    assert mac['TmdFiftyPolSmpVsPolSixtySetup'] == M.f2(S['vii']['60']['Setup']['mean'])
    lines = {n: line for line in txt.splitlines()
             for n in re.findall(r'\\newcommand\{\\([A-Za-z]+)\}', line)}
    for n, line in lines.items():
        if 'NoSetup' in n:
            assert 'combinations without setups' in line, line
        elif 'Setup' in n:
            assert 'combinations with setups' in line, line
    assert '| combinations without setups |' in md and '| combinations with setups |' in md


@pytest.mark.parametrize('origin', ['start', 'request'])
def test_the_groups_leave_every_existing_output_unchanged(rworld, monkeypatch, origin):
    """Run once with the groups and once without them (GROUPS empty, the outputs of the
    script before the groups were added, its CSV with the column group all): every macro
    line of the second run is a line of the first, in the same order; the JSON of the first
    without the entries the groups add is the second; the CSV of the first without the
    column group and without the rows of the groups is the second without that column."""
    run = rargs if origin == 'request' else args
    got = {}
    for label in ('groups', 'none'):
        if label == 'none':
            monkeypatch.setattr(M, 'GROUPS', ())
        assert M.main(run(rworld)) == 0
        J, mac, txt, rows, md = outputs(rworld)
        got[label] = (J, txt.splitlines(), rows)
    (Jg, Lg, Rg), (Jn, Ln, Rn) = got['groups'], got['none']
    pos = {line: k for k, line in enumerate(Lg)}
    assert all(line in pos for line in Ln)
    assert [pos[line] for line in Ln] == sorted(pos[line] for line in Ln)
    assert len(Lg) > len(Ln) and any('NoSetup' in x for x in Lg) and not any('Setup' in x for x in Ln)
    assert 'groups' in Jg and 'groups' not in Jn and without_groups(Jg) == Jn
    assert {x['group'] for x in Rn} == {'all'}
    drop = lambda rows: [{k: v for k, v in x.items() if k != 'group'} for x in rows]  # noqa: E731
    assert drop([x for x in Rg if x['group'] == 'all']) == drop(Rn)
    assert {x['group'] for x in Rg} == {'all', 'no_setups', 'setups'}
    k = [x['group'] for x in Rg].index('no_setups')
    assert {x['group'] for x in Rg[:k]} == {'all'} and 'all' not in {x['group'] for x in Rg[k:]}


@pytest.mark.parametrize('origin', ['start', 'request'])
def test_pool_mean_is_the_cell_weighted_mean_of_the_two_groups(rworld, origin):
    assert M.main((rargs if origin == 'request' else args)(rworld)) == 0
    J = outputs(rworld)[0]
    found = []

    def walk(x):
        if isinstance(x, dict):
            if all(isinstance(x.get(k), dict) and 'mean' in x[k] for k in ('Pool', *GROUP_OF)):
                found.append(x)
            for v in x.values():
                walk(v)
    walk(J['results'])
    for x in found:
        w = {k: x[k]['n'] * x[k]['c'] * x[k].get('k', 1) for k in GROUP_OF}
        assert x['Pool']['n'] == x['NoSetup']['n'] == x['Setup']['n']      # a complete world
        assert sum(w.values()) == x['Pool']['n'] * x['Pool']['c'] * x['Pool'].get('k', 1)
        want = sum(w[k] * x[k]['mean'] for k in w) / sum(w.values())
        assert x['Pool']['mean'] == pytest.approx(want, abs=1e-12)
    # start: (i) and (vi) 2 x 3 x 2 each, (ii) and (iii) 2 x 3 each, (iv) 2; request: the
    # same at 60 s only (2 x 2, 2 x 2, 2, 2, 2) and (vii) 1, (viii) 2
    assert len(found) == (38 if origin == 'start' else 17)


def synthetic(combos, request):
    """One size of four instances on the given combinations, for the analysis itself (as in
    the tests of the macro names above)."""
    match = [float(t) for t in range(601)]
    curve = M.log_grid(600, extra=M.BUDGETS)
    grid = sorted(set(match) | set(curve))
    n, C = 4, len(combos)
    rng = np.random.default_rng(0)
    g = np.array(grid)[:, None, None]
    rule = 500 + rng.integers(0, 50, (n, C)).astype(float)
    p = rule - 10
    D = dict(instances=list(range(n)), arms=list(M.P.ARMS), n_descriptors=n, rule=rule,
             pol_sel=p, all_pol=np.stack([p, p + 1, p - 1]), seeds_with_arrays=list(SEEDS))
    if not request:
        D['A'] = {arm: np.broadcast_to(rule - 5, (len(grid), n, C)).copy() for arm in M.P.ARMS}
        R = M.analyse({SIZE: D}, list(combos), [60, 600], 30, 0.05, grid, match, curve, 'sel')
    else:
        later = np.where(g >= 1.0, rule - 5 - (g > 100), np.nan)
        o = np.full((n, C), 2.0)
        D.update(A={arm: later.copy() for arm in M.P.ARMS},
                 A_present={arm: np.ones((n, C), dtype=bool) for arm in M.P.ARMS},
                 o_rule=o - 1, o_pol=o, o_res=o - 1, S=np.where(g >= 2.0, p - 3 + 0 * g, np.nan),
                 S_present=np.ones((n, C), dtype=bool), policy_budget_s=120.0, budgets=[60, 600])
        R = M.analyse_request({SIZE: D}, list(combos), 30, 0.05, grid, match, curve, 'sel')
    info = dict(setting='exact', selection_text='s302 0.8', cpsat_arm='ref', boot=30,
                alpha=0.05, origin='request' if request else 'start', campaigns=['some'],
                sel=SEL, k=3, horizon=600, primary=600)
    return R, M.macro_names(M.macro_lines(R, info))


def test_a_campaign_without_setups_has_no_setup_entry_and_no_setup_macro(world):
    no_setups = list(GROUP_OF['NoSetup'])
    assert M.main(args(world, '--combos', *no_setups, partial=True)) == 0
    out = json.load(open(world / 'partial' / 'timed_inference.json'))
    keys = set()

    def walk(x):
        if isinstance(x, dict):
            keys.update(x)
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(out)
    assert 'Setup' not in keys and 'NoSetup' in keys
    assert list(out['groups']['entries']) == ['NoSetup']
    assert out['groups']['entries']['NoSetup']['combinations'] == no_setups
    blk = out['results'][SIZE]['ii']['cpsat']['60']
    assert blk['NoSetup'] == blk['Pool'] and blk['Pool']['combinations'] == no_setups
    rows = list(csv.DictReader(open(world / 'partial' / 'timed_curves.csv')))
    assert {x['group'] for x in rows} == {'all', 'no_setups'}
    # macros are written in full mode only, which needs the eight combinations: the same
    # campaign through the analysis and the macro lines
    # a group with one combination in the campaign is absent as well, as Pool would be
    for combos in (no_setups, ['N', 'L', 'S']):
        for request in (False, True):
            R, names = synthetic(combos, request)
            S = R[SIZE]
            assert not [n for n in names if re.search(r'(?<!No)Setup', n)]
            assert [n for n in names if 'NoSetup' in n]
            assert 'Setup' not in S['iv']['selected'] and 'NoSetup' in S['iv']['selected']
            if request:
                assert 'Setup' not in S['ix'] and 'NoSetup' in S['ix']
                assert list(S['group_curve_start_s']) == ['NoSetup']
            assert {c['group'] for c in S['group_curves']} == {'no_setups'}
            M.check_macro_names(names, os.path.join(M.ROOT, 'manuscript', 'macros_timed.tex'))


# ---------------------------------------------------------------------------
# the decision table: bold markers from the printed values

def macro_values(txt):
    """{name: value} of a macro file, a value with one level of braces included (562{,}796)."""
    return dict(re.findall(r'\\newcommand\{\\([A-Za-z]+)\}\{((?:[^{}]|\{[^{}]*\})*)\}', txt))


def set_cpsat_rule_final(root, final):
    """Every record of CP-SAT started from the rules ends at final(i, j) from 50 s of the
    solver call on."""
    camp = camp_of(root)
    for c in camp.cells(next(s for s in camp.segments if s.arm == 'cpsat_rule')):
        j = M.COMBOS.index(c.combo)
        R0 = r0(c.idx, j)
        write_record(camp, c, j, objective=final(c.idx, j),
                     anytime=[[3.0, R0 - 5, 100.0], [50.0, final(c.idx, j), 100.0]])


DECISION_COLS = ('rules', 'policy_sample', 'memetic_rule', 'cpsat_rule', 'memetic_policy',
                 'cpsat_policy')


def decision_macro_names(bw, key, sw='Fifty'):
    """{column: (value macro, bold macro)} of one decision row at 50x25n30 (sw FiftyCommon:
    of the table on the common instances), written out here so that the test does not take
    the names from the script."""
    return dict(rules=(f'Tmd{sw}RuleVsPol{key}Sel', f'Tmd{sw}RuleVsPol{bw}{key}SelBold'),
                policy_sample=(f'Tmd{sw}PolSmpVsPol{bw}{key}',
                               f'Tmd{sw}PolSmpVsPol{bw}{key}Bold'),
                memetic_rule=(f'Tmd{sw}MemRuleStartVsPol{bw}{key}Sel',
                              f'Tmd{sw}MemRuleStartVsPol{bw}{key}SelBold'),
                cpsat_rule=(f'Tmd{sw}CpRuleStartVsPol{bw}{key}Sel',
                            f'Tmd{sw}CpRuleStartVsPol{bw}{key}SelBold'),
                memetic_policy=(f'Tmd{sw}MemPolStartVsPol{bw}{key}',
                                f'Tmd{sw}MemPolStartVsPol{bw}{key}Bold'),
                cpsat_policy=(f'Tmd{sw}CpPolStartVsPol{bw}{key}',
                              f'Tmd{sw}CpPolStartVsPol{bw}{key}Bold'))


def recompute_bold(root, S, mac):
    """Every row of the decision table (budget x Pool and each group) against the macros: the
    bold marker of each of the six columns is \\bfseries exactly when the column's printed
    value macro holds the lowest value of the row, and the JSON row states the same values and
    bold columns. Returns {(budget, entry): (printed values, bold columns)}."""
    t_all = {(i, j): p97_records(root, i, j)[0]['t_all'] for i in EVID for j in range(8)}
    members = dict({g[0]: g[2] for g in M.GROUPS}, Pool=M.COMBOS)
    out = {}
    for b in (1, 10, 60):
        bw = M.number_word(b)
        for key, combos in members.items():
            ready = max(t_all[i, M.COMBOS.index(r)] for i in EVID for r in combos) <= b + 1e-9
            names = decision_macro_names(bw, key)
            assert tuple(names) == DECISION_COLS
            printed = {col: (mac.get(v) if col != 'rules' or ready else None)
                       for col, (v, _) in names.items()}
            nums = {col: float(v) for col, v in printed.items() if v is not None}
            want = {col for col, v in nums.items() if v == min(nums.values())} if nums else set()
            got = {col for col, (_, bn) in names.items() if mac[bn] == M.BOLD}
            assert got == want, (b, key, printed)
            assert all(mac[bn] in ('', M.BOLD) for _, bn in names.values())
            row = S['decision'][str(b)][key]
            assert row['values'] == printed and set(row['bold']) == want
            out[b, key] = (printed, want)
    return out


def test_bold_markers_equal_a_recomputation_from_the_printed_values(rworld):
    """Every row of the decision table: the bold markers over the six columns equal a
    recomputation from the printed value macros (recompute_bold). CP-SAT started from the
    rules is set to end where the memetic algorithm started from the rules ends, 0.001 longer
    on one cell, so at 60 s the two columns print the same lowest value while their unrounded
    means differ: both are bold, and the searches started from the policy's schedule (which
    end higher) are not. At 10 s the policy has no schedule on one cell with setups, so that
    row has dashes and the rules win alone; at 1 s no column has a number."""
    set_cpsat_rule_final(rworld, lambda i, j: r0(i, j) - 40 + (0.001 if (i, j) == (3, 0) else 0))
    assert M.main(rargs(rworld)) == 0
    J, _, txt, _, md = outputs(rworld)
    mac = macro_values(txt)
    S = J['results'][SIZE]
    rows = recompute_bold(rworld, S, mac)
    cases = dict(tie=0, dash=0, empty=0)
    for (b, key), (printed, want) in rows.items():
        nums = [v for v in printed.values() if v is not None]
        cases['tie'] += len(want) > 1
        cases['dash'] += bool(nums) and len(nums) < len(DECISION_COLS)
        cases['empty'] += not nums
        if len(want) > 1 and key in ('Pool', 'NoSetup'):
            # a tie in print, not in the unrounded values
            vi = S['vi']
            assert vi['cpsat'][str(b)]['selected'][key]['mean'] \
                > vi['memetic'][str(b)]['selected'][key]['mean']
    assert cases['tie'] == 3 and cases['dash'] == 2 and cases['empty'] == 3
    assert set(S['decision']['60']['Setup']['bold']) == {'memetic_rule', 'cpsat_rule'}
    assert S['decision']['10']['Setup']['bold'] == ['rules']
    assert S['decision']['10']['Setup']['values']['policy_sample'] is None
    t_all = [p97_records(rworld, i, j)[0]['t_all'] for i in EVID for j in range(8)]
    none = dict.fromkeys(DECISION_COLS)
    assert S['decision']['1']['Pool'] == dict(
        values=none, bold=[], rules_from_s=pytest.approx(max(t_all)), combinations=list(M.COMBOS),
        n_instances=none, instances=none, combinations_used=none, same_instances=True)
    # the Markdown has one line per row, the bold values in bold
    lines = [x for x in md.splitlines() if re.match(r'\| (1|10|60) \| ', x)]
    assert len(lines) == 3 * 3
    assert '| 10 | combinations with setups | **' in md
    assert J['decision_table']['columns']['rules'] == 'best of the four dispatching rules'
    assert list(J['decision_table']['columns']) == list(DECISION_COLS)


def set_memetic_policy_final(root, drop):
    """Every record of the memetic algorithm started from the policy's schedule finds a
    schedule `drop` below the policy's at 10 s of its process (after PS - 10 at 5 s)."""
    camp = camp_of(root)
    for c in camp.cells(next(s for s in camp.segments if s.arm == 'memetic_policy')):
        j = M.COMBOS.index(c.combo)
        PS = pol(SEL, c.idx, j)
        write_record(camp, c, j, makespan=PS - drop,
                     anytime=[[0.0, 1.0, 1.0, PS], [4.0, 5.0, 5.0, PS - 10],
                              [9.0, 10.0, 10.0, PS - drop]])


def test_bold_marker_moves_to_a_search_started_from_the_policy_when_it_is_lowest(rworld):
    """The memetic algorithm started from the policy's schedule is set to end 60 below the
    policy, lower than every other column at 60 s: in each row at 60 s it alone is bold, and
    the column that is bold in the four-column table (the memetic algorithm started from the
    rules, R0 - 40 = policy - 20) loses its marker. Every other row passes the recomputation
    over the six columns."""
    set_memetic_policy_final(rworld, 60)
    assert M.main(rargs(rworld)) == 0
    J, _, txt, _, md = outputs(rworld)
    mac = macro_values(txt)
    S = J['results'][SIZE]
    rows = recompute_bold(rworld, S, mac)
    for key in ('Pool', 'NoSetup', 'Setup'):
        printed, want = rows[60, key]
        assert want == {'memetic_policy'}, (key, printed)
        assert S['decision']['60'][key]['bold'] == ['memetic_policy']
        names = decision_macro_names('Sixty', key)
        assert mac[names['memetic_policy'][1]] == M.BOLD
        assert mac[names['memetic_rule'][1]] == ''
        # the four columns of the earlier table alone would bold the rule-started search
        four = {c: float(printed[c]) for c in DECISION_COLS[:4] if printed[c] is not None}
        assert min(four, key=four.get) == 'memetic_rule'
        assert float(printed['memetic_policy']) < min(four.values())
    assert any(f'**{S["decision"]["60"]["Pool"]["values"]["memetic_policy"]}** (27)' in x
               for x in md.splitlines() if x.startswith('| 60 | Pool |'))


def test_rule_column_takes_part_from_the_time_its_schedules_exist():
    """The rule column has one value per entry; it has a number in a budget's row when the
    rules have a schedule on every cell of the entry (o_rule <= budget), in no row before,
    whatever the other columns hold."""
    combos = list(M.COMBOS)
    o_rule = np.full((3, 8), 2.0)
    o_rule[1, combos.index('LS')] = 12.0                # one late cell with setups
    o_rule[2, combos.index('N')] = np.nan               # no rule record: left out
    entry = lambda m: dict(mean=m, lo=m, hi=m)          # noqa: E731
    S = dict(budgets=[1, 10, 12], iv=dict(selected={k: entry(5.0) for k in ('Pool', 'NoSetup',
                                                                             'Setup')}),
             vii={'10': {'NoSetup': entry(5.004), 'Setup': entry(-1.0)},
                  '12': {'NoSetup': entry(4.9)}},
             vi=dict(memetic={'12': dict(selected=dict(Setup=dict(no_schedule_yet=1,
                                                                  n_cells=12)))}))
    D = M.decision_table(S, o_rule, combos)
    assert D['1']['NoSetup']['values']['rules'] is None and D['1']['NoSetup']['bold'] == []
    assert D['10']['NoSetup']['values']['rules'] == '+5.00'
    assert D['10']['NoSetup']['bold'] == ['rules', 'policy_sample']   # +5.00 both, in print
    assert D['10']['Setup']['values']['rules'] is None
    assert D['10']['Setup']['bold'] == ['policy_sample']
    assert D['10']['Pool']['rules_from_s'] == D['10']['Setup']['rules_from_s'] == 12.0
    assert D['12']['Setup']['bold'] == ['rules'] and D['12']['NoSetup']['bold'] == ['policy_sample']
    assert D['12']['Setup']['values']['memetic_rule'] is None
    assert D['12']['NoSetup']['rules_from_s'] == 2.0


def test_policy_started_columns_equal_comparison_ii_of_the_same_run(rworld):
    """The two columns of the searches started from the policy's schedule, in every row of
    the decision table on the request clock: the printed value of comparison (ii) of the same
    run (its value macro, the same text), null and no macro where (ii) has no number; the
    instances and combinations of each value are those of (ii), one set with the other
    columns of the row."""
    assert M.main(rargs(rworld)) == 0
    J, mac, txt, _, md = outputs(rworld)
    S = J['results'][SIZE]
    seen = {col: {'number': 0, 'null': 0} for col in ('memetic_policy', 'cpsat_policy')}
    for b in (1, 10, 60):
        bw = M.number_word(b)
        for key in ('Pool', 'NoSetup', 'Setup'):
            row = S['decision'][str(b)][key]
            assert row['same_instances'] is True
            for m, col, word in (('memetic', 'memetic_policy', 'Mem'),
                                 ('cpsat', 'cpsat_policy', 'Cp')):
                x = S['ii'][m][str(b)][key]
                name = f'TmdFifty{word}PolStartVsPol{bw}{key}'
                assert decision_macro_names(bw, key)[col][0] == name
                if 'mean' in x:
                    assert row['values'][col] == M.f2(x['mean']) == mac[name]
                    assert row['n_instances'][col] == x['n'] == 27
                    assert row['instances'][col] == EVID
                    assert row['combinations_used'][col] == x['combinations']
                    seen[col]['number'] += 1
                else:
                    assert 'no_schedule_yet' in x
                    assert row['values'][col] is None and name not in mac
                    assert row['n_instances'][col] is None and row['instances'][col] is None
                    seen[col]['null'] += 1
    # 1 s: no schedule anywhere; 10 s: the late greedy schedule in LS empties Pool and Setup
    assert seen == {col: {'number': 4, 'null': 5} for col in seen}
    assert J['decision_table']['columns']['memetic_policy'] == \
        "memetic algorithm started from the policy's schedule"
    assert J['decision_table']['columns']['cpsat_policy'] == \
        "CP-SAT started from the policy's schedule"
    assert "memetic algorithm started from the policy's schedule | CP-SAT started from the " \
           "policy's schedule | lowest | same instances |" in md


def test_decision_mismatches_names_a_row_whose_columns_have_different_instances():
    """decision_mismatches: one text for a row whose columns with a number are means over
    different instances, naming each column's instances; nothing for a row with one set,
    nothing without the cells (a row built without `used`)."""
    combos = list(M.COMBOS)
    entry = lambda m: dict(mean=m, lo=m, hi=m)          # noqa: E731
    S = dict(budgets=[10], instances=[3, 4, 5], iv=dict(selected={'NoSetup': entry(5.0)}),
             vii={'10': {'NoSetup': entry(-1.0)}}, vi={},
             ii=dict(memetic={'10': {'NoSetup': entry(-3.0)}},
                     cpsat={'10': {'NoSetup': entry(-2.0)}}))
    o = np.full((3, 8), 2.0)
    four = ['N', 'L', 'W', 'LW']
    used = {('rules', None): {'NoSetup': ([0, 1], four)},
            ('policy_sample', '10'): {'NoSetup': ([0, 1], four)},
            ('memetic_policy', '10'): {'NoSetup': ([0, 1], four)},
            ('cpsat_policy', '10'): {'NoSetup': ([0, 1], four)}}
    row = M.decision_table(S, o, combos, used)['10']['NoSetup']
    assert row['same_instances'] is True and row['instances']['rules'] == [3, 4]
    assert M.decision_mismatches({SIZE: dict(decision={'10': {'NoSetup': row}})}) == []
    used[('memetic_policy', '10')] = {'NoSetup': ([0], four)}
    row = M.decision_table(S, o, combos, used)['10']['NoSetup']
    assert row['same_instances'] is False and row['n_instances']['memetic_policy'] == 1
    assert row['bold'] == ['memetic_policy']                   # the values are unaffected
    bad = M.decision_mismatches({SIZE: dict(decision={'10': {'NoSetup': row}})})
    assert len(bad) == 1
    assert bad[0].startswith(f'decision table, {SIZE}, 10 s, combinations without setups: the '
                             'columns with a number are not means over one set of instances')
    assert 'memetic_policy 1 instances (3), combinations N,L,W,LW' in bad[0]
    assert 'rules 2 instances (3-4)' in bad[0] and 'memetic_rule' not in bad[0]
    # a row built without the cells states no set and is never a mismatch
    plain = M.decision_table(S, o, combos)['10']['NoSetup']
    assert 'same_instances' not in plain
    assert M.decision_mismatches({SIZE: dict(decision={'10': {'NoSetup': plain}})}) == []


def drop_first_instance(monkeypatch, col, budget, key):
    """decision_table as the script runs it, with the first instance of one value's cells
    left out (in memory, after every number is computed): a row with a mismatch."""
    real = M.decision_table

    def patched(S, o_rule, combos, used=None):
        cells = dict(used)
        rows, cs = cells[(col, budget)][key]
        cells[(col, budget)] = dict(cells[(col, budget)], **{key: (rows[1:], cs)})
        return real(S, o_rule, combos, cells)
    monkeypatch.setattr(M, 'decision_table', patched)


def test_full_mode_refuses_a_decision_row_over_different_instances(rworld, capsys,
                                                                    monkeypatch):
    """With the cells of one value of the decision table cut by one instance, the full mode
    refuses with the text of decision_mismatches and leaves no macro file; the partial mode
    marks the row and goes on."""
    drop_first_instance(monkeypatch, 'memetic_policy', '60', 'Pool')
    err = rrefused(rworld, capsys)
    assert f'decision table, {SIZE}, 60 s, Pool: the columns with a number are not means over ' \
           'one set of instances' in err
    assert 'memetic_policy 26 instances (4-29)' in err and 'rules 27 instances (3-29)' in err
    assert not (rworld / 'out' / 'results' / 'timed_inference.json').exists()
    assert M.main(rargs(rworld, partial=True)) == 0
    J = json.load(open(rworld / 'partial' / 'timed_inference.json'))
    row = J['results'][SIZE]['decision']['60']['Pool']
    assert row['same_instances'] is False and row['n_instances']['memetic_policy'] == 26
    md = open(rworld / 'partial' / 'timed_inference.md').read()
    line = next(x for x in md.splitlines() if x.startswith('| 60 | Pool |'))
    assert line.endswith('| no |') and '(26)' in line and '(27)' in line


# ---------------------------------------------------------------------------
# the campaign as designed: instance ranges in the tokens, and the decision table on the
# instances common to every method of a size

def test_a_range_of_the_policy_token_is_its_design(rworld, capsys):
    """Request clock, every token on 3-11, the policy records too: the full mode reads the
    campaign as designed and states it; the methods share their instances, so there is no
    second decision table. A cell of the plan outside the range of the policy token still
    refuses (it has no policy record)."""
    nine = ' '.join(f'{t}:3-11' for t in ARMS4.split()) + f' {POL_TOKEN}:3-11'
    assert M.main(rargs(rworld, arms=nine)) == 0
    J, mac, txt, rows, md = outputs(rworld)
    S = J['results'][SIZE]
    assert S['instances'] == list(range(3, 12)) and J['gaps'] == []
    assert S['design'][M.POLICY_ARM] == dict(instances=list(range(3, 12)), range_given=True)
    assert all(d['range_given'] and d['instances'] == list(range(3, 12))
               for d in S['design'].values())
    assert {mac[f'TmdFifty{w}InstN'] for w in ('', 'MemRuleStart', 'CpRuleStart', 'MemPolStart',
                                               'CpPolStart', 'PolSmp')} == {'9'}
    assert J['new_records_valid'] == {M.RULE_ARM: 72, M.POLICY_ARM: 72}
    assert 'decision_common' not in S and not [n for n in mac if 'Common' in n]
    assert all(row['same_instances'] is True and 'by_design' not in row
               for rr in S['decision'].values() for row in rr.values())
    assert S['vii']['60']['Pool']['n'] == 9 and S['verdict']['memetic']['selected']['rows']['N']['n'] == 9
    err = rrefused(rworld, capsys, arms=f'{ARMS4} {POL_TOKEN}:3-11')
    assert f'{M.POLICY_ARM}@30/{SIZE}/N/012: not in the plan of {POL_TOKEN}:3-11' in err


COMMON_ARMS = 'memetic_rule@600 memetic_policy@600 cpsat_rule@600:3-12 cpsat_policy@600'


def test_decision_table_on_the_common_instances(rworld):
    """CP-SAT started from the rules on 3-12, the other methods on 3-29: the rows of the
    decision table are means over different instances by design (by_design), which the full
    mode keeps, and a second table on the common instances 3-12 holds the same six columns.
    It equals the decision table of the campaign whose every token is on 3-12: rows (values,
    bold columns, instances, combinations), value macros and bold markers, whose names carry
    Common after the size word."""
    assert M.main(rargs(rworld, arms=f'{COMMON_ARMS} {POL_TOKEN}')) == 0
    J, mac, txt, rows, md = outputs(rworld)
    S = J['results'][SIZE]
    dc = S['decision_common']
    assert dc['instances'] == list(range(3, 13)) and dc['n_instances'] == 10
    assert mac['TmdFiftyCommonInstN'] == mac['TmdFiftyCpRuleStartInstN'] == '10'
    assert mac['TmdFiftyInstN'] == mac['TmdFiftyMemRuleStartInstN'] == '27'
    row = S['decision']['60']['Pool']
    assert row['same_instances'] is False and row['by_design'] is True
    assert row['n_instances'] == dict(rules=27, policy_sample=27, memetic_rule=27, cpsat_rule=10,
                                      memetic_policy=27, cpsat_policy=27)
    assert next(x for x in md.splitlines() if x.startswith('| 60 | Pool |')).endswith(
        '| no, by design |')
    assert '### Decision table on the instances common to every method of the size (3-12, 10 ' \
           'instances)' in md
    lines = {n: x for x in txt.splitlines() for n in re.findall(r'\\newcommand\{\\([A-Za-z]+)\}', x)}
    assert 'on the 10 instances common to every method of the size (3-12)' in \
        lines['TmdFiftyCommonMemRuleStartVsPolSixtyPoolSel']
    assert 'since the request' in lines['TmdFiftyCommonMemRuleStartVsPolSixtyPoolSel']
    assert 'each over the instances of its design' in lines['TmdFiftyCpRuleStartVsPolSixtyPoolSelBold']
    ten = ' '.join(f'{t.partition(":")[0]}:3-12' for t in COMMON_ARMS.split()) + \
        f' {POL_TOKEN}:3-12'
    assert M.main(rargs(rworld, arms=ten)) == 0
    J2, mac2, *_ = outputs(rworld)
    S2 = J2['results'][SIZE]
    assert 'decision_common' not in S2
    seen = 0
    for b, rr in S2['decision'].items():
        bw = M.number_word(int(b))
        for key, row2 in rr.items():
            assert dc['rows'][b][key] == row2, (b, key)
            ours, theirs = (decision_macro_names(bw, key, sw) for sw in ('FiftyCommon', 'Fifty'))
            for col in DECISION_COLS:
                assert mac[ours[col][1]] == mac2[theirs[col][1]], (b, key, col)
                assert mac.get(ours[col][0]) == mac2.get(theirs[col][0]), (b, key, col)
                seen += ours[col][0] in mac
    # value macros named by the rows: 60 s, 3 rows x 6 columns; 10 s, the row without setups
    # and the rules of the other two (the late greedy schedule of LS 7 lies inside 3-12); 1 s,
    # the rules only, whose one value per entry is named at every budget
    assert seen == 18 + 6 + 2 + 3 and any(mac[n] == M.BOLD for n in mac
                                          if n.startswith('TmdFiftyCommon'))


def test_the_common_table_leaves_every_other_output_unchanged(rworld, monkeypatch):
    """The run with the table on the common instances and the same run without it (the
    design of the columns switched off), both partial with their macro files: the JSON
    without the keys the table adds is the same, every macro of the second run has its value
    in the first, the first adds only macros of the table (Common after the size word), and
    the CSV is the same bytes."""
    got = {}
    for label in ('with', 'without'):
        if label == 'without':
            monkeypatch.setattr(M, 'column_design', lambda instances, design: {})
        assert M.main(rargs(rworld, '--partial-macros', arms=f'{COMMON_ARMS} {POL_TOKEN}',
                            partial=True)) == 0
        part = rworld / 'partial'
        got[label] = (json.load(open(part / 'timed_inference.json')),
                      macro_values(open(part / 'macros_timed.tex').read()),
                      open(part / 'timed_curves.csv', 'rb').read())
    (Jw, Mw, Cw), (Jn, Mn, Cn) = got['with'], got['without']
    added = {'decision_common', 'by_design', 'common'}

    def strip(x):
        if isinstance(x, dict):
            return {k: strip(v) for k, v in x.items() if k not in added}
        return [strip(v) for v in x] if isinstance(x, list) else x
    assert 'decision_common' in Jw['results'][SIZE] and 'decision_common' not in Jn['results'][SIZE]
    assert strip(Jw) == Jn
    assert all(Mw[k] == v for k, v in Mn.items())
    new = set(Mw) - set(Mn)
    assert new and all(n.startswith('TmdFiftyCommon') for n in new)
    assert Cw == Cn


def test_full_mode_refuses_a_column_that_is_not_its_design(rworld, capsys, monkeypatch):
    """At a size with the table on the common instances, a row of the first table whose
    column is not a mean over its design cells, and a row of the second table that is not
    one set, are refused in full mode (one instance of CP-SAT cut from both, in memory)."""
    drop_first_instance(monkeypatch, 'cpsat_rule', '60', 'Pool')
    err = rrefused(rworld, capsys, arms=f'{COMMON_ARMS} {POL_TOKEN}')
    assert f'decision table, {SIZE}, 60 s, Pool: the columns with a number are not means over ' \
           'one set of instances' in err and 'or over the instances of its design' in err
    assert 'cpsat_rule 9 instances (4-12)' in err
    assert f'decision table on the common instances, {SIZE}, 60 s, Pool' in err


def test_partial_macros_writes_the_full_mode_macros_into_the_partial_folder(rworld, capsys):
    """--partial-macros: refused without --partial; with it the partial folder also holds
    macros_timed.tex, whose lines are those of the full mode on the same records after one
    header line that marks the file partial; a partial run without it writes no macro file;
    a refusal of a partial run with it removes the macro file."""
    assert M.main(rargs(rworld, '--partial-macros')) == 2
    assert '--partial-macros is accepted only with --partial' in capsys.readouterr().err
    assert not (rworld / 'out').exists() and not (rworld / 'partial').exists()
    three = ['timed_curves.csv', 'timed_inference.json', 'timed_inference.md']
    assert M.main(rargs(rworld, partial=True)) == 0
    assert sorted(os.listdir(rworld / 'partial')) == three
    assert M.main(rargs(rworld, '--partial-macros', partial=True)) == 0
    assert sorted(os.listdir(rworld / 'partial')) == ['macros_timed.tex'] + three
    assert not (rworld / 'out').exists()
    part = open(rworld / 'partial' / 'macros_timed.tex').read().splitlines()
    assert part[1] == ('% Partial run (partial, --partial-macros): built from the cells that '
                       'exist, for a test build; not the macro file of record.')
    assert M.main(rargs(rworld)) == 0
    full = open(rworld / 'out' / 'manuscript' / 'macros_timed.tex').read().splitlines()
    assert [part[0]] + part[2:] == full
    assert not any(x.startswith('% Partial run') for x in full)
    assert any(x.startswith('\\newcommand{\\TmdFiftyMemPolStartVsPolSixtyPoolBold}') for x in part)
    # a refusal (a misshapen rule array refuses in partial mode too) removes the macro file only
    np.save(rworld / 'res' / f'{SIZE}_xmask' / 'Result_SPT_N.npy', np.zeros((29, 2)) + 1)
    assert M.main(rargs(rworld, '--partial-macros', partial=True)) == 1
    assert 'has shape (29, 2), want (30, 2)' in capsys.readouterr().err
    assert sorted(os.listdir(rworld / 'partial')) == three


# ---------------------------------------------------------------------------
# the size of the solver's model

def as_pieces(log):
    """The log as the solver hands it to p85: the initial model's header and its statistics
    in one piece of several lines (as in the records of the configuration selection)."""
    return log[:4] + ['\n'.join(log[4:8])] + log[8:]


def head_variables(rec):
    """The test's own reading of a synthetic log: the line after the initial model."""
    head = rec.get('solver_log_head')
    if not head:
        return None
    lines = '\n'.join([head] if isinstance(head, str) else head).splitlines()
    k = next(k for k, x in enumerate(lines) if x.startswith('Initial optimization model'))
    m = re.match(r"#Variables: ([\d']+) ", lines[k + 1])
    return int(m.group(1).replace("'", '')) if m else None


def test_variable_count_parsing_and_printing():
    log = solver_log(562796, presolved=207000)
    assert M.initial_variables(dict(solver_log_head=log)) == (562796, None)
    assert M.initial_variables(dict(solver_log_head='\n'.join(log))) == (562796, None)
    # one piece of the log holds the header and the variables line (p85's records)
    assert '\n#Variables: 562\'796 ' in as_pieces(log)[4]
    assert M.initial_variables(dict(solver_log_head=as_pieces(log))) == (562796, None)
    # the presolved model's count is never taken
    assert M.initial_variables(dict(solver_log_head=log[:5] + log[6:]))[0] is None
    assert M.initial_variables({})[1] == 'no solver_log_head'
    assert M.initial_variables(dict(solver_log_head=[]))[1] == 'no solver_log_head'
    # no initial model; a variables line only after the next model; a line that is not read
    assert 'no line starting' in M.initial_variables(dict(solver_log_head=log[:4]))[1]
    late = log[:5] + log[7:]
    assert M.initial_variables(dict(solver_log_head=late))[0] is None
    assert "no '#Variables:' line" in M.initial_variables(dict(solver_log_head=late))[1]
    bad = log[:5] + ['#Variables: many'] + log[6:]
    assert M.initial_variables(dict(solver_log_head=bad))[0] is None
    assert M.count_text(562796) == '562{,}796' and M.count_text(999) == '999'
    assert M.count_text(1413656) == '1{,}413{,}656' and M.count_text(19249.0) == '19{,}249'
    rec = dict(solver_log_head=log, build_s=12.055, peak_rss_gb=15.593,
               anytime=[[44.717, 517.0, 201.0], [60.0, 515.0, 201.0]])
    assert M.model_fields(rec) == dict(variables=562796, variables_reason=None, build_s=12.055,
                                       peak_rss_gb=15.593, first_incumbent_s=44.717)
    assert M.model_fields(dict(anytime=[]))['first_incumbent_s'] is None


def test_solver_model_equals_a_recomputation_from_the_records(rworld):
    """(x) per combination, Pool and each group: median, minimum and maximum of the records
    of the group's cells, recomputed from the record files of CP-SAT started from the rules;
    a record without solver_log_head and one whose log has no variables line before the
    next model are counted and listed and leave the count out; the records of the
    policy-started arm do not enter; the same numbers on both clocks."""
    camp = camp_of(rworld)
    for arm, wild in (('cpsat_rule', False), ('cpsat_policy', True)):
        for c in camp.cells(next(s for s in camp.segments if s.arm == arm)):
            j = M.COMBOS.index(c.combo)
            R0 = r0(c.idx, j)
            if wild:                    # values that would show if this arm entered
                write_record(camp, c, j, build_s=99.0, peak_rss_gb=99.0,
                             solver_log_head=solver_log(7))
                continue
            first = 2.0 + 0.1 * c.idx + 0.25 * j
            write_record(camp, c, j, build_s=round(4.0 + 0.05 * c.idx + 0.5 * j, 3),
                         anytime=[[first, R0 - 5, 100.0], [50.0, R0 - 25, 100.0]])
    gone = camp.record(cell(camp, 'cpsat_rule', 'S', 5))
    rec = json.load(open(gone))
    del rec['solver_log_head']
    json.dump(rec, open(gone, 'w'))
    late = camp.record(cell(camp, 'cpsat_rule', 'LSW', 9))
    rec = json.load(open(late))
    rec['solver_log_head'] = rec['solver_log_head'][:5] + rec['solver_log_head'][7:]
    json.dump(rec, open(late, 'w'))
    text = camp.record(cell(camp, 'cpsat_rule', 'N', 4))            # a log stored as one text
    rec = json.load(open(text))
    rec['solver_log_head'] = '\n'.join(rec['solver_log_head'])
    json.dump(rec, open(text, 'w'))
    for i in EVID[::2]:                         # logs in pieces of several lines, as p85 stores
        p = camp.record(cell(camp, 'cpsat_rule', 'W', i))
        rec = json.load(open(p))
        rec['solver_log_head'] = as_pieces(rec['solver_log_head'])
        json.dump(rec, open(p, 'w'))
    got = {}
    for origin, run in (('start', args), ('request', rargs)):
        assert M.main(run(rworld)) == 0
        J, _, txt, _, md = outputs(rworld)
        got[origin] = (J['results'][SIZE]['solver_model'], macro_values(txt), md, txt)
    sm, mac, md, txt = got['start']
    assert got['request'][0] == sm
    recs = {(i, r): json.load(open(camp.record(cell(camp, 'cpsat_rule', r, i))))
            for i in EVID for r in M.COMBOS}
    fields = dict(variables=head_variables, build_s=lambda x: x['build_s'],
                  peak_rss_gb=lambda x: x['peak_rss_gb'],
                  first_incumbent_s=lambda x: x['anytime'][0][0])
    text_of = dict(variables=lambda v: f'{v:,.0f}'.replace(',', '{,}'),
                   build_s=lambda v: f'{v:.2f}', peak_rss_gb=lambda v: f'{v:.1f}',
                   first_incumbent_s=lambda v: f'{v:.1f}')
    word = dict(variables='Vars', build_s='Build', peak_rss_gb='Memory',
                first_incumbent_s='FirstIncumbent')
    entries = dict({r: [r] for r in M.COMBOS}, Pool=list(M.COMBOS), **GROUP_OF)
    assert list(sm['entries']) == list(entries)
    for key, combos in entries.items():
        x = sm['entries'][key]
        cells = [recs[i, r] for i in EVID for r in combos]
        assert x['n_cells'] == len(cells) and x['combinations'] == list(combos)
        assert mac[f'TmdFiftyCpModelCells{key}'] == str(len(cells))
        for f, read in fields.items():
            vals = [v for v in (read(c) for c in cells) if v is not None]
            want = dict(median=float(np.median(vals)), min=float(min(vals)),
                        max=float(max(vals)), n_cells=len(vals))
            assert x[f] == want, (key, f)
            for k, w in (('median', 'Median'), ('min', 'Min'), ('max', 'Max')):
                assert mac[f'TmdFiftyCp{word[f]}{w}{key}'] == text_of[f](want[k])
        miss = sum(head_variables(c) is None for c in cells)
        assert x['variables_missing'] == miss and mac[f'TmdFiftyCpVarsMissing{key}'] == str(miss)
        assert x['variables']['n_cells'] == len(cells) - miss
    assert sm['entries']['Setup']['variables_missing'] == 2
    assert sm['entries']['NoSetup']['variables_missing'] == 0
    assert sm['entries']['S']['variables_missing'] == sm['entries']['LSW']['variables_missing'] == 1
    assert sm['entries']['N']['variables']['n_cells'] == 27        # the text log is read
    assert sm['entries']['W']['variables']['n_cells'] == 27        # and the logs in pieces
    reasons = {(c['combination'], c['instance']): c['reason'] for c in sm['missing']['variables']}
    assert reasons == {('S', 5): 'no solver_log_head',
                       ('LSW', 9): "no '#Variables:' line after the line starting "
                                   "'Initial optimization model'"}
    assert not any(sm['missing'][k] for k in ('build_s', 'peak_rss_gb', 'first_incumbent_s'))
    assert sm['n_cells'] == 216 and sm['entries']['Pool']['build_s']['max'] < 99
    # a count prints with grouped digits; the macros of a group name it in words
    assert re.fullmatch(r'\d{3}\{,\}\d{3}', mac['TmdFiftyCpVarsMedianSetup'])
    assert 'no value of the variables of the initial model' in md and '(no solver_log_head)' in md
    for line in txt.splitlines():
        name = re.match(r'\\newcommand\{\\(TmdFiftyCp(ModelCells|Vars|Build|Memory|'
                        r'FirstIncumbent)[A-Za-z]*)\}', line)
        if name and name.group(1).endswith('NoSetup'):
            assert 'combinations without setups' in line
        elif name and name.group(1).endswith('Setup'):
            assert 'combinations with setups' in line


@pytest.mark.parametrize('origin', ['start', 'request'])
def test_decision_table_and_model_size_leave_every_existing_output_unchanged(
        rworld, monkeypatch, origin):
    """Run once as is and once with the two additions switched off (no decision rows, no
    model cells: the outputs of the script before they were added): every macro line of the
    second run is a line of the first, in the same order; the JSON of the first without the
    keys they add is the second; the CSV is the same bytes."""
    run = rargs if origin == 'request' else args
    got = {}
    for label in ('with', 'without'):
        if label == 'without':
            monkeypatch.setattr(M, 'MODEL_ARM', None)
            monkeypatch.setattr(M, 'decision_table', lambda *a: {})
        assert M.main(run(rworld)) == 0
        J, _, txt, _, _ = outputs(rworld)
        got[label] = (J, txt.splitlines(),
                      open(rworld / 'out' / 'results' / 'timed_curves.csv', 'rb').read())
    (Jw, Lw, Cw), (Jn, Ln, Cn) = got['with'], got['without']
    pos = {line: k for k, line in enumerate(Lw)}
    assert all(line in pos for line in Ln)
    assert [pos[line] for line in Ln] == sorted(pos[line] for line in Ln)
    new = [x for x in Lw if x not in set(Ln)]
    assert new and all(re.match(r'\\newcommand\{\\TmdFifty(Cp(ModelCells|Vars|Build|Memory|'
                                r'FirstIncumbent)|[A-Za-z]+Bold\})', x) for x in new)
    assert any('Bold}' in x for x in new) == (origin == 'request')
    added = {'decision', 'solver_model', 'decision_table'}

    def strip(x):
        if isinstance(x, dict):
            return {k: strip(v) for k, v in x.items() if k not in added}
        return [strip(v) for v in x] if isinstance(x, list) else x
    # switched off, the decision rows are an empty entry and no model size is written
    assert strip(Jw) == strip(Jn) and 'solver_model' in Jw and 'solver_model' not in Jn
    assert all(not S.get('decision') and 'solver_model' not in S for S in Jn['results'].values())
    assert any(S.get('decision') for S in Jw['results'].values()) == (origin == 'request')
    assert Cw == Cn


def test_new_macro_names_are_unique_over_every_size():
    """The bold markers (request clock) and the macros of (x) (every clock) of every size,
    with the names the script already writes, pass the script's own check."""
    def cells(n):
        return [dict(combo=r, idx=i, record=f'rec/{r}/{i}', variables=1000 + i + j,
                     variables_reason=None, build_s=1.0 + i, peak_rss_gb=2.0 + j,
                     first_incumbent_s=3.0 + i + j)
                for i in range(n) for j, r in enumerate(M.COMBOS)]
    match = [float(t) for t in range(601)]
    curve = M.log_grid(600, extra=M.TRAINING_BUDGETS)
    grid = sorted(set(match) | set(curve))
    n, C = 4, len(M.COMBOS)
    rng = np.random.default_rng(0)
    g = np.array(grid)[:, None, None]
    out = {}
    for request in (False, True):
        data = {}
        for size in M.SIZE_WORD:
            rule = 500 + rng.integers(0, 50, (n, C)).astype(float)
            p = rule - 10
            D = dict(instances=list(range(n)), arms=list(M.P.ARMS), n_descriptors=n, rule=rule,
                     pol_sel=p, all_pol=np.stack([p, p + 1, p - 1]),
                     seeds_with_arrays=list(SEEDS), model_cells=cells(n))
            if request:
                later = np.where(g >= 1.0, rule - 5 - (g > 100), np.nan)
                D.update(A={arm: later.copy() for arm in M.P.ARMS},
                         A_present={arm: np.ones((n, C), dtype=bool) for arm in M.P.ARMS},
                         o_rule=np.full((n, C), 1.0), o_pol=np.full((n, C), 2.0),
                         o_res=np.full((n, C), 1.0), S=np.where(g >= 2.0, p - 3 + 0 * g, np.nan),
                         S_present=np.ones((n, C), dtype=bool), policy_budget_s=120.0,
                         budgets=list(M.TRAINING_BUDGETS if size == M.P.TRAINING_SIZE
                                      else M.REQUEST_BUDGETS))
            else:
                D['A'] = {arm: np.broadcast_to(rule - 5, (len(grid), n, C)).copy()
                          for arm in M.P.ARMS}
            data[size] = D
        R = (M.analyse_request(data, list(M.COMBOS), 30, 0.05, grid, match, curve, 'sel')
             if request else
             M.analyse(data, list(M.COMBOS), list(M.BUDGETS), 30, 0.05, grid, match, curve,
                       'sel'))
        info = dict(setting='exact', selection_text='s302 0.8', cpsat_arm='ref', boot=30,
                    alpha=0.05, origin='request' if request else 'start', campaigns=['all'],
                    sel=SEL, k=3, horizon=600, primary=600)
        names = M.macro_names(M.macro_lines(R, info))
        M.check_macro_names(names, os.path.join(M.ROOT, 'manuscript', 'macros_timed.tex'))
        out[request] = names
    for request, names in out.items():
        bold = [x for x in names if x.endswith('Bold')]
        model = [x for x in names if re.match(r'Tmd(Fifty|Eighty|Ten)Cp(ModelCells|Vars|Build|'
                                              r'Memory|FirstIncumbent)', x)]
        # (x): 11 entries x (cells, 4 fields x 3 statistics, missing counts) per size
        assert len(model) == len(set(model)) == 3 * 11 * (1 + 4 * 3 + 1)
        # bold markers: per size its budgets x 3 pooled entries x 6 columns
        assert len(bold) == (3 * (3 + 3 + 4) * 6 if request else 0)
    assert {'TmdFiftyRuleVsPolSixtyPoolSelBold', 'TmdEightyPolSmpVsPolSixHundredSetupBold',
            'TmdTenMemRuleStartVsPolTenNoSetupSelBold',
            'TmdTenCpRuleStartVsPolThreeHundredSetupSelBold',
            'TmdFiftyMemPolStartVsPolSixtyPoolBold', 'TmdEightyCpPolStartVsPolSixHundredSetupBold',
            'TmdTenMemPolStartVsPolTenNoSetupBold',
            'TmdFiftyCpVarsMedianSetup', 'TmdEightyCpMemoryMaxLSW', 'TmdTenCpFirstIncumbentMinPool',
            'TmdFiftyCpBuildMedianNoSetup', 'TmdEightyCpModelCellsN', 'TmdTenCpVarsMissingSetup'
            } <= set(out[True])
    # every bold marker names a value macro of the same row that the script writes
    for x in (x for x in out[True] if x.endswith('Bold')):
        v = x[:-len('Bold')]
        if 'RuleVsPol' in v and 'RuleStart' not in v:
            v = re.sub(r'RuleVsPol(One|Ten|Sixty|ThreeHundred|SixHundred)', 'RuleVsPol', v)
        assert v in out[True], x
