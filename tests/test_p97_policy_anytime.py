"""Runner of the policy's sampled schedules within a time budget
(scripts/p97_policy_anytime.py): the seed rule, batches, the best-so-far trajectory, the
cut at the budget, records that are never overwritten, validity, the plan's skip rule,
the dry run, the summary over batch sizes, the rule arm and the reader of the
trajectory. Stubs stand in for the checkpoint and the rule rollouts, so no model is
loaded and no solver runs."""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'scripts'))
sys.path.insert(0, os.path.join(HERE, '..'))
import p97_policy_anytime as P  # noqa: E402
import timed_campaign_plan as TCP  # noqa: E402
from baselines.memetic.records import incumbent_at  # noqa: E402


# ---------------------------------------------------------------------------
# stand-ins for the policy
# ---------------------------------------------------------------------------

def stub_makespans(greedy, seed, batch):
    """500 for the greedy schedule; row r of a sampled batch: 470 + (seed + 7 r) % 37."""
    if greedy:
        return [500.0]
    return [float(470 + (seed + 7 * r) % 37) for r in range(batch)]


class FakeClock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t


class ClockRoller:
    """Each rollout advances the fake clock by `step` seconds in `n` decisions and
    checks the deadline before each decision, as PolicyRoller does."""

    def __init__(self, clock, step, n=4):
        self.clock, self.step, self.n = clock, step, n
        self.calls = []

    def rollout_batch(self, greedy, seed, batch, deadline, clock):
        self.calls.append((greedy, seed, batch))
        for _ in range(self.n):
            if clock() >= deadline:
                return None
            self.clock.t += self.step / self.n
        ms = stub_makespans(greedy, seed, batch)
        return ms, np.zeros((len(ms), 3), dtype=int), np.repeat(np.array(ms)[:, None], 3, 1)


class SleepRoller:
    """Real-time stand-in for a worker process: each rollout sleeps spec['stub_dt']."""

    def __init__(self, spec):
        self.dt = spec['stub_dt']

    def rollout_batch(self, greedy, seed, batch, deadline, clock):
        end = time.monotonic() + self.dt
        while time.monotonic() < end:
            if clock() >= deadline:
                return None
            time.sleep(0.005)
        ms = stub_makespans(greedy, seed, batch)
        return ms, np.tile(np.arange(3), (len(ms), 1)), np.repeat(np.array(ms)[:, None], 3, 1)

    def info(self):
        return {'stub': True}

    def thread_record(self):
        return {}


def stub_validator(spec, amch, ct):
    ct = np.asarray(ct, dtype=float)
    return [], (np.asarray(amch), ct - 1, ct)


def stub_spec(budget=1.0, workers=2, dt=0.12, batch=1):
    return dict(model='10x25+family+stub', checkpoint='-', config='-', training_seed=None,
                environment='legacy', mask={'first_mask': 'legacy', 'mask_ties': 'legacy'},
                mask_check='-', size='50x25n30', combination='N', instance=3,
                budget=budget, workers=workers, batch=batch, base_seed=50, t0=time.monotonic(),
                affinity=sorted(os.sched_getaffinity(0)), development=False,
                idle_before={'idle': True, 'busy_processes': []}, stub_dt=dt)


def good_record(**kw):
    rec = dict(development_run=False, validated=True, budget_clock='wall',
               idle_before={'idle': True, 'busy_processes': []}, workers=2, affinity=[8, 9],
               per_worker=[dict(worker=0, core=8, affinity=[8], cpu_share=0.97),
                           dict(worker=1, core=9, affinity=[9], cpu_share=0.95)],
               greedy_makespan=500.0, model='10x25+family+stub', training_seed=None,
               size='50x25n30', combination='N', instance='instance_003')
    rec.update(kw)
    return rec


def running_min_at(rows, t):
    vals = [r[6] for r in rows if r[4] <= t + 1e-9]
    return min(vals) if vals else None


# ---------------------------------------------------------------------------
# seeds
# ---------------------------------------------------------------------------

def test_rollout_seed_is_a_pure_function_of_base_worker_and_index():
    assert P.rollout_seed(50, 3, 7) == 50 * 2 ** 26 + 3 * 2 ** 20 + 7
    assert P.rollout_seed(50, 3, 7) == P.rollout_seed(50, 3, 7)
    seen = set()
    for b in (0, 1, 50):
        for w in range(8):
            for j in range(200):
                s = P.rollout_seed(b, w, j)
                assert P.seed_parts(s) == (b, w, j)
                seen.add(s)
    assert len(seen) == 3 * 8 * 200
    for bad in ((50, 64, 0), (50, 0, 2 ** 20), (-1, 0, 0)):
        try:
            P.rollout_seed(*bad)
        except ValueError:
            continue
        raise AssertionError(f'{bad} accepted')


def test_worker_zero_starts_greedy_and_every_other_rollout_is_seeded():
    clock = FakeClock()
    for w in (0, 1):
        r = ClockRoller(clock, step=0.1)
        clock.t = 0.0
        out = P.worker_loop(r, w, 50, 0.0, 1.0, clock)
        kinds = [x[2] for x in out['rows']]
        assert kinds[0] == ('greedy' if w == 0 else 'sampled')
        assert kinds[1:] == ['sampled'] * (len(kinds) - 1)
        for x in out['rows']:
            assert x[3] == (None if x[2] == 'greedy' else P.rollout_seed(50, w, x[1]))
        assert [x[1] for x in out['rows']] == list(range(len(out['rows'])))


# ---------------------------------------------------------------------------
# the budget and the trajectory
# ---------------------------------------------------------------------------

def test_rollout_ending_after_the_budget_is_not_in_the_record():
    # stopped inside: the deadline is checked before every decision
    clock = FakeClock(0.1)
    out = P.worker_loop(ClockRoller(clock, step=0.4), 0, 50, 0.0, 1.0, clock)
    assert [x[4] for x in out['rows']] == [0.5, 0.9] and out['discarded']
    # completed after the budget: one decision that jumps past it
    clock = FakeClock(0.1)
    out = P.worker_loop(ClockRoller(clock, step=0.35, n=1), 0, 50, 0.0, 1.0, clock)
    assert [x[4] for x in out['rows']] == [0.45, 0.8] and out['discarded']
    assert all(x[4] <= 1.0 for x in out['rows'])
    # real clock, two worker processes
    rec, sched, errors = P.run_cell(stub_spec(), factory=SleepRoller, validator=stub_validator)
    assert not errors
    assert rec['n_schedules'] > 0
    assert all(r[4] <= rec['budget_s'] for r in rec['schedules'])
    assert all(w['stop_s'] >= rec['budget_s'] for w in rec['per_worker'])


def test_trajectory_is_non_increasing_and_the_running_minimum():
    rows = [[1, 0, 'sampled', 1, 2.0, 1.0, 510.0, 1.0, 0],
            [0, 0, 'greedy', None, 1.5, 0.5, 520.0, 1.0, 0],
            [0, 1, 'sampled', 2, 2.5, 1.5, 505.0, 2.0, 0],
            [1, 1, 'sampled', 3, 3.0, 2.0, 505.0, 2.0, 0],
            [1, 2, 'sampled', 4, 4.0, 3.0, 490.0, 3.0, 0],
            [0, 2, 'sampled', 5, 4.5, 3.5, 499.0, 3.0, 0]]
    merged = P.merge([{'rows': rows[1::2]}, {'rows': rows[0::2]}])
    assert [r[4] for r in merged] == sorted(r[4] for r in rows)
    anytime, src = P.trajectory(merged, search_start=1.0)
    ms = [a[3] for a in anytime]
    assert ms == [520.0, 510.0, 505.0, 490.0]
    assert all(b < a for a, b in zip(ms, ms[1:]))
    for a in anytime:
        assert a[3] == running_min_at(merged, a[1]) and a[0] == a[1] - 1.0
    assert src == [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 2, 0]]
    # on a run of the cell with two worker processes
    rec, _, errors = P.run_cell(stub_spec(), factory=SleepRoller, validator=stub_validator)
    assert not errors
    ms = [a[3] for a in rec['anytime']]
    assert all(b < a for a, b in zip(ms, ms[1:]))
    for t in [r[4] for r in rec['schedules']]:
        assert incumbent_at(rec, t, 'wall', 'process') == running_min_at(rec['schedules'], t)
    assert rec['makespan'] == min(r[6] for r in rec['schedules'])
    assert rec['greedy_makespan'] == 500.0


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------

def test_existing_record_is_not_overwritten(tmp_path):
    p = str(tmp_path / 'N' / 'instance_003.json')
    sched = (np.arange(3), np.array([0.0, 1, 2]), np.array([1.0, 2, 3]))
    assert P.write_record(p, {'n': 1}, sched)
    assert not P.write_record(p, {'n': 2}, sched)
    assert json.load(open(p))['n'] == 1
    assert json.load(open(p))['schedule_file'] == 'instance_003.schedule.npz'
    z = np.load(P.schedule_path(p))
    assert set(z.files) == {'amch', 'st', 'ct', 'ms'} and float(z['ms']) == 3.0
    assert sorted(os.listdir(tmp_path / 'N')) == ['instance_003.json',
                                                  'instance_003.schedule.npz']


def test_share_below_the_bar_marks_the_record_invalid():
    assert P.classify(good_record(), stored={}) == (
        True, 'share; no stored greedy result to compare')
    low = good_record()
    low['per_worker'][1]['cpu_share'] = 0.85
    ok, why = P.classify(low, stored={})
    assert not ok and 'cpu_share 0.85' in why and 'worker 1' in why
    assert P.classify(good_record(), stored={'f.npy': 500.0})[0]
    assert not P.classify(good_record(), stored={'f.npy': 501.0})[0]
    assert P.classify(good_record(development_run=True), stored={}) == (False, 'development run')
    assert not P.classify(good_record(greedy_makespan=None), stored={})[0]
    assert not P.classify(good_record(idle_before={'idle': False, 'busy_processes': ['x']}),
                          stored={})[0]
    shared = good_record()
    shared['per_worker'][1].update(core=8, affinity=[8])
    assert not P.classify(shared, stored={})[0]


def plan_args(root, **kw):
    a = dict(sizes=['50x25n30'], combos=['N'], instances='3-4', budget=5.0, workers=2,
             batch=1, base_seed=50, root=str(root), max_invalid=2, development=False)
    a.update(kw)
    return argparse.Namespace(**a)


def test_plan_skips_a_valid_record_and_reruns_an_invalid_one(tmp_path):
    run = dict(model='10x25+family+stub', training_seed=None, environment=P.REG.env())
    a = plan_args(tmp_path)
    tag = P.run_tag(run)
    base = dict(good_record(), budget_s=5.0, workers=2, batch=1, base_seed=50,
                model=run['model'], environment=run['environment'])
    P.write_record(P.record_path(a.root, '50x25n30', 5.0, 1, tag, 'N', 3), base)
    P.write_record(P.record_path(a.root, '50x25n30', 5.0, 1, tag, 'N', 4),
                   dict(base, per_worker=[dict(worker=0, core=8, affinity=[8], cpu_share=0.5),
                                          dict(worker=1, core=9, affinity=[9], cpu_share=0.95)]))
    states = {c['instance']: c['state'] for c in P.plan_cells(a, run)}
    assert states == {3: 'done', 4: 'invalid'}
    other = plan_args(tmp_path, budget=5.0, workers=4)
    assert {c['state'] for c in P.plan_cells(other, run)} == {'foreign'}


def tree(root):
    return sorted(os.path.join(d, f) for d, _, fs in os.walk(root) for f in fs)


def test_dry_run_writes_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('PACT_ENV', 'legacy')
    monkeypatch.delenv('PACT_SEEDS', raising=False)
    notes = os.path.join(P.ROOT, P.NOTES)
    had_notes = os.path.exists(notes)
    before = tree(tmp_path)
    P.main(['--dry', '--budget', '5', '--root', str(tmp_path), '--sizes', '50x25n30',
            '--combos', 'N', 'LSW', '--instances', '3-4', '--workers', '2', '--development'])
    out = capsys.readouterr().out
    assert 'dry run: nothing started, nothing written' in out
    assert 'to run 4' in out and 'slot-hours' in out
    assert tree(tmp_path) == before == []
    assert os.path.exists(notes) == had_notes


def test_record_round_trips_through_the_reader(tmp_path):
    rec, sched, errors = P.run_cell(stub_spec(budget=1.0), factory=SleepRoller,
                                    validator=stub_validator)
    assert not errors
    p = str(tmp_path / 'N' / 'instance_003.json')
    assert P.write_record(p, rec, sched)
    back = json.load(open(p))
    rows = back['schedules']
    grid = [round(x, 3) for x in np.linspace(0.0, back['budget_s'], 41)]
    for t in grid:
        want = running_min_at(rows, t)
        assert incumbent_at(back, t, 'wall', 'process') == want
        ts = t - back['search_start_wall_s']
        if ts >= 0:
            assert incumbent_at(back, ts, 'wall', 'search') == want
    start = TCP.record_rows(back, 'memetic', times=tuple(grid), origin='start')
    assert [r['makespan'] for r in start] == [running_min_at(rows, t) for t in grid]
    assert all(r['start_makespan'] is None for r in start)
    assert back['anytime_columns'][1] == 'wall_s_from_process_start'
    assert not math.isnan(back['cpu_share'])


# ---------------------------------------------------------------------------
# batches
# ---------------------------------------------------------------------------

def test_seed_of_a_batch_is_a_pure_function_of_base_worker_and_batch_index():
    clock = FakeClock()
    runs = []
    for _ in range(2):
        for w in (0, 3):
            r = ClockRoller(clock, step=0.1)
            clock.t = 0.0
            out = P.worker_loop(r, w, 50, 0.0, 1.0, clock, batch=4)
            runs.append((w, r.calls, [x[:7] + x[8:] for x in out['rows']]))   # CPU left out
            for greedy, seed, batch in r.calls:
                assert batch == (1 if greedy else 4)
            j0 = 1 if w == 0 else 0
            assert [c[1] for c in r.calls if not c[0]] == [
                P.rollout_seed(50, w, j) for j in range(j0, j0 + len(r.calls) - j0)]
    assert runs[:2] == runs[2:]                     # the same seeds and rows again


def test_a_batch_contributes_its_rows_with_one_time():
    clock = FakeClock()
    out = P.worker_loop(ClockRoller(clock, step=0.1), 0, 50, 0.0, 1.0, clock, batch=4)
    groups = {}
    for r in out['rows']:
        groups.setdefault(r[1], []).append(r)
    assert [len(g) for g in groups.values()] == [1] + [4] * (len(groups) - 1)
    assert groups[0][0][2] == 'greedy' and groups[0][0][3] is None
    for j, g in groups.items():
        assert [r[8] for r in g] == list(range(len(g)))
        assert len({(r[3], r[4], r[5], r[7]) for r in g}) == 1      # seed, t, t_resident, CPU
        if j:
            assert [r[6] for r in g] == stub_makespans(False, g[0][3], 4)
    # the improving entry holds the batch's first row with its lowest makespan
    for imp in out['improving']:
        g = groups[imp['index']]
        low = min(r[6] for r in g)
        assert imp['makespan'] == low and imp['row'] == [r[6] for r in g].index(low)


def test_incomplete_batch_is_discarded_whole():
    clock = FakeClock(0.1)
    out = P.worker_loop(ClockRoller(clock, step=0.4), 1, 50, 0.0, 1.0, clock, batch=3)
    assert out['discarded'] and [r[4] for r in out['rows']] == [0.5] * 3 + [0.9] * 3
    assert out['n_batches'] == 2
    rec, _, errors = P.run_cell(stub_spec(budget=1.0, batch=3), factory=SleepRoller,
                                validator=stub_validator)
    assert not errors and rec['batch'] == 3
    sizes = {}
    for r in rec['schedules']:
        sizes[(r[0], r[1])] = sizes.get((r[0], r[1]), 0) + 1
    assert all(n == (1 if (w, j) == (0, 0) else 3) for (w, j), n in sizes.items())
    assert all(r[4] <= rec['budget_s'] for r in rec['schedules'])


def test_trajectory_takes_one_improvement_per_batch_from_its_best_row():
    rows = [[0, 0, 'greedy', None, 1.0, 0.5, 500.0, 1.0, 0],
            [1, 0, 'sampled', 9, 2.0, 1.0, 490.0, 2.0, 0],
            [1, 0, 'sampled', 9, 2.0, 1.0, 480.0, 2.0, 1],
            [1, 0, 'sampled', 9, 2.0, 1.0, 480.0, 2.0, 2],
            [0, 1, 'sampled', 8, 2.5, 2.0, 485.0, 2.0, 0]]
    merged = P.merge([{'rows': rows[:1] + rows[4:]}, {'rows': rows[1:4]}])
    anytime, src = P.trajectory(merged, search_start=0.5)
    assert [a[3] for a in anytime] == [500.0, 480.0]
    assert src == [[0, 0, 0], [1, 0, 1]]


def test_folder_name_carries_the_batch_and_a_record_of_another_batch_is_foreign(tmp_path):
    run = dict(model='10x25+family+stub', training_seed=301, environment=P.REG.env())
    tag = P.run_tag(run)
    assert os.path.basename(P.folder(str(tmp_path), '50x25n30', 600, 4, tag)) == \
        '50x25n30_polsample600s_b4_s301'
    a = plan_args(tmp_path, batch=4)
    base = dict(good_record(), budget_s=5.0, workers=2, batch=4, base_seed=50,
                model=run['model'], environment=run['environment'])
    P.write_record(P.record_path(a.root, '50x25n30', 5.0, 4, tag, 'N', 3), base)
    P.write_record(P.record_path(a.root, '50x25n30', 5.0, 4, tag, 'N', 4), dict(base, batch=1))
    states = {c['instance']: c['state'] for c in P.plan_cells(a, run)}
    assert states == {3: 'done', 4: 'foreign'}
    # the plan of batch 1 looks in its own folder
    assert {c['state'] for c in P.plan_cells(plan_args(tmp_path, batch=1), run)} == {'run'}


# ---------------------------------------------------------------------------
# summary over batch sizes
# ---------------------------------------------------------------------------

def summary_fixture(root, model, ms_by_batch, cells):
    run = dict(model=model, training_seed=None)
    tag = P.run_tag(run)
    for (combo, idx), batches in cells.items():
        for b in batches:
            rec = dict(good_record(), budget_s=40.0, batch=b, workers=2, base_seed=50,
                       model=model, environment=P.REG.env(), combination=combo,
                       instance=P.instance_name(idx), n_schedules=10 * b,
                       anytime=[[0.5, 1.0, 1.0, 500.0], [19.5, 20.0, 9.0, ms_by_batch[b]]],
                       search_start_wall_s=0.5)
            assert P.write_record(P.record_path(str(root), '50x25n30', 40.0, b, tag, combo, idx),
                                  rec)


def run_summary(root, monkeypatch, capsys, model):
    monkeypatch.setenv('PACT_ENV', 'legacy')
    monkeypatch.delenv('PACT_SEEDS', raising=False)
    rc = P.main(['--summary', '--budget', '40', '--root', str(root), '--sizes', '50x25n30',
                 '--combos', 'N', '--instances', '3-4', '--batches', '2', '1', '--workers', '2',
                 '--model', model])
    return rc, capsys.readouterr().out


def test_summary_pairs_cells_and_writes_nothing(tmp_path, monkeypatch, capsys):
    model = '10x25+family+stub'
    monkeypatch.setenv('PACT_ENV', 'legacy')
    summary_fixture(tmp_path / 'a', model, {1: 490.0, 2: 480.0},
                    {('N', 3): (1, 2), ('N', 4): (1,)})
    notes = os.path.join(P.ROOT, P.NOTES)
    had_notes = os.path.exists(notes)
    before = tree(tmp_path)
    rc, out = run_summary(tmp_path / 'a', monkeypatch, capsys, model)
    assert rc == 0 and tree(tmp_path) == before and os.path.exists(notes) == had_notes
    assert '1 of 2 cells compared' in out
    lines = {ln.split()[0]: ln.split() for ln in out.splitlines()
             if ln.strip()[:1].isdigit() and '/' in ln}
    # batch, valid, 10 s, 30 s, 40 s, schedules: the gap of the one paired cell only
    assert lines['1'][1:] == ['2/2', '0.000', '-2.000', '-2.000', '10.0']
    assert lines['2'][1:] == ['1/2', '0.000', '-4.000', '-4.000', '20.0']
    assert 'lowest pooled mean at the budget: batch 2' in out
    # equal means: the smaller batch
    summary_fixture(tmp_path / 'b', model, {1: 490.0, 2: 490.0}, {('N', 3): (1, 2)})
    rc, out = run_summary(tmp_path / 'b', monkeypatch, capsys, model)
    assert rc == 0 and 'lowest pooled mean at the budget: batch 1' in out


# ---------------------------------------------------------------------------
# rule arm
# ---------------------------------------------------------------------------

class StubRules:
    """Stand-in for the rule rollouts: a short sleep, then the makespan of spec['stub_ms']."""

    def __init__(self, spec):
        self.ms = spec['stub_ms']

    def rollout(self, rule):
        time.sleep(0.05)
        ms = float(self.ms[rule])
        return ms, np.arange(3), np.full(3, ms)

    def eligible(self, amch, ct):
        return True, 'pass'

    def info(self):
        return {'stub': True}


STUB_MS = {'FIFO': 139.0, 'MOR': 152.0, 'SPT': 132.0, 'MWKR': 132.0}


def test_rule_cell_rolls_out_each_rule_once_and_takes_the_earlier_of_equal_makespans():
    spec = dict(size='10x25', combination='N', instance=0, workers=2, rules=P.RULE_ORDER,
                t0=time.monotonic(), budget=0.0, environment='exact',
                affinity=sorted(os.sched_getaffinity(0)), development=False,
                idle_before={'idle': True, 'busy_processes': []}, stub_ms=STUB_MS)
    rec, sched, errors, raw = P.run_rule_cell(spec, factory=StubRules, validator=stub_validator)
    assert not errors and rec['validated']
    assert [r['rule'] for r in rec['rules']] == list(P.RULE_ORDER) and sorted(raw) == sorted(STUB_MS)
    assert [r['worker'] for r in rec['rules']] == [0, 1, 0, 1]
    assert {r['rule']: r['makespan'] for r in rec['rules']} == STUB_MS
    assert rec['best_rule'] == 'SPT' and rec['best_makespan'] == 132.0
    assert rec['t_all'] == max(r['t'] for r in rec['rules']) and rec['t_best'] <= rec['t_all']
    ms = [a[3] for a in rec['anytime']]
    assert ms[-1] == 132.0 and all(b < a for a, b in zip(ms, ms[1:]))
    assert float(np.max(sched[2])) == 132.0


def rule_record(**kw):
    per = [dict(worker=w, core=8 + w, affinity=[8 + w], cpu_share=0.97) for w in range(4)]
    rec = dict(good_record(workers=4, affinity=[8, 9, 10, 11], per_worker=per),
               arm=P.RULE_ARM, size='10x25', instance='instance_000',
               rules=[dict(rule=k, makespan=v) for k, v in STUB_MS.items()],
               best_rule='SPT', best_makespan=132.0)
    rec.update(kw)
    return rec


def test_rule_record_is_invalid_when_a_rule_makespan_differs_from_the_stored_one():
    cache = dict(file='cache/N/instance_000.npz', rule_ms=dict(STUB_MS), rule='SPT', ms=132.0)
    ok, why = P.classify_rules(rule_record(), cache)
    assert ok and 'equal to cache/N/instance_000.npz' in why
    other = rule_record(rules=[dict(rule=k, makespan=v + (1.0 if k == 'MOR' else 0.0))
                               for k, v in STUB_MS.items()])
    ok, why = P.classify_rules(other, cache)
    assert not ok and why.startswith('MOR makespan 153 differs from the stored 152')
    assert not P.classify_rules(rule_record(best_rule='MWKR'), cache)[0]
    assert not P.classify_rules(rule_record(rules=rule_record()['rules'][:3]), cache)[0]
    assert not P.classify_rules(rule_record(development_run=True), cache)[0]
    low = rule_record()
    low['per_worker'][2]['cpu_share'] = 0.5
    assert not P.classify_rules(low, cache)[0]


def test_rule_dry_run_writes_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('PACT_ENV', 'exact')
    monkeypatch.delenv('PACT_SEEDS', raising=False)
    notes = os.path.join(P.ROOT, P.NOTES)
    had_notes = os.path.exists(notes)
    before = tree(tmp_path)
    rc = P.main(['--arm', 'rules', '--dry', '--root', str(tmp_path), '--sizes', '10x25',
                 '--combos', 'N', 'LSW', '--instances', '0-1', '--workers', '2', '--development'])
    out = capsys.readouterr().out
    assert rc == 0 and 'dry run: nothing started, nothing written' in out
    assert 'to run 4' in out and '_rulestart' in out
    assert tree(tmp_path) == before == [] and os.path.exists(notes) == had_notes
