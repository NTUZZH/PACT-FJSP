"""Planner of the timed campaign (scripts/timed_campaign_plan.py): cells, batches,
cores, memory pairing, skip and requeue, refusals, slot-hours, the reader and the
pilot's selection. Synthetic records only; no solver runs."""
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'scripts'))
import timed_campaign_plan as P  # noqa: E402

SLOT_A, SLOT_B = list(range(8, 16)), list(range(16, 24))


def camp(tmp_path, *extra, cmd='plan'):
    args = [cmd, '--result-root', str(tmp_path), '--policy-seed', '303',
            '--policy-cache', str(tmp_path / 'pol_s{seed}_{size}'),
            '--bars', str(tmp_path / 'bars.json'), '--logdir', str(tmp_path / 'logs'),
            '--own-state', str(tmp_path / 'own.txt'), '--peer-state', 'none'] + list(extra)
    if cmd == 'next':
        args += ['--slot', 'A', '--state', str(tmp_path / 'state')]
    a = P.build_parser().parse_args(args)
    return P.Campaign(a)


def write_record(c, cell, **kw):
    """A record of the right protocol at the planned path."""
    rec = dict(c.expected_fields(cell))
    if cell.seg.runner == 'memetic':
        rec.update(validated=True, threads=1, workers=1, cpu_share=0.999,
                   affinity=[SLOT_A[0]], makespan=500, objective=500.0,
                   anytime=[[0.0, 1.0, 1.0, 510], [5.0, 6.0, 6.0, 500]],
                   search_start_wall_s=1.0, search_start_cpu_s=1.0,
                   seeds=[{'source': 'warm_cache', 'makespan': 520}])
    else:
        rec.update(hint='complete', validated=True, status='FEASIBLE', cpu_share=0.95,
                   affinity=SLOT_A, objective=500.0, warm_makespan=520.0, build_s=30.0,
                   anytime=[[40.0, 515.0, 300.0], [900.0, 500.0, 300.0]])
    rec.update(kw)
    p = c.record(cell)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(rec, open(p, 'w'))
    return p


# ---------------------------------------------------------------------------
# cells, folders, commands

def test_cells_are_instance_major_and_folders_follow_the_runners(tmp_path):
    c = camp(tmp_path)
    seg = c.segments[0]
    assert seg.label == 'memetic_rule@1800' and seg.instances == list(range(3, 30))
    cells = c.cells(seg)
    assert len(cells) == 27 * 2 * 8
    assert [(x.idx, x.size, x.combo) for x in cells[:9]] == \
        [(3, '50x25n30', r) for r in P.COMBOS] + [(3, '80x25n30', 'N')]
    assert c.record(cells[0]).endswith(
        'or_solution/FAMILY/50x25n30_memetic1800s/N/instance_003_seed1.json')
    cp = c.cells(c.segments[1])[0]
    assert c.record(cp).endswith(
        'test_results/FAMILY/50x25n30_cparms/evidence_rule1800s/ref_fullhint/N/instance_003.json')
    c2 = camp(tmp_path, '--arms', 'memetic_policy cpsat_policy@600:3-9')
    m, s = c2.cells(c2.segments[0])[0], c2.cells(c2.segments[1])[0]
    assert '50x25n30_memetic1800s_pactwarm_s303' in c2.record(m)
    assert 'evidence_pactwarm600s_s303/ref_fullhint' in c2.record(s)
    assert c2.segments[1].instances == list(range(3, 10))


def test_commands_carry_the_arm_settings(tmp_path):
    c = camp(tmp_path, '--arms', 'memetic_policy cpsat_policy')
    m = c.cells(c.segments[0])[2]                       # S at 50 modules
    cmd = c.command(m, [10])
    assert cmd[:3] == ['taskset', '-c', '10'] and 'NUMBA_NUM_THREADS=1' in cmd
    assert cmd[cmd.index('--warm_cache') + 1] == 'or_solution/FAMILY/warmstart_cache_xmask_50x25n30'
    assert cmd[cmd.index('--seed_from') + 1] == str(tmp_path / 'pol_s303_50x25n30')
    assert '--seed_schedules' in cmd and cmd[cmd.index('--budget') + 1] == '1800'
    s = c.cells(c.segments[1])[0]
    cmd = c.command(s, SLOT_B)
    assert cmd[:3] == ['taskset', '-c', '16-23']
    assert cmd[cmd.index('--slots') + 1] == '16-23' and cmd[cmd.index('--workers') + 1] == '8'
    assert cmd[cmd.index('--hint') + 1] == str(tmp_path / 'pol_s303_50x25n30')
    assert cmd[cmd.index('--out') + 1] == 'evidence_pactwarm1800s_s303'
    assert cmd[cmd.index('--root') + 1] == str(tmp_path / 'test_results' / 'FAMILY')


def test_a_cell_is_never_planned_twice_at_two_budgets(tmp_path):
    with pytest.raises(SystemExit):
        camp(tmp_path, '--arms', 'cpsat_rule@1800:3-5 cpsat_rule@600:5-29')
    c = camp(tmp_path, '--arms', 'cpsat_rule@1800:3-5 cpsat_rule@600:6-29')
    assert [s.label for s in c.segments] == ['cpsat_rule@1800', 'cpsat_rule@600']


# ---------------------------------------------------------------------------
# batches, cores and memory

def test_memetic_batches_put_one_process_on_each_core(tmp_path):
    c = camp(tmp_path)
    batches = P.remaining_batches(c)[0]
    assert len(batches) == 54 and all(len(b.cells) == 8 for b in batches)
    b = batches[0]
    cores = P.batch_cores(c, b, SLOT_A)
    assert [x[0] for x in cores] == SLOT_A          # combination k on the k-th core
    assert b.need == pytest.approx(8 * 0.75)
    # requeued cells keep the cores of their combinations, one cell per combination
    cells = [x for x in c.cells(c.segments[0]) if x.combo in ('S', 'LSW')][:6]
    part = P.Batch(c, P.memetic_batch(c, cells, 8))
    assert [(x.combo, x.idx) for x in part.cells] == [('S', 3), ('LSW', 3)]
    assert [x[0] for x in P.batch_cores(c, part, SLOT_B)] == [18, 23]
    # a slot of four cores takes four cells, one per core
    four = P.Batch(c, P.memetic_batch(c, c.cells(c.segments[0]), 4))
    assert [x.combo for x in four.cells] == ['N', 'L', 'S', 'W']
    assert [x[0] for x in P.batch_cores(c, four, [16, 17, 18, 19])] == [16, 17, 18, 19]


def test_no_core_holds_two_processes_and_memory_stays_in_budget(tmp_path):
    c = camp(tmp_path, '--arms', 'memetic_rule cpsat_rule memetic_policy',
             '--instances', '3-6')
    per_seg = P.remaining_cells(c)
    started, idle, end = P.simulate(c, per_seg)
    cores_of = {'A': SLOT_A, 'B': SLOT_B}
    for n, b, s, e in started:
        cs = [x for cc in P.batch_cores(c, b, cores_of[n]) for x in cc]
        if b.seg.runner == 'memetic':
            assert len(cs) == len(set(cs))           # one process per core
        assert set(cs) <= set(cores_of[n])
    for i, (n1, b1, s1, e1) in enumerate(started):
        for n2, b2, s2, e2 in started[i + 1:]:
            if s1 < e2 - 1e-9 and s2 < e1 - 1e-9:   # running at the same time
                assert n1 != n2                     # never two batches on one slot
                assert b1.need + b2.need <= c.a.mem_budget + 1e-9
    # every 80-module full model with setups runs beside an idle slot at 40 GB
    big = [x for x in started if x[1].cls == 'cpsat-big80']
    assert len(big) == 4 * 4
    for n1, b1, s1, e1 in big:
        assert not any(s1 < e2 - 1e-9 and s2 < e1 - 1e-9 for n2, _, s2, e2 in started
                       if n2 != n1)
    # the arms start in order: memetic_rule ends before any later arm starts,
    # and memetic_policy fills a wait only when no cpsat_rule cell fits
    last_mr = max(s for _, b, s, _ in started if b.seg.arm == 'memetic_rule')
    assert min(s for _, b, s, _ in started if b.seg.arm != 'memetic_rule') >= last_mr - 1e-9
    for n, b, s, e in started:
        if b.seg.arm != 'memetic_policy':
            continue
        beside = [ob for m, ob, s2, e2 in started if m != n and s2 <= s + 1e-9 < e2]
        need, large = sum(ob.need for ob in beside), any(ob.large for ob in beside)
        pending = [lb for _, lb, s2, _ in started if lb.seg.arm == 'cpsat_rule' and s2 > s + 1e-9]
        for lb in pending:
            assert lb.need + need > c.a.mem_budget + 1e-9 or (lb.large and large)


def test_memory_rule_pairs_by_need(tmp_path):
    c = camp(tmp_path, '--arms', 'cpsat_rule memetic_rule', '--instances', '3')
    cells = P.remaining_cells(c)[0]
    assert c.need_class(cells[0]) == 'cpsat-small'         # 50 modules, N
    big80 = [x for x in cells if c.need_class(x) == 'cpsat-big80']
    big50 = [x for x in cells if c.need_class(x) == 'cpsat-big50']
    assert {x.combo for x in big80} == P.SETUP and c.need(big80[0]) == 40
    assert c.need(big50[0]) == 24
    # beside a 50-module full model only small cells fit; beside nothing, anything
    pick = P.choose(c, [big50 + cells, []], 8, 24.0)
    assert pick.need <= 16
    assert P.choose(c, [big80, []], 8, 4.0) == 'wait'
    assert P.choose(c, [big80, []], 8, 0.0).ids == {big80[0].id}
    # a memetic batch fits beside a 50-module full model, not beside an 80-module one
    mem = P.remaining_cells(c)[1]
    assert P.choose(c, [[], mem], 8, 24.0).need == pytest.approx(6.0)
    assert P.choose(c, [[], mem], 8, 40.0) == 'wait'
    # one solve of 30 GB or more at a time, whatever the budget
    c.a.mem_budget = 1e9
    assert P.choose(c, [big80, []], 8, 40.0, other_large=True) == 'wait'
    assert P.choose(c, [big80 + big50, []], 8, 40.0, other_large=True).need == 24
    c.a.mem_budget = 40.0
    # a small arm never binds the rule
    cl = camp(tmp_path, '--arms', 'cpsat_rule', '--instances', '3', '--cpsat-arm', 'lns')
    assert {b.need for b in P.remaining_batches(cl)[0]} == {4.0}


def test_a_narrow_slot_runs_memetic_arms_only(tmp_path):
    c = camp(tmp_path, '--arms', 'memetic_rule cpsat_rule memetic_policy', '--instances', '3-4',
             '--slots', '8-15', '16-19')
    started, idle, end = P.simulate(c, P.remaining_cells(c))
    on_b = [b for n, b, _, _ in started if n == 'B']
    assert on_b and all(b.seg.runner == 'memetic' and len(b.cells) <= 4 for b in on_b)
    assert any(b.seg.runner == 'cpsat' for n, b, _, _ in started if n == 'A')
    # B moves on to memetic_policy while A still runs cpsat_rule cells
    cp_end = max(e for _, b, _, e in started if b.seg.arm == 'cpsat_rule')
    assert min(s for n, b, s, _ in started if b.seg.arm == 'memetic_policy') < cp_end
    for n, b, s, e in started:
        if n == 'B':
            assert {x for cc in P.batch_cores(c, b, [16, 17, 18, 19]) for x in cc} <= {16, 17, 18, 19}
    # without a slot of eight cores the CP-SAT arm cannot run
    c4 = camp(tmp_path, '--arms', 'cpsat_rule', '--instances', '3', '--slots', '16-19')
    assert any('no slot with 8 cores' in r for r in P.check_inputs(c4))
    with pytest.raises(SystemExit):
        P.simulate(c4, P.remaining_cells(c4))


def test_cpsat_commands_carry_the_memory_state_files(tmp_path):
    c = camp(tmp_path, '--arms', 'cpsat_rule', '--instances', '3',
             '--own-state', str(tmp_path / 'own.txt'), '--peer-state', '/x/peer.txt',
             '--control-core', '3')
    cmd = c.command(c.cells(c.segments[0])[0], SLOT_A)
    assert f'PACT_MEMORY_STATE={tmp_path / "own.txt"}' in cmd
    assert 'PACT_PEER_MEMORY_STATE=/x/peer.txt' in cmd
    assert cmd.index('PACT_MEMORY_STATE=' + str(tmp_path / 'own.txt')) < cmd.index(
        'scripts/p85_cpsat_arms.py')
    assert P.check_inputs(c) == []
    c.a.peer_state = 'none'
    assert not any(x.startswith('PACT_PEER') for x in c.command(c.cells(c.segments[0])[0], SLOT_A))
    c.a.peer_state = ''
    assert any('PEER_STATE' in r for r in P.check_inputs(c))
    c.a.peer_state, c.a.control_core = 'none', 9
    assert any('control core 9' in r for r in P.check_inputs(c))
    # small arms need no peer state
    cl = camp(tmp_path, '--arms', 'cpsat_rule', '--instances', '3', '--cpsat-arm', 'lns')
    assert P.check_inputs(cl) == []


def test_needs_equal_the_runner(tmp_path):
    pytest.importorskip('ortools')
    import p85_cpsat_arms as R
    for arm in P.PILOT_ARMS:
        for size in P.SIZES:
            for r in P.COMBOS:
                assert P.cpsat_need_gb(arm, size, r) == R.needs_gb(arm, size, r)[0]


def test_slot_hours_and_wall_clock(tmp_path):
    c = camp(tmp_path)
    started, idle, end = P.simulate(c, P.remaining_cells(c))
    mem = [x for x in started if x[1].seg.runner == 'memetic']
    cp = [x for x in started if x[1].seg.runner == 'cpsat']
    assert len(mem) == 54 and sum(len(b.cells) for _, b, _, _ in cp) == 432
    # 432 single-thread runs of 1800 s on 16 cores: 27 slot-hours of budget
    assert sum(1800 * len(b.cells) / 8 for _, b, _, _ in mem) / 3600 == pytest.approx(27.0)
    assert sum(b.seg.budget for _, b, _, _ in cp) / 3600 == pytest.approx(216.0)
    # memetic: 27 batches of 1805 s per slot
    assert max(e for _, _, _, e in mem) == pytest.approx(27 * 1805.0)
    # without the memory budget the CP-SAT cells share the slots evenly
    free = P.simulate(c, P.remaining_cells(c), budget=1e9)[2]
    work = sum(c.duration(x) for x in c.cells(c.segments[1]))
    assert free == pytest.approx(27 * 1805.0 + work / 2, abs=4000)
    # with it, the 108 big cells at 80 modules run beside an idle slot
    big80 = 108 * (1800 + P.OVERHEAD_S['cpsat-big80'])
    assert end - 27 * 1805.0 == pytest.approx(big80 + (work - big80) / 2, rel=0.01)


# ---------------------------------------------------------------------------
# state: skip, requeue, refusals

def test_finished_cells_are_skipped(tmp_path):
    c = camp(tmp_path, '--arms', 'cpsat_rule', '--instances', '3')
    cells = c.cells(c.segments[0])
    write_record(c, cells[0])
    write_record(c, cells[1], cpu_share=0.80)          # below 0.9, no frozen bar: pending
    st = P.cell_states(c, c.segments[0])
    assert st[cells[0].id][1] == 'valid' and st[cells[1].id][1] == 'pending'
    ids = {x.id for b in P.remaining_batches(c)[0] for x in b.cells}
    assert cells[0].id not in ids and cells[1].id not in ids and len(ids) == 14


def test_invalid_records_are_moved_and_rerun_then_held(tmp_path, capsys):
    c = camp(tmp_path, '--arms', 'memetic_rule', '--instances', '3', cmd='next')
    os.makedirs(c.a.state)
    cell = c.cells(c.segments[0])[0]
    p = write_record(c, cell, cpu_share=0.5)
    assert P.cell_states(c, c.segments[0])[cell.id][1] == 'invalid'
    assert P.cmd_next(c, c.a) == 0
    out = capsys.readouterr().out
    assert not os.path.exists(p) and P.n_invalid(p) == 1
    assert cell.id in out                               # scheduled again
    write_record(c, cell, affinity=[3])                 # a core outside the slots
    P.move_invalid(p, 'test')
    assert P.n_invalid(p) == 2
    assert P.cell_states(c, c.segments[0])[cell.id][1] == 'held'
    ids = {x.id for b in P.remaining_batches(c)[0] for x in b.cells}
    assert cell.id not in ids


def test_record_of_another_protocol_is_refused(tmp_path):
    c = camp(tmp_path, '--arms', 'cpsat_rule', '--instances', '3')
    cell = c.cells(c.segments[0])[0]
    write_record(c, cell, workers=2)
    assert P.cell_states(c, c.segments[0])[cell.id][1] == 'foreign'
    c.a.strict, c.a.all_commands = True, False
    assert P.cmd_plan(c, c.a) == 1


def test_policy_arms_are_refused_while_a_schedule_is_missing(tmp_path, capsys):
    c = camp(tmp_path, '--arms', 'memetic_rule cpsat_policy', '--instances', '3')
    ref = P.check_inputs(c)
    assert len(ref) == 1 and ref[0].startswith('cpsat_policy@1800: REFUSED, 16 policy schedules')
    c.a.strict, c.a.all_commands = True, False
    assert P.cmd_plan(c, c.a) == 1
    assert 'A REAL START WOULD BE REFUSED' in capsys.readouterr().out
    for size in P.SIZES:                                # fill the policy cache
        for r in P.COMBOS:
            d = tmp_path / f'pol_s303_{size}' / r
            d.mkdir(parents=True, exist_ok=True)
            (d / 'instance_003.npz').write_bytes(b'')
    assert P.check_inputs(c) == []
    c0 = camp(tmp_path, '--arms', 'cpsat_policy', '--instances', '3')
    c0.a.policy_seed = None
    assert 'no --policy-seed' in P.check_inputs(c0)[0]


def test_classify_counts_and_stops_on_a_wholly_invalid_batch(tmp_path):
    c = camp(tmp_path, '--arms', 'memetic_rule', '--instances', '3', cmd='next')
    os.makedirs(c.a.state)
    notes = str(tmp_path / 'notes.txt')
    b = P.remaining_batches(c)[0][0]
    path = tmp_path / 'batch.tsv'

    def batch_file():
        lines = [f'# slot=A cores=8-15 arm={b.seg.label} runner=memetic need_gb=6 class=memetic']
        for x, cc in zip(b.cells, P.batch_cores(c, b, SLOT_A)):
            lines.append('\t'.join([str(cc[0]), x.id, c.record(x), 'log', 'cmd']))
        path.write_text('\n'.join(lines) + '\n')

    batch_file()
    a = c.a
    a.batch, a.notes = str(path), notes
    assert P.cmd_classify(c, a) == 4                    # nothing written
    assert len(open(os.path.join(a.state, 'skip.txt')).read().split()) == 8
    for x in b.cells:
        write_record(c, x, cpu_share=0.6)
    assert P.cmd_classify(c, a) == 3                    # all invalid: stop
    assert all(P.n_invalid(c.record(x)) == 1 for x in b.cells)
    for i, x in enumerate(b.cells):
        write_record(c, x, cpu_share=0.6 if i == 0 else 0.99)
    assert P.cmd_classify(c, a) == 0
    assert any('valid 7, pending 0, invalid 1, no record 0' in x for x in open(notes))


# ---------------------------------------------------------------------------
# reader

def test_reader_rows(tmp_path):
    c = camp(tmp_path, '--arms', 'memetic_rule cpsat_rule', '--instances', '3',
             '--sizes', '50x25n30', '--combos', 'S')
    m, s = c.cells(c.segments[0])[0], c.cells(c.segments[1])[0]
    write_record(c, m, anytime=[[0.0, 1.0, 1.0, 515], [0.5, 1.5, 1.5, 508],
                                [59.0, 60.0, 59.9, 501], [1500.0, 1501.0, 1500.5, 497]],
                 makespan=497, search_start_wall_s=1.0)
    write_record(c, s, cpu_share=0.5, anytime=[[45.0, 521.0, 300.0], [70.0, 512.0, 300.0],
                                               [1799.0, 509.0, 301.0]],
                 objective=509.0, warm_makespan=520.0, build_s=30.0, arm='ref_fullhint')
    rows = P.trajectory_rows(c)
    mr = {r['t']: r for r in rows if r['arm'] == 'memetic_rule'}
    assert [mr[t]['makespan'] for t in P.TIMES] == [508, 508, 501, 501, 501, 497]
    assert mr[1800]['makespan'] == mr[1800]['final_makespan']
    assert mr[1]['valid'] is True and mr[1]['pre_search_s'] == 1.0
    cr = {r['t']: r for r in rows if r['arm'] == 'cpsat_rule'}
    # the hint counts from time 0; the first incumbent above it does not replace it
    assert [cr[t]['makespan'] for t in P.TIMES] == [520.0, 520.0, 520.0, 512.0, 512.0, 509.0]
    assert cr[1]['search_makespan'] is None and cr[60]['search_makespan'] == 521.0
    assert cr[1800]['makespan'] == cr[1800]['final_makespan']
    assert cr[1]['valid'] is None                       # 0.5 without a frozen bar: pending
    # from the start of the arm the model build comes first
    st = {r['t']: r['makespan'] for r in P.record_rows(json.load(open(c.record(s))),
                                                         'cpsat', (60, 100), 'start')}
    assert st == {60: 520.0, 100: 512.0}
    # a 600-s record has no 1800-s row
    r600 = json.load(open(c.record(s)))
    r600['budget_s'] = 600.0
    assert [r['t'] for r in P.record_rows(r600, 'cpsat')] == [1, 10, 60, 300, 600]


# ---------------------------------------------------------------------------
# the pilot's selection

def _pilot(tmp_path, value, share=0.95, skip=()):
    root = tmp_path / 'test_results' / 'FAMILY'
    for arm in P.PILOT_ARMS:
        for size in P.SIZES:
            for r in P.PILOT_REGIMES:
                for i in P.PILOT_INSTANCES:
                    if (arm, size, r, i) in skip:
                        continue
                    d = root / f'{size}_cparms' / 'pilot' / arm / r
                    d.mkdir(parents=True, exist_ok=True)
                    v = value(arm, size, r, i)
                    json.dump(dict(hint='complete', validated=True, status='FEASIBLE',
                                   cpu_share=share, affinity=SLOT_A, objective=v,
                                   warm_makespan=v + 10, budget_s=600.0, workers=8,
                                   arm=arm, build_s=1.0, hint_source=f'cache_{size}',
                                   anytime=[[5.0, v + 2, 1.0], [400.0, v, 1.0]],
                                   development_run=False),
                              open(d / f'instance_{i:03d}.json', 'w'))
    return str(root)


def test_selection_rule_and_tie_to_the_simpler_arm(tmp_path):
    # compact is best by 0.2% on every cell; radius by 0.25%
    off = {'compact': 0.0, 'radius': -0.5}
    root = _pilot(tmp_path, lambda a, s, r, i: 1000.0 - 2.0 * (a == 'compact')
                  - 2.5 * (a == 'radius'))
    res = P.select_pilot(root, P.SIZES, [SLOT_A, SLOT_B], str(tmp_path / 'none.json'))
    assert res['refused'] == []
    assert res['scores']['radius']['all'] == pytest.approx(0.0)
    assert res['scores']['compact']['all'] == pytest.approx(0.05, abs=1e-3)
    assert res['chosen'] == 'compact'                   # within 0.10 points and simpler
    assert res['within_tie'] == ['compact', 'radius']
    assert res['scores']['ref']['all'] == pytest.approx(0.25, abs=1e-3)
    assert res['at'][300]['ref']['all'] == pytest.approx(0.25, abs=1e-3)
    del off


def test_selection_lowest_score_wins_outside_the_tie(tmp_path):
    root = _pilot(tmp_path, lambda a, s, r, i: 1000.0 - 5.0 * (a == 'lns'))
    res = P.select_pilot(root, P.SIZES, [SLOT_A, SLOT_B], str(tmp_path / 'none.json'))
    assert res['chosen'] == 'lns' and res['within_tie'] == ['lns']


def test_selection_refuses_missing_invalid_and_pending_records(tmp_path):
    root = _pilot(tmp_path, lambda *k: 1000.0, skip={('lns', '80x25n30', 'N', 2)})
    res = P.select_pilot(root, P.SIZES, [SLOT_A, SLOT_B], str(tmp_path / 'none.json'))
    assert res['refused'] and res['refused'][0].startswith('missing')
    root = _pilot(tmp_path, lambda *k: 1000.0, share=0.8)
    res = P.select_pilot(root, P.SIZES, [SLOT_A, SLOT_B], str(tmp_path / 'none.json'))
    assert res['refused'] and all(x.startswith('pending') for x in res['refused'])
    res = P.select_pilot(root, P.SIZES, [list(range(0, 8))], str(tmp_path / 'none.json'))
    assert res['refused'] and all(x.startswith('invalid') for x in res['refused'])


# ---------------------------------------------------------------------------
# the training size 10x25

TRAIN_CACHE = 'or_solution/FAMILY/warmstart_cache_xmask_10x25'


def test_training_size_takes_instances_0_to_29_and_its_own_folders(tmp_path):
    c = camp(tmp_path, '--arms', 'memetic_rule@600 cpsat_rule@600', '--sizes', '10x25')
    m, s = c.segments
    assert m.instances == list(range(30)) and m.by_size == {'10x25': list(range(30))}
    assert m.instance_text() == '0-29'
    cells = c.cells(m)
    assert len(cells) == 30 * 8 and [x.idx for x in cells[:9]] == [0] * 8 + [1]
    assert c.record(cells[0]).endswith('or_solution/FAMILY/10x25_memetic600s/N/instance_000_seed1.json')
    cmd = c.command(cells[2], [18])                      # S of instance 000
    assert cmd[:3] == ['taskset', '-c', '18']
    assert cmd[cmd.index('--size') + 1] == '10x25'
    assert cmd[cmd.index('--instances') + 1] == '0'
    assert cmd[cmd.index('--warm_cache') + 1] == TRAIN_CACHE
    assert cmd[cmd.index('--out') + 1] == str(tmp_path / 'or_solution' / 'FAMILY' / '10x25_memetic600s')
    cp = c.cells(s)[0]
    assert c.record(cp).endswith(
        'test_results/FAMILY/10x25_cparms/evidence_rule600s/ref_fullhint/N/instance_000.json')
    cmd = c.command(cp, SLOT_B)
    assert cmd[cmd.index('--size') + 1] == '10x25' and cmd[cmd.index('--hint') + 1] == TRAIN_CACHE
    assert c.expected_fields(cells[0])['warm_cache'] == TRAIN_CACHE


def test_production_sizes_keep_their_exclusion_beside_the_training_size(tmp_path):
    c = camp(tmp_path, '--sizes', '50x25n30', '80x25n30', '10x25')
    seg = c.segments[0]
    assert seg.by_size == {'50x25n30': list(range(3, 30)), '80x25n30': list(range(3, 30)),
                           '10x25': list(range(30))}
    assert seg.instance_text() == '3-29 at 50x25n30; 3-29 at 80x25n30; 0-29 at 10x25'
    cells = c.cells(seg)
    assert len(cells) == 27 * 2 * 8 + 30 * 8
    for size in P.SIZES:
        assert {x.idx for x in cells if x.size == size} == set(range(3, 30))
    assert {x.idx for x in cells if x.size == '10x25'} == set(range(30))
    # instance-major: 000-002 at the training size first, then every size of 003
    assert [(x.size, x.idx) for x in cells[:24:8]] == [('10x25', 0), ('10x25', 1), ('10x25', 2)]
    assert [(x.size, x.idx) for x in cells[24:48:8]] == [('50x25n30', 3), ('80x25n30', 3),
                                                         ('10x25', 3)]
    # given instances apply to every size, as before
    g = camp(tmp_path, '--sizes', '50x25n30', '10x25', '--instances', '5-7')
    assert g.segments[0].by_size == {'50x25n30': [5, 6, 7], '10x25': [5, 6, 7]}
    assert g.segments[0].instance_text() == '5-7'
    # a cell is never planned twice: the tokens may split the instances by size
    camp(tmp_path, '--sizes', '50x25n30', '10x25', '--arms', 'memetic_rule@600:0-2 memetic_rule@1800:3-29')
    with pytest.raises(SystemExit):                     # instance 000 of 10x25 twice
        camp(tmp_path, '--sizes', '50x25n30', '10x25', '--arms', 'memetic_rule@600 memetic_rule@1800:0-2')
    # at the production size alone the same tokens do not overlap (000-002 are not defaults)
    camp(tmp_path, '--sizes', '50x25n30', '--arms', 'memetic_rule@600 memetic_rule@1800:0-2')


def test_default_campaign_is_unchanged_by_the_training_size(tmp_path):
    c = camp(tmp_path)
    assert c.sizes == list(P.SIZES) and P.TRAINING_SIZE not in c.sizes
    for seg in c.segments:
        assert seg.by_size == {s: list(range(3, 30)) for s in P.SIZES}
        assert seg.instance_text() == '3-29'


def test_training_size_cpsat_cells_are_small(tmp_path):
    c = camp(tmp_path, '--arms', 'cpsat_rule', '--sizes', '10x25')
    for arm in P.PILOT_ARMS:
        c.a.cpsat_arm = arm
        for x in c.cells(c.segments[0])[:8]:
            assert c.need(x) == 4.0 and c.need_class(x) == 'cpsat-small'
            assert c.duration(x) == 1800 + P.OVERHEAD_S['cpsat-small']
    c.a.cpsat_arm = 'ref_fullhint'
    assert P.check_inputs(c) == []                      # no peer state needed: nothing large


def test_training_size_needs_equal_the_runner(tmp_path):
    pytest.importorskip('ortools')
    import p85_cpsat_arms as R
    for arm in P.PILOT_ARMS:
        for r in P.COMBOS:
            assert P.cpsat_need_gb(arm, P.TRAINING_SIZE, r) == R.needs_gb(arm, P.TRAINING_SIZE, r)[0]
            assert R.needs_gb(arm, P.TRAINING_SIZE, r) == (4.0, False)


def test_training_size_is_refused_without_its_cache_or_instances(tmp_path):
    c = camp(tmp_path, '--arms', 'memetic_rule@600', '--sizes', '10x25',
             '--rule-cache', str(tmp_path / 'cache_{size}'))
    ref = P.check_inputs(c)
    assert len(ref) == 1 and ref[0].startswith('memetic_rule@600: REFUSED, 240 rule schedules missing')
    assert str(tmp_path / 'cache_10x25' / 'N' / 'instance_000.npz') in ref[0]
    for r in P.COMBOS:                                  # all but one file
        d = tmp_path / 'cache_10x25' / r
        d.mkdir(parents=True, exist_ok=True)
        for i in range(30):
            if (r, i) != ('LSW', 29):
                (d / f'instance_{i:03d}.npz').write_bytes(b'')
    ref = P.check_inputs(c)
    assert len(ref) == 1 and 'REFUSED, 1 rule schedules missing' in ref[0]
    (tmp_path / 'cache_10x25' / 'LSW' / 'instance_029.npz').write_bytes(b'')
    assert P.check_inputs(c) == []
    c.a.strict, c.a.all_commands, c.a.layouts = True, False, []
    assert P.cmd_plan(c, c.a) == 0
    # an instance the set does not have (100 test instances, 000-099)
    c = camp(tmp_path, '--arms', 'memetic_rule@600:98-100', '--sizes', '10x25',
             '--rule-cache', str(tmp_path / 'cache_{size}'))
    ref = P.check_inputs(c)
    assert any('8 descriptors missing (first data/FAMILY/10x25/desc/N/instance_100.desc.json)' in r
               for r in ref)


def test_training_size_cache_in_the_repository_passes_the_plan(tmp_path):
    if not os.path.isdir(os.path.join(P.ROOT, TRAIN_CACHE)):
        pytest.skip('no training-size cache (scripts/p92_warmstart_xmask_training_size.py)')
    c = camp(tmp_path, '--arms', 'memetic_rule@600', '--sizes', '10x25', '--slots', '16-23')
    assert P.check_inputs(c) == []


def test_training_size_on_one_slot(tmp_path):
    c = camp(tmp_path, '--arms', 'memetic_rule@600', '--sizes', '10x25', '--slots', '16-23')
    started, idle, end = P.simulate(c, P.remaining_cells(c))
    assert [n for n, *_ in started] == ['A'] * 30 and all(len(b.cells) == 8 for _, b, _, _ in started)
    assert end == pytest.approx(30 * 605.0)              # 240 runs of 600 s, eight at a time
    for n, b, s, e in started:
        cs = [x for cc in P.batch_cores(c, b, list(range(16, 24))) for x in cc]
        assert cs == list(range(16, 24))                 # combination k on core 16 + k
        assert len({x.idx for x in b.cells}) == 1 and b.need == pytest.approx(6.0)
    # a finished cell is skipped, a cell of another protocol refused
    cell = c.cells(c.segments[0])[0]
    write_record(c, cell, affinity=[16])
    assert P.cell_states(c, c.segments[0])[cell.id][1] == 'valid'
    write_record(c, c.cells(c.segments[0])[1], warm_cache='or_solution/FAMILY/warmstart_cache')
    assert P.cell_states(c, c.segments[0])[c.cells(c.segments[0])[1].id][1] == 'foreign'


def test_launch_note_of_a_training_size_batch(tmp_path, monkeypatch):
    monkeypatch.setattr(P, 'busy_on', lambda cores, window: ([], []))
    monkeypatch.setattr(P, 'git_commit', lambda: 'abc1234')
    for sizes, want, purpose in (
            (['10x25'], 'no instance of 10x25 selected a configuration', 'at the training size'),
            (['50x25n30'], 'instances 000-002 selected the configurations and are not evidence',
             'the search side of the decision table')):
        c = camp(tmp_path, '--arms', 'memetic_rule@600', '--sizes', *sizes, '--slots', '16-23',
                 cmd='next')
        b = P.remaining_batches(c)[0][0]
        path = tmp_path / f'batch_{sizes[0]}.tsv'
        lines = [f'# slot=A cores=16-23 arm={b.seg.label} runner=memetic need_gb=6 class=memetic']
        for x, cc in zip(b.cells, P.batch_cores(c, b, list(range(16, 24)))):
            lines.append('\t'.join([str(cc[0]), x.id, c.record(x), 'log', 'cmd']))
        path.write_text('\n'.join(lines) + '\n')
        notes = tmp_path / f'notes_{sizes[0]}.txt'
        c.a.batch, c.a.notes, c.a.window, c.a.force = str(path), str(notes), 0.0, False
        assert P.cmd_prelaunch(c, c.a) == 0
        text = notes.read_text()
        assert want in text and purpose in text
        assert f'data/FAMILY/{sizes[0]}/desc' in text
