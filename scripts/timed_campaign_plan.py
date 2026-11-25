r"""Plan, batches, validity and trajectories of the timed campaign (tracker row C4),
run by scripts/run_timed_campaign.sh.

Arms (one long run per cell; shorter budgets are read from its trajectory):
  memetic_rule    memetic algorithm (eval_memetic_family.py), frozen default
                  configuration, seeded with the dispatching-rule schedule of the
                  exact-environment cache, one thread, 1800 s
  memetic_policy  as memetic_rule plus the policy's schedule (--seed_from)
  cpsat_rule      the CP-SAT arm the pilot selects (scripts/p85_cpsat_arms.py,
                  --cpsat-arm), 8 workers, hinted from the same rule cache
  cpsat_policy    as cpsat_rule, hinted with the policy's schedule
An arm token may carry its own budget and instances, name[@budget[:instances]],
e.g. 'cpsat_rule@600:3-29'; the defaults are --budget (1800 s) and --instances.
Given instances apply to every size. Without them each size takes its evidence
instances (DEFAULT_INSTANCES): 003-029 at the production sizes 50x25n30 and
80x25n30, whose instances 000-002 selected the configurations, and 000-029 at
the training size 10x25 (--sizes 10x25), where no selection was made. Two
tokens of one arm may not share an instance of one size, so a cell is never
rerun at a shorter budget.

Sizes: the production sets 50x25n30 and 80x25n30 (the default) and the
training size 10x25 (data/FAMILY/10x25; rule cache
or_solution/FAMILY/warmstart_cache_xmask_10x25, instances 000-029, built by
scripts/p92_warmstart_xmask_training_size.py).

Cells, batches and slots. A cell is (arm, size, combination, instance). Each
slot (SLOT_A, SLOT_B: core lists; default 8-15 and 16-23; slot B may have 0
to 8 cores) runs one batch at a time:
  memetic  one single-thread process per core of the slot; on a slot with a
           core per combination, at most one cell of each combination, which
           runs on that combination's core (on a full slot, the eight
           combinations of one instance at one size), else the next cells in
           order; so no core ever holds two processes. Every process of a
           batch has the same wall-clock budget, so the batch ends with them.
  CP-SAT   one cell: one solve with 8 workers, only on a slot with 8 cores.
A slot's current arm is the first arm in ARMS that it can run and that has a
cell not yet started (an arm thus starts on a slot once every cell of the arms
before it that the slot can run has started; a narrow slot B runs the memetic
arms while slot A runs CP-SAT). The slot takes the first batch of that arm
(or, when none of it fits, of a later arm) whose memory need, added to the
need of the batch the other slot is running, stays within the memory budget (--mem-budget, 40 GB: another project holds up
to about 20 GB of the 62), and never a second solve of LARGE_GB or more while
one runs (p85 announces such solves in one shared state file); if none fits,
the slot waits for the other slot's batch to end. Memory needs: CP-SAT cells
the value p85_cpsat_arms.needs_gb waits for (40 GB for a full model with
setups at 80 modules, 24 GB at 50 modules, 4 GB otherwise, so every cell at
10 modules); a memetic batch
--mem-memetic GB per process (0.75: 0.58 to 0.61 GB per solving process at 50
and 80 modules, measured on the smoke runs, plus 0.02 GB for its parent). So at
the default budget a slot that holds an 80-module full model with setups runs
beside an idle slot, and a 50-module one beside memetic batches or small
CP-SAT cells. The CP-SAT commands carry PACT_MEMORY_STATE (--own-state) and
PACT_PEER_MEMORY_STATE (--peer-state, required when such solves are planned;
'none' for no hold), read by p85's memory gate.

Record folders (the runners skip an existing record and never overwrite it):
  or_solution/FAMILY/<size>_memetic<B>s/<R>/instance_<i>_seed1.json
  or_solution/FAMILY/<size>_memetic<B>s_pactwarm_s<seed>/<R>/instance_<i>_seed1.json
  test_results/FAMILY/<size>_cparms/evidence_rule<B>s/<arm>/<R>/instance_<i>.json
  test_results/FAMILY/<size>_cparms/evidence_pactwarm<B>s_s<seed>/<arm>/<R>/instance_<i>.json
(--result-root moves both trees under another folder, for untimed smoke runs).
An invalid record moves to <R>/invalid/<time>_<name> and its cell runs again; a
cell with --max-invalid such files is held for a look by hand.

Validity (scripts/timing_validity.py): memetic records by
classify_single_thread (one core of a slot, cpu_share >= 0.9); CP-SAT records
by classify with the slots, the type key key_cparms and the campaign's frozen
bar file (--bars); a CP-SAT cell below 0.9 without a frozen bar is pending
(kept, not rerun, not yet evidence).

Subcommands:
  plan      the dry run: cells, commands, done counts, slot-hours, simulated
            pairing and wall-clock hours (--strict: exit 1 when a start must be
            refused)
  next      the next batch for a free slot (moves invalid records first);
            exit 2 when nothing is left, 5 when the slot must wait
  prelaunch process check on the batch's slot and the launch note; exit 1 when
            a process above 5% CPU is on the slot's cores
  classify  validity of a finished batch, invalid records moved aside; exit 3
            when every record of the batch is invalid, 4 when none was written
  table     tidy rows: incumbent at 1, 10, 60, 300, 600, 1800 s per cell
  select    the pilot's selection rule (report_cpsat_arms.md, section 4)
  bars      CPU-share bars per cell type of CP-SAT folders (--write merges new
            types into a bar file and never changes a frozen one)
No solver is imported here.
"""
import argparse
import csv
import glob
import json
import os
import shlex
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, ROOT)
import timing_validity as T  # noqa: E402
from baselines.memetic.records import incumbent_at  # noqa: E402

COMBOS = ('N', 'L', 'S', 'W', 'LS', 'LW', 'SW', 'LSW')
SETUP = {'S', 'LS', 'SW', 'LSW'}
SIZES = ('50x25n30', '80x25n30')
# the training size: data/FAMILY/10x25 holds 100 test instances, the campaign
# uses 000-029 (the instances of its rule cache)
TRAINING_SIZE = '10x25'
# the evidence instances of a size when no instances are given: 000-002 of the
# production sets selected the configurations (the CP-SAT pilot) and are left
# out; no selection was made at the training size. Any other size folder keeps
# the old default.
DEFAULT_INSTANCES = {'50x25n30': '3-29', '80x25n30': '3-29', TRAINING_SIZE: '0-29'}
OTHER_DEFAULT_INSTANCES = '3-29'
TIMES = (1, 10, 60, 300, 600, 1800)
ARMS = {'memetic_rule': ('memetic', 'rule'), 'memetic_policy': ('memetic', 'policy'),
        'cpsat_rule': ('cpsat', 'rule'), 'cpsat_policy': ('cpsat', 'policy')}
# the pilot's arms, in the order of simplicity of its tie rule
PILOT_ARMS = ('ref', 'ref_fullhint', 'ref_presolve_light_fullhint',
              'ref_lnsonly_fullhint', 'compact', 'fixassign', 'radius', 'lns')
CPSAT_KIND = {'ref': 'ref', 'ref_fullhint': 'ref', 'ref_presolve_light_fullhint': 'ref',
              'ref_lnsonly_fullhint': 'ref', 'compact': 'full', 'fixassign': 'full',
              'radius': 'full', 'lns': 'lns'}
PILOT_REGIMES, PILOT_INSTANCES = ('S', 'SW', 'LSW', 'N'), (0, 1, 2)
TIE_PP = 0.10
# Seconds a cell holds its slot beyond the budget (estimates for the plan):
# the memetic loads the instance, validates the seeds and loads the numba
# kernels from the cache before its window; a p85 cell imports the solver,
# validates the hint twice, builds the model outside the budget for full
# models (26 s at 50x25 S, about 65 s at 80x25, report_cpsat_arms.md) and
# waits 5 s on its empty queue at the end.
OVERHEAD_S = {'memetic': 5.0, 'cpsat-big80': 80.0, 'cpsat-big50': 40.0, 'cpsat-small': 10.0}
# p85 announces a solve that waits for this much memory or more in one shared
# state file (PACT_MEMORY_STATE) and holds it while the other project loads
# (PACT_PEER_MEMORY_STATE, PACT_PEER_MIN_GB), so one such solve at a time
LARGE_GB = 30.0
LAYOUTS = ('8-15 16-23', '8-15 16-19')

ARM_TEXT = {
    'memetic_rule': dict(
        purpose='anytime trajectory of the memetic algorithm seeded with the dispatching-rule '
                'schedule (incumbent at 1, 10, 60, 300, 600, 1800 s from one run); the search '
                'side of the decision table by time budget (C4)',
        expected='shorter than the policy within minutes on N, S and SW, behind it on LSW at 80 '
                 'modules (development figures, round two); if it is behind everywhere, the policy '
                 'leads at every budget; if ahead everywhere, the policy rests on response time'),
    'memetic_policy': dict(
        purpose='same search started also from the policy schedule: does the policy improve the '
                'search at every budget (paired with memetic_rule per cell)',
        expected='no longer than memetic_rule at every budget on most cells (development: shorter '
                 'on 20 of 24); if not, the policy adds nothing as a start to this search'),
    'cpsat_rule': dict(
        purpose='anytime trajectory of the CP-SAT comparator selected by the pilot, hinted with '
                'the dispatching-rule schedule; the solver side of the decision table (C4)',
        expected='improves the hint on the combinations without setups within seconds and on the '
                 'setup combinations after presolve; if it never leaves the hint on setups, the '
                 'solver claim is stated as such'),
    'cpsat_policy': dict(
        purpose='same CP-SAT arm hinted with the policy schedule: the policy as a start for the '
                'solver (paired with cpsat_rule per cell)',
        expected='no longer than cpsat_rule at every budget; if longer, the hint does not carry '
                 'over to the solver'),
}
# launch-note text of a batch whose cells are all at the training size
ARM_TEXT_TRAINING = {
    'memetic_rule': dict(
        purpose='anytime trajectory of the memetic algorithm seeded with the dispatching-rule '
                'schedule at the training size (incumbent at each budget up to the run\'s budget, '
                'from one run), so that the same metaheuristic appears at 10, 50 and 80 modules in '
                'the time-budget exhibit (Q3); the policy is paired on these instances',
        expected='search with minutes is ahead of one greedy schedule at this size (the "search is '
                 'the better tool" row of the decision table); if not, that row says so'),
}


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

def parse_instances(spec):
    out = []
    for part in str(spec).split(','):
        part = part.strip()
        if '-' in part:
            a, b = part.split('-')
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return sorted(set(out))


def compress(nums):
    """[3, 4, 5, 9] -> '3-5,9'"""
    nums, out = sorted(nums), []
    i = 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        out.append(f'{nums[i]}' if i == j else f'{nums[i]}-{nums[j]}')
        i = j + 1
    return ','.join(out)


def parse_cores(spec):
    """'8-15' -> [8..15]; '16-17,20' -> [16, 17, 20]."""
    out = []
    for part in str(spec).split(','):
        part = part.strip()
        if not part:
            continue
        a, b = (part.split('-') + [part])[:2]
        a, b = int(a), int(b)
        if b < a:
            raise ValueError(f'bad core list {spec}')
        out += list(range(a, b + 1))
    if len(set(out)) != len(out):
        raise ValueError(f'core listed twice in {spec}')
    return sorted(out)


def contiguous(cores):
    return cores == list(range(cores[0], cores[-1] + 1))


def bnum(b):
    return f'{b:g}'


def rel(path):
    p = os.path.abspath(os.path.join(ROOT, path))
    return os.path.relpath(p, ROOT) if p.startswith(ROOT + os.sep) else p


def make_slots(specs):
    """[(name, cores, spec)] for the non-empty core lists, named A, B, ..."""
    slots = []
    for spec in specs:
        if str(spec).strip():
            slots.append((chr(ord('A') + len(slots)), parse_cores(spec), spec))
    if not slots:
        raise SystemExit('REFUSING: no slot')
    used = [c for _, cs, _ in slots for c in cs]
    if len(set(used)) != len(used):
        raise SystemExit('REFUSING: slots overlap')
    return slots


class Segment:
    """One arm token. instances: a list that applies to every size, or None
    for each size's evidence instances (DEFAULT_INSTANCES). by_size holds the
    instances per size; instances is their union, in order."""

    def __init__(self, arm, budget, instances, sizes):
        self.arm, self.budget = arm, float(budget)
        self.by_size = {s: (list(instances) if instances is not None else
                            parse_instances(DEFAULT_INSTANCES.get(s, OTHER_DEFAULT_INSTANCES)))
                        for s in sizes}
        self.instances = sorted(set().union(*self.by_size.values()))
        self.runner, self.start = ARMS[arm]
        self.label = f'{arm}@{bnum(self.budget)}'

    def instance_text(self):
        """'3-29' when every size has the same instances, else per size."""
        lists = list(self.by_size.values())
        if all(v == lists[0] for v in lists):
            return compress(lists[0])
        return '; '.join(f'{compress(v)} at {s}' for s, v in self.by_size.items())


class Cell:
    __slots__ = ('seg', 'size', 'combo', 'idx')

    def __init__(self, seg, size, combo, idx):
        self.seg, self.size, self.combo, self.idx = seg, size, combo, idx

    @property
    def id(self):
        return f'{self.seg.label}/{self.size}/{self.combo}/{self.idx:03d}'

    @property
    def name(self):
        return f'instance_{self.idx:03d}'


class Campaign:
    def __init__(self, a):
        self.a = a
        self.sizes = list(a.sizes)
        self.combos = list(a.combos)
        for c in self.combos:
            if c not in COMBOS:
                raise SystemExit(f'unknown combination {c}')
        self.slots = make_slots(a.slots)
        self.width_max = max(len(cs) for _, cs, _ in self.slots)
        self.segments = []
        for tok in a.arms.split():
            arm, _, rest = tok.partition('@')
            if arm not in ARMS:
                raise SystemExit(f'unknown arm {arm} (one of {", ".join(ARMS)})')
            b, _, inst = rest.partition(':')
            spec = inst or a.instances            # none given: each size's default
            self.segments.append(Segment(arm, b or a.budget,
                                         parse_instances(spec) if spec else None, self.sizes))
        seen = {}
        for s in self.segments:
            for size, insts in s.by_size.items():
                for i in insts:
                    if (s.arm, size, i) in seen:
                        raise SystemExit(f'REFUSING: instance {i} of {s.arm} in two tokens '
                                         f'({seen[s.arm, size, i]} and {s.label}); a cell is '
                                         f'never rerun')
                    seen[s.arm, size, i] = s.label
        if a.cpsat_arm not in PILOT_ARMS:
            raise SystemExit(f'unknown CP-SAT arm {a.cpsat_arm} (one of {", ".join(PILOT_ARMS)})')

    # folders ---------------------------------------------------------------
    def or_root(self):
        return (os.path.join(self.a.result_root, 'or_solution', 'FAMILY') if self.a.result_root
                else 'or_solution/FAMILY')

    def tr_root(self):
        return (os.path.join(self.a.result_root, 'test_results', 'FAMILY') if self.a.result_root
                else 'test_results/FAMILY')

    def rule_cache(self, size):
        return self.a.rule_cache.format(size=size)

    def policy_cache(self, size):
        return self.a.policy_cache.format(seed=self.a.policy_seed, size=size)

    def cpsat_out(self, seg):
        b = bnum(seg.budget)
        return (f'evidence_rule{b}s' if seg.start == 'rule'
                else f'evidence_pactwarm{b}s_s{self.a.policy_seed}')

    def folder(self, seg, size):
        b = bnum(seg.budget)
        if seg.runner == 'memetic':
            f = f'{size}_memetic{b}s' + ('' if seg.start == 'rule'
                                          else f'_pactwarm_s{self.a.policy_seed}')
            return os.path.join(self.or_root(), f)
        return os.path.join(self.tr_root(), f'{size}_cparms', self.cpsat_out(seg), self.a.cpsat_arm)

    def record(self, c):
        d = os.path.join(self.folder(c.seg, c.size), c.combo)
        if c.seg.runner == 'memetic':
            return os.path.join(d, f'{c.name}_seed{self.a.memetic_seed}.json')
        return os.path.join(d, f'{c.name}.json')

    def start_files(self, c):
        """The schedule files a cell starts from (all must exist)."""
        out = [os.path.join(self.rule_cache(c.size), c.combo, f'{c.name}.npz')]
        if c.seg.start == 'policy':
            pol = os.path.join(self.policy_cache(c.size), c.combo, f'{c.name}.npz')
            out = [pol] if c.seg.runner == 'cpsat' else out + [pol]
        return out

    # cells -----------------------------------------------------------------
    def cells(self, seg):
        """Instance-major: every combination and size of one instance first
        (a size takes only its own instances)."""
        own = {s: set(v) for s, v in seg.by_size.items()}
        return [Cell(seg, size, combo, i) for i in seg.instances
                for size in self.sizes if i in own[size] for combo in self.combos]

    def need_class(self, c):
        if c.seg.runner == 'memetic':
            return 'memetic'
        n = cpsat_need_gb(self.a.cpsat_arm, c.size, c.combo)
        return {40.0: 'cpsat-big80', 24.0: 'cpsat-big50'}.get(n, 'cpsat-small')

    def need(self, c):
        if c.seg.runner == 'memetic':
            return self.a.mem_memetic
        return cpsat_need_gb(self.a.cpsat_arm, c.size, c.combo)

    def duration(self, c):
        return c.seg.budget + OVERHEAD_S[self.need_class(c)]

    def command(self, c, cores):
        py = self.a.py
        env = ['env', 'OMP_NUM_THREADS=1', 'MKL_NUM_THREADS=1', 'CUDA_VISIBLE_DEVICES=']
        if c.seg.runner == 'memetic':
            cmd = (['taskset', '-c', str(cores[0])] + env + ['NUMBA_NUM_THREADS=1', py,
                   'eval_memetic_family.py', '--size', c.size, '--regimes', c.combo,
                   '--instances', str(c.idx), '--budget', bnum(c.seg.budget),
                   '--seed', str(self.a.memetic_seed), '--seed_schedules',
                   '--warm_cache', rel(self.rule_cache(c.size))])
            if c.seg.start == 'policy':
                cmd += ['--seed_from', rel(self.policy_cache(c.size))]
            return cmd + ['--out', rel(self.folder(c.seg, c.size))]
        spec = f'{cores[0]}-{cores[-1]}'
        hint = self.rule_cache(c.size) if c.seg.start == 'rule' else self.policy_cache(c.size)
        env += [f'PACT_MEMORY_STATE={rel(self.a.own_state)}']
        if self.a.peer_state and self.a.peer_state != 'none':
            env += [f'PACT_PEER_MEMORY_STATE={self.a.peer_state}']
        cmd = (['taskset', '-c', spec] + env + [py, 'scripts/p85_cpsat_arms.py',
               '--arm', self.a.cpsat_arm, '--size', c.size, '--regimes', c.combo,
               '--instances', str(c.idx), '--budget', bnum(c.seg.budget),
               '--workers', str(self.a.workers), '--jobs', '1',
               '--seed', str(self.a.cpsat_seed), '--slots', spec, '--hint', rel(hint),
               '--out', self.cpsat_out(c.seg), '--solver-log'])
        if self.a.result_root:
            cmd += ['--root', rel(self.tr_root())]
        return cmd

    def expected_fields(self, c):
        """Fields an existing record must carry to count for this plan."""
        e = {'size': c.size, 'regime': c.combo, 'instance': c.name, 'budget_s': c.seg.budget}
        if c.seg.runner == 'memetic':
            e.update(seed=self.a.memetic_seed, budget_clock='wall', max_decodes=None,
                     seed_schedules=True, warm_cache=rel(self.rule_cache(c.size)),
                     seed_from=[rel(self.policy_cache(c.size))] if c.seg.start == 'policy' else [])
        else:
            hint = self.rule_cache(c.size) if c.seg.start == 'rule' else self.policy_cache(c.size)
            e.update(arm=self.a.cpsat_arm, workers=self.a.workers, seed=self.a.cpsat_seed,
                     hint_source=rel(hint), development_run=False)
        return e


def cpsat_need_gb(arm, size, regime):
    """p85_cpsat_arms.needs_gb, without the solver import (a test keeps the two equal)."""
    big = CPSAT_KIND[arm] in ('ref', 'full') and arm not in ('fixassign', 'radius') \
        and regime in SETUP
    if big and size.startswith('80'):
        return 40.0
    if big and size.startswith('50'):
        return 24.0
    return 4.0


# ---------------------------------------------------------------------------
# state of the cells
# ---------------------------------------------------------------------------

def invalid_dir(path):
    return os.path.join(os.path.dirname(path), 'invalid')


def n_invalid(path):
    base = os.path.basename(path)
    return len(glob.glob(os.path.join(invalid_dir(path), f'*_{base}')))


def classify_paths(camp, runner, paths):
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        return {}
    slots = [cs for _, cs, _ in camp.slots]
    if runner == 'memetic':
        return T.classify_single_thread(paths, slots=slots)
    out = T.classify(paths, slots=slots, thresh=camp.a.bars, key=T.key_cparms)
    for p in paths:            # a record without hint == 'complete' is not a p85 record
        out.setdefault(p, (False, 'not a p85 record'))
    return out


def foreign_reason(camp, c, rec):
    for k, v in camp.expected_fields(c).items():
        have = rec.get(k)
        if isinstance(v, float) and isinstance(have, (int, float)):
            if abs(have - v) > 1e-9:
                return f'{k}={have!r}, plan {v!r}'
        elif have != v:
            return f'{k}={have!r}, plan {v!r}'
    return None


def cell_states(camp, seg):
    """{cell id: (cell, state, reason)}, state in valid, pending, invalid,
    foreign, held, missing."""
    cells = camp.cells(seg)
    recs = {c.id: camp.record(c) for c in cells}
    cls = classify_paths(camp, seg.runner, list(recs.values()))
    out = {}
    for c in cells:
        p = recs[c.id]
        if os.path.exists(p):
            fr = foreign_reason(camp, c, json.load(open(p)))
            if fr:
                out[c.id] = (c, 'foreign', fr)
                continue
            ok, why = cls[p]
            out[c.id] = (c, {True: 'valid', None: 'pending', False: 'invalid'}[ok], why)
        elif n_invalid(p) >= camp.a.max_invalid:
            out[c.id] = (c, 'held', f'{n_invalid(p)} invalid records in {invalid_dir(p)}')
        else:
            out[c.id] = (c, 'missing', '')
    return out


def move_invalid(path, why, log=None):
    d = invalid_dir(path)
    os.makedirs(d, exist_ok=True)
    stamp = time.strftime('%Y%m%d-%H%M%S') + f'.{time.time_ns() // 1000 % 1000000:06d}'
    moved = []
    files = [path]
    sched = path[:-len('.json')] + '.schedule.npz'
    if os.path.exists(sched):
        files.append(sched)
    for f in files:
        dst = os.path.join(d, f'{stamp}_{os.path.basename(f)}')
        try:
            os.replace(f, dst)
            moved.append(dst)
        except FileNotFoundError:
            pass
    if log is not None:
        log.write(f'moved invalid record ({why}): {path} -> {d}/\n')
    return moved


# ---------------------------------------------------------------------------
# batches and the choice of a slot's next batch
# ---------------------------------------------------------------------------

class Batch:
    def __init__(self, camp, cells):
        self.cells = cells
        self.seg = cells[0].seg
        self.need = (len(cells) * camp.a.mem_memetic if self.seg.runner == 'memetic'
                     else max(camp.need(c) for c in cells))
        self.large = self.need >= LARGE_GB
        self.duration = max(camp.duration(c) for c in cells)
        self.cls = camp.need_class(cells[0])
        self.ids = {c.id for c in cells}


def memetic_batch(camp, cells, width):
    """The next memetic batch of a slot with `width` cores, from the arm's
    remaining cells in order: with a core per combination, at most one cell of
    each combination (it runs on that combination's core); else the first
    `width` cells."""
    if width >= len(camp.combos):
        out, seen = [], set()
        for c in cells:
            if c.combo not in seen:
                out.append(c)
                seen.add(c.combo)
                if len(out) == len(camp.combos):
                    break
        return out
    return cells[:width]


def batch_cores(camp, batch, cores):
    """Cores of the batch's cells: combination k on the k-th core of the slot
    when the slot has a core per combination, else in batch order."""
    if batch.seg.runner == 'cpsat':
        return [cores for _ in batch.cells]
    if len(cores) >= len(camp.combos):
        return [[cores[camp.combos.index(c.combo)]] for c in batch.cells]
    return [[cores[j]] for j in range(len(batch.cells))]


def can_run(camp, seg, width):
    """CP-SAT arms need a slot with a core per worker; memetic arms any slot."""
    return width >= camp.a.workers if seg.runner == 'cpsat' else width >= 1


def choose(camp, per_seg, width, other_need, other_large=False):
    """per_seg: list over arms (in ARMS order) of the remaining cells not yet
    started, in order. The slot's current arm is the first arm it can run
    that has such cells; it takes that arm's first batch that fits beside the
    other slot (memory budget; one solve of LARGE_GB or more at a time). When
    none of it fits, the first fitting batch of a later arm the slot can run
    fills the wait (with the default budget nothing fits beside an 80-module
    full model with setups). Returns a Batch, 'wait' or 'done'."""
    fits = lambda b: (b.need + other_need <= camp.a.mem_budget + 1e-9  # noqa: E731
                      and not (b.large and other_large))
    result = 'done'
    for seg, cells in zip(camp.segments, per_seg):
        if not cells or not can_run(camp, seg, width):
            continue
        if seg.runner == 'memetic':
            b = Batch(camp, memetic_batch(camp, cells, width))
            if fits(b):
                return b
        else:
            for c in cells:
                b = Batch(camp, [c])
                if fits(b):
                    return b
        result = 'wait'
    return result


def remaining_cells(camp, states=None, skip=(), running_ids=()):
    per_seg = []
    for seg in camp.segments:
        st = states[seg.label] if states else cell_states(camp, seg)
        per_seg.append([c for c, s, _ in st.values() if s in ('missing', 'invalid')
                        and c.id not in skip and c.id not in running_ids])
    return per_seg


def remaining_batches(camp, states=None, skip=(), running_ids=()):
    """The batches of a full-width slot, in order (for counts and tests)."""
    out = []
    for seg, cells in zip(camp.segments, remaining_cells(camp, states, skip, running_ids)):
        lst, left = [], list(cells)
        while left:
            pick = (memetic_batch(camp, left, camp.width_max) if seg.runner == 'memetic'
                    else left[:1])
            lst.append(Batch(camp, pick))
            ids = {c.id for c in pick}
            left = [c for c in left if c.id not in ids]
        out.append(lst)
    return out


def simulate(camp, per_seg, budget=None, slots=None):
    """Discrete-event run of choose() with the estimated durations.
    Returns the started batches [(slot, batch, start, end)], idle seconds per
    slot while waiting for memory, and the end time."""
    budget = camp.a.mem_budget if budget is None else budget
    slots = slots or camp.slots
    saved, camp.a.mem_budget = camp.a.mem_budget, budget
    names = [n for n, _, _ in slots]
    width = {n: len(cs) for n, cs, _ in slots}
    free = {n: 0.0 for n in names}
    cur = {n: None for n in names}            # (batch, end)
    retired = set()
    started, idle_mem = [], {n: 0.0 for n in names}
    left = [list(c) for c in per_seg]
    try:
        while len(retired) < len(names):
            n = min((x for x in names if x not in retired), key=lambda x: (free[x], names.index(x)))
            t = free[n]
            others = [cur[o] for o in names if o != n and cur[o] is not None and cur[o][1] > t + 1e-9]
            pick = choose(camp, left, width[n], sum(b.need for b, _ in others),
                          any(b.large for b, _ in others))
            if pick == 'done':
                retired.add(n)
                cur[n] = None
                continue
            if pick == 'wait':
                if not others:
                    raise SystemExit('REFUSING: a batch needs more memory than the budget alone')
                nxt = min(e for _, e in others)
                idle_mem[n] += nxt - t
                free[n] = nxt
                cur[n] = None
                continue
            k = camp.segments.index(pick.seg)
            left[k] = [c for c in left[k] if c.id not in pick.ids]
            cur[n] = (pick, t + pick.duration)
            started.append((n, pick, t, t + pick.duration))
            free[n] = t + pick.duration
    finally:
        camp.a.mem_budget = saved
    if any(left):
        raise SystemExit('REFUSING: no slot can run ' + ', '.join(
            s.label for s, c in zip(camp.segments, left) if c))
    end = max((e for *_, e in started), default=0.0)
    return started, idle_mem, end


def pairing(slots, started, end):
    """Hours with each combination of batch classes running at once."""
    names = [n for n, _, _ in slots]
    ivs = {n: sorted((s, e, b.cls) for x, b, s, e in started if x == n) for n in names}
    pts = sorted({0.0, end} | {s for v in ivs.values() for s, _, _ in v}
                 | {e for v in ivs.values() for _, e, _ in v})
    acc = {}
    for a, b in zip(pts, pts[1:]):
        mid = (a + b) / 2
        key = []
        for n in names:
            c = next((cls for s, e, cls in ivs[n] if s <= mid < e), 'idle')
            key.append(c)
        k = ' + '.join(sorted(key))
        acc[k] = acc.get(k, 0.0) + (b - a) / 3600
    return acc


# ---------------------------------------------------------------------------
# plan (dry run)
# ---------------------------------------------------------------------------

def check_inputs(camp):
    """Refusals that block a real start: missing descriptors, rule caches or
    policy schedules, records that do not match the plan, oversize batches."""
    refuse = []
    for seg in camp.segments:
        if seg.start == 'policy' and camp.a.policy_seed is None:
            refuse.append(f'{seg.label}: no --policy-seed')
            continue
        miss_desc, miss_start = [], []
        for c in camp.cells(seg):
            dp = os.path.join('data/FAMILY', c.size, 'desc', c.combo, f'{c.name}.desc.json')
            if not os.path.exists(os.path.join(ROOT, dp)):
                miss_desc.append(dp)
            for f in camp.start_files(c):
                if not os.path.exists(os.path.join(ROOT, f)):
                    miss_start.append(f)
        if miss_desc:
            refuse.append(f'{seg.label}: {len(miss_desc)} descriptors missing (first {miss_desc[0]})')
        if miss_start:
            kind = 'policy schedules' if seg.start == 'policy' else 'rule schedules'
            refuse.append(f'{seg.label}: REFUSED, {len(miss_start)} {kind} missing '
                          f'(first {miss_start[0]})')
        if seg.runner == 'cpsat':
            ok = [n for n, cs, _ in camp.slots if can_run(camp, seg, len(cs))]
            if not ok:
                refuse.append(f'{seg.label}: no slot with {camp.a.workers} cores')
            for n, cs, spec in camp.slots:
                if n in ok and not contiguous(cs):
                    refuse.append(f'{seg.label}: slot {n} ({spec}) is not one range of cores '
                                  f'(p85 --slots takes a-b)')
            if any(camp.need(c) >= LARGE_GB for c in camp.cells(seg)) and not camp.a.peer_state:
                refuse.append(f'{seg.label}: full models of {LARGE_GB:g} GB or more need '
                              f'PEER_STATE (the other project\'s memory state file, or none)')
        big = max((camp.need(c) for c in camp.cells(seg)), default=0)
        if seg.runner == 'memetic':
            big = min(camp.width_max, len(camp.combos)) * camp.a.mem_memetic
        if big > camp.a.mem_budget + 1e-9:
            refuse.append(f'{seg.label}: a batch needs {big:g} GB, above the budget '
                          f'{camp.a.mem_budget:g} GB')
    if camp.a.control_core is not None:
        for n, cs, spec in camp.slots:
            if camp.a.control_core in cs:
                refuse.append(f'the control core {camp.a.control_core} lies in slot {n} ({spec})')
    return refuse


def summarize_cells(cells):
    by = {}
    for c in cells:
        by.setdefault((c.size, c.combo), []).append(c.idx)
    return [f'{s} {r}: {compress(v)}' for (s, r), v in sorted(
        by.items(), key=lambda kv: (kv[0][0], COMBOS.index(kv[0][1])))]


def cmd_plan(camp, a):
    out = sys.stdout
    names = [n for n, _, _ in camp.slots]
    states = {seg.label: cell_states(camp, seg) for seg in camp.segments}
    print(f'timed campaign plan, {time.strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'slots: {", ".join(f"{n} = cores {spec}" for n, _, spec in camp.slots)}; '
          f'memory budget {camp.a.mem_budget:g} GB; memetic {camp.a.mem_memetic:g} GB per '
          f'process; CP-SAT arm {camp.a.cpsat_arm}, {camp.a.workers} workers'
          + ('' if camp.a.workers == 8 else ' (SMOKE ONLY: not the campaign setting)'))
    if camp.a.result_root:
        print(f'result root {camp.a.result_root} (not the repository folders)')
    refuse = check_inputs(camp)
    for seg in camp.segments:
        st = states[seg.label]
        cnt = {}
        for _, s, _ in st.values():
            cnt[s] = cnt.get(s, 0) + 1
        print(f'\n== {seg.label}: {seg.runner}, start from the {seg.start} schedule, budget '
              f'{bnum(seg.budget)} s, instances {seg.instance_text()}, '
              f'{len(camp.sizes)} sizes x {len(camp.combos)} combinations')
        print(f'   cells {len(st)}: valid {cnt.get("valid", 0)}, pending {cnt.get("pending", 0)}, '
              f'invalid (moved and rerun at start) {cnt.get("invalid", 0)}, held '
              f'{cnt.get("held", 0)}, foreign {cnt.get("foreign", 0)}, to run '
              f'{cnt.get("missing", 0) + cnt.get("invalid", 0)}')
        for c, s, why in st.values():
            if s in ('foreign', 'held'):
                print(f'   {s.upper()} {c.id}: {why}')
            if s == 'foreign':
                refuse.append(f'{c.id}: a record that does not match the plan ({why})')
        more = {1: '', 2: ' (and the other size)'}.get(len(camp.sizes), ' (and the other sizes)')
        print(f'   folder {camp.folder(seg, camp.sizes[0])}{more}')
    per_seg = remaining_cells(camp, states)
    started, idle_mem, end = simulate(camp, per_seg)
    print('\n== simulated run (estimated durations: budget plus '
          + ', '.join(f'{k} {v:g} s' for k, v in OVERHEAD_S.items()) + ')')
    t_prev = {n: 0.0 for n in names}
    for seg in camp.segments:
        rows = [(n, b, s, e) for n, b, s, e in started if b.seg is seg]
        if not rows:
            print(f'\n-- {seg.label}: nothing to run')
            continue
        s0, e0 = min(r[2] for r in rows), max(r[3] for r in rows)
        bud = sum(seg.budget for _ in rows) / 3600
        occ = sum(e - s for _, _, s, e in rows) / 3600
        print(f'\n-- {seg.label}: {sum(len(b.cells) for _, b, _, _ in rows)} cells in '
              f'{len(rows)} batches; {bud:.1f} slot-hours of budget, {occ:.1f} slot-hours '
              f'with overheads; wall clock {(e0 - s0) / 3600:.1f} h (hours {s0 / 3600:.1f} to '
              f'{e0 / 3600:.1f} of the run)')
        for n, cores, spec in camp.slots:
            mine = [(b, s, e) for x, b, s, e in rows if x == n]
            if not mine:
                print(f'   slot {n} ({spec}): no cell')
                continue
            cells = [c for b, _, _ in mine for c in b.cells]
            print(f'   slot {n} ({spec}): {len(cells)} cells, {len(mine)} batches, '
                  f'{sum(e - s for _, s, e in mine) / 3600:.1f} slot-hours')
            for line in summarize_cells(cells):
                print(f'     {line}')
            b0 = mine[0][0]
            print('     first batch:')
            for c, cc in zip(b0.cells, batch_cores(camp, b0, cores)):
                print('       ' + shlex.join(camp.command(c, cc)))
    print(f'\n== whole run: wall clock {end / 3600:.1f} h on {len(names)} slot(s); idle while '
          f'waiting for memory: ' + ', '.join(f'slot {n} {idle_mem[n] / 3600:.1f} h' for n in names))
    for n in names:
        busy = sum(e - s for x, _, s, e in started if x == n)
        print(f'   slot {n}: busy {busy / 3600:.1f} h, idle {(end - busy) / 3600:.1f} h '
              f'(memory {idle_mem[n] / 3600:.1f}, end of run {(end - busy - idle_mem[n]) / 3600:.1f})')
    if len(names) > 1 and started:
        print('   hours by what the slots run at once:')
        for k, h in sorted(pairing(camp.slots, started, end).items(), key=lambda kv: -kv[1]):
            print(f'     {k:30s} {h:7.1f} h')
        free_end = simulate(camp, remaining_cells(camp, states), budget=1e9)[2]
        print(f'   without the memory rule the same cells take {free_end / 3600:.1f} h; '
              f'the rule costs {(end - free_end) / 3600:.1f} h of wall clock')
    if a.layouts:
        print('\n== the same cells on other slot layouts (same rules and estimates)')
        for lay in a.layouts:
            sl = make_slots(lay.split())
            try:
                st_, idle_, end_ = simulate(camp, remaining_cells(camp, states), slots=sl)
            except SystemExit as e:
                print(f'   {lay:22s} {e}')
                continue
            per = []
            for seg in camp.segments:
                r = [x for x in st_ if x[1].seg is seg]
                if r:
                    per.append(f'{seg.label} {(max(x[3] for x in r) - min(x[2] for x in r)) / 3600:.1f} h')
            busy = {n: sum(e - s for x, _, s, e in st_ if x == n) for n, _, _ in sl}
            print(f'   slots {" + ".join(f"{sp} ({len(cs)} cores)" for _, cs, sp in sl)}: wall '
                  f'clock {end_ / 3600:.1f} h; ' + '; '.join(per) + '; idle '
                  + ', '.join(f'{n} {(end_ - busy[n]) / 3600:.1f} h (memory {idle_[n] / 3600:.1f})'
                              for n, _, _ in sl))
    if a.all_commands:
        print('\n== every command, in the simulated order')
        for n, b, s, e in started:
            cores = dict((x, cs) for x, cs, _ in camp.slots)[n]
            for c, cc in zip(b.cells, batch_cores(camp, b, cores)):
                print(f'[{n} {s / 3600:7.2f} h] ' + shlex.join(camp.command(c, cc)))
    if refuse:
        print('\nA REAL START WOULD BE REFUSED:')
        for r in refuse:
            print(f'  {r}')
    return 1 if (refuse and a.strict) else 0


# ---------------------------------------------------------------------------
# launcher steps: next, prelaunch, classify
# ---------------------------------------------------------------------------

def cell_of(camp, cid):
    label, size, combo, idx = cid.split('/')
    seg = next(s for s in camp.segments if s.label == label)
    return Cell(seg, size, combo, int(idx))


def read_batch(path):
    head, lines = {}, []
    with open(path) as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith('#'):
                for kv in line[1:].split():
                    k, _, v = kv.partition('=')
                    head[k] = v
            elif line:
                cores, cid, rec, log, cmd = line.split('\t')
                lines.append(dict(cores=cores, id=cid, record=rec, log=log, cmd=cmd))
    return head, lines


def running(state):
    ids, need = {}, {}
    for p in glob.glob(os.path.join(state, 'running_*.tsv')):
        h, lines = read_batch(p)
        ids[h['slot']] = {x['id'] for x in lines}
        need[h['slot']] = float(h.get('need_gb', 0))
    return ids, need


def read_skip(state):
    p = os.path.join(state, 'skip.txt')
    return set(open(p).read().split()) if os.path.exists(p) else set()


def cmd_next(camp, a):
    names = {n: (cs, spec) for n, cs, spec in camp.slots}
    if a.slot not in names:
        raise SystemExit(f'unknown slot {a.slot}')
    cores, spec = names[a.slot]
    run_ids, run_need = running(a.state)
    others = set().union(*[v for k, v in run_ids.items() if k != a.slot]) if run_ids else set()
    other_need = sum(v for k, v in run_need.items() if k != a.slot)
    other_large = any(v >= LARGE_GB for k, v in run_need.items() if k != a.slot)
    states = {}
    for seg in camp.segments:
        st = cell_states(camp, seg)
        for cid, (c, s, why) in list(st.items()):
            if s == 'invalid' and cid not in others:
                move_invalid(camp.record(c), why, sys.stderr)
            if s == 'foreign':
                raise SystemExit(f'REFUSING: {cid}: record does not match the plan ({why})')
        states[seg.label] = st
    per_seg = remaining_cells(camp, states, skip=read_skip(a.state), running_ids=others)
    pick = choose(camp, per_seg, len(cores), other_need, other_large)
    if pick == 'done':
        print('done')
        return 2
    if pick == 'wait':
        print(f'wait: no batch of the current arm fits beside {other_need:g} GB')
        return 5
    logdir = os.path.join(a.logdir, 'timed_campaign', pick.seg.label.replace('@', '_'))
    os.makedirs(logdir, exist_ok=True)
    print(f'# slot={a.slot} cores={spec} arm={pick.seg.label} runner={pick.seg.runner} '
          f'need_gb={pick.need:g} class={pick.cls} cells={len(pick.cells)}')
    for c, cc in zip(pick.cells, batch_cores(camp, pick, cores)):
        log = os.path.join(logdir, f'{c.size}_{c.combo}_{c.name}.log')
        print('\t'.join([','.join(map(str, cc)), c.id, camp.record(c), log,
                         shlex.join(camp.command(c, cc))]))
    return 0


def _proc_snapshot():
    snap = {}
    for tdir in glob.glob('/proc/[0-9]*/task/[0-9]*'):
        try:
            s = open(tdir + '/stat').read()
        except OSError:
            continue
        f = s[s.rfind(')') + 2:].split()
        snap[tdir] = (int(f[11]) + int(f[12]), int(f[36]))
    return snap


def _cmdline(pid):
    try:
        return open(f'/proc/{pid}/cmdline', 'rb').read().replace(b'\0', b' ').decode(
            errors='replace').strip()[:150]
    except OSError:
        return '?'


def _ancestors():
    out, pid = set(), os.getpid()
    while pid > 1:
        out.add(pid)
        try:
            pid = int(open(f'/proc/{pid}/stat').read().rsplit(')', 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
    return out


def busy_on(cores, window=2.0):
    """Processes on the given cores: threads that used more than 5% of a core
    over the sampling window, and processes whose lifetime CPU share (ps
    pcpu) is above 5% and that last ran there (the pilot launcher's rule)."""
    cores, mine = set(cores), _ancestors()
    a = _proc_snapshot()
    time.sleep(window)
    b = _proc_snapshot()
    hz = os.sysconf('SC_CLK_TCK')
    busy = {}
    for k, (tb, cpu) in b.items():
        pid = int(k.split('/')[2])
        if cpu not in cores or pid in mine:
            continue
        pct = 100.0 * (tb - a.get(k, (tb, cpu))[0]) / hz / window
        if pct > 5:
            busy[pid] = f'pid {pid} thread {k.split("/")[4]} core {cpu} {pct:.0f}% now: {_cmdline(pid)}'
    ps = subprocess.run(['ps', '-eo', 'pid,psr,pcpu,etime,args', '--no-headers'],
                        capture_output=True, text=True).stdout.splitlines()
    every = []
    for line in ps:
        f = line.split(None, 4)
        if len(f) < 5:
            continue
        pid, psr, pcpu = int(f[0]), int(f[1]), float(f[2])
        if pcpu > 5:
            every.append(line[:160])
            if psr in cores and pid not in mine and pid not in busy:
                busy[pid] = f'pid {pid} core {psr} {pcpu:.0f}% (lifetime): {f[4][:120]}'
    return list(busy.values()), every


def git_commit():
    try:
        return subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT,
                              capture_output=True, text=True).stdout.strip() or '?'
    except OSError:
        return '?'


def mem_available_gb():
    for line in open('/proc/meminfo'):
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) / 2**20
    return 0.0


def cmd_prelaunch(camp, a):
    head, lines = read_batch(a.batch)
    cores = parse_cores(head['cores'])
    busy, every = busy_on(cores, a.window)
    seg = next(s for s in camp.segments if s.label == head['arm'])
    _, run_need = running(a.state) if a.state else ({}, {})
    other = sum(v for k, v in run_need.items() if k != head['slot'])
    sizes = sorted({x['id'].split('/')[1] for x in lines})
    txt = (ARM_TEXT_TRAINING[seg.arm] if sizes == [TRAINING_SIZE] and seg.arm in ARM_TEXT_TRAINING
           else ARM_TEXT[seg.arm])
    prod = [s for s in sizes if s != TRAINING_SIZE]
    if TRAINING_SIZE in sizes:
        sel = ((f'instances 000-002 of {", ".join(prod)} selected the configurations and are '
                f'not evidence; ' if prod else '')
               + f'no instance of {TRAINING_SIZE} selected a configuration (they were frozen on '
                 f'the production sets), so every instance of {TRAINING_SIZE} is evidence')
    else:
        sel = 'instances 000-002 selected the configurations and are not evidence'
    start = ', '.join(sorted({rel(os.path.dirname(os.path.dirname(f)))
                              for x in lines for c in [cell_of(camp, x['id'])]
                              for f in camp.start_files(c)}))
    note = [
        f'=== timed batch {time.strftime("%Y-%m-%d %H:%M:%S")}, commit {git_commit()}, slot '
        f'{head["slot"]} (cores {head["cores"]}), arm {seg.label}, {len(lines)} cell(s)',
        f'1. Purpose: {txt["purpose"]}.',
        f'2. Expected: {txt["expected"]}.',
        '3. Contamination: new records only (the runner skips an existing record and never '
        f'overwrites); {sel}; the '
        'slot is checked below for every process above 5% of a core; cpu_share, affinity and '
        'load are in every record; an invalid record moves to invalid/ and its cell runs again.',
        f'4. Data: data/FAMILY/{",".join(sizes)}/desc/<combination>/instance_<i>; start '
        f'schedules from {start} (validated by the runner before the timed window); budget '
        f'{bnum(seg.budget)} s.',
        f'load average {" ".join(open("/proc/loadavg").read().split()[:3])}; memory available '
        f'{mem_available_gb():.1f} GB; need of this batch {head["need_gb"]} GB, of the batch '
        f'running on the other slot {other:g} GB, budget {camp.a.mem_budget:g} GB',
        'processes above 5% CPU, with the core they last ran on:',
    ] + [f'  {x}' for x in every] + ['cells:'] + [
        f'  {x["id"]} on cores {x["cores"]}: {x["cmd"]}' for x in lines]
    if busy:
        note += [f'{"FORCED (not a measurement)" if a.force else "REFUSED"}: cores '
                 f'{head["cores"]} are not idle:'] + [f'  {x}' for x in busy]
    with open(a.notes, 'a') as f:
        f.write('\n'.join(note) + '\n')
    print('\n'.join(note[-len(busy) - 1:]) if busy else
          f'slot {head["slot"]} idle; launch note appended to {a.notes}')
    if a.force and not camp.a.result_root:
        raise SystemExit('REFUSING: --force only with --result-root')
    return 1 if (busy and not a.force) else 0


def cmd_classify(camp, a):
    head, lines = read_batch(a.batch)
    seg = next(s for s in camp.segments if s.label == head['arm'])
    present = [x for x in lines if os.path.exists(x['record'])]
    cls = classify_paths(camp, seg.runner, [x['record'] for x in present])
    counts = {'valid': 0, 'pending': 0, 'invalid': 0, 'missing': 0}
    msgs = []
    for x in lines:
        if x not in present:
            counts['missing'] += 1
            msgs.append(f'  NO RECORD {x["id"]} (log {x["log"]}); not retried in this run')
            with open(os.path.join(a.state, 'skip.txt'), 'a') as f:
                f.write(x['id'] + '\n')
            continue
        ok, why = cls[x['record']]
        r = json.load(open(x['record']))
        tag = {True: 'valid', None: 'pending', False: 'invalid'}[ok]
        counts[tag] += 1
        msgs.append(f'  {tag:7s} {x["id"]} share {r.get("cpu_share")} ({why})')
        if ok is False:
            move_invalid(x['record'], why)
            msgs.append(f'          moved to {invalid_dir(x["record"])}/, runs again')
    line = (f'--- batch done {time.strftime("%Y-%m-%d %H:%M:%S")}, slot {head["slot"]}, '
            f'{seg.label}: valid {counts["valid"]}, pending {counts["pending"]}, invalid '
            f'{counts["invalid"]}, no record {counts["missing"]}')
    with open(a.notes, 'a') as f:
        f.write(line + '\n' + '\n'.join(msgs) + '\n')
    print(line)
    print('\n'.join(msgs))
    n_rec = counts['valid'] + counts['pending'] + counts['invalid']
    if n_rec and counts['invalid'] == n_rec:
        print(f'STOP: every record of the batch on slot {head["slot"]} is invalid; '
              f'something is on cores {head["cores"]}')
        return 3
    if not n_rec:
        return 4
    return 0


# ---------------------------------------------------------------------------
# reader of the trajectories
# ---------------------------------------------------------------------------

def record_rows(rec, runner, times=TIMES, origin='search'):
    """Incumbent at each time t (wall-clock seconds, inclusive) of one record.

    origin 'search': from the start of the search (memetic: the timed window;
    CP-SAT: the call to the solver, after the model build for full models;
    the LNS arm: its start, builds inside); 'start': memetic from the start of
    the solving process, CP-SAT from the start of the arm (model build
    included; the instance load and the hint validation before it are not
    timed by p85). Rows for t beyond the budget are left out.

    makespan = min(start schedule, search incumbent): the schedule handed to
    the search counts from time 0, as in the runners' own credit rule."""
    budget = float(rec['budget_s'])
    if runner == 'memetic':
        seeds = [s['makespan'] for s in rec.get('seeds', []) if s.get('makespan') is not None]
        start_ms = min(seeds) if seeds else rec.get('seed_best')
        pre = rec.get('search_start_wall_s', 0.0)
        final = rec['makespan']
    else:
        start_ms = rec.get('warm_makespan')
        pre = rec.get('build_s', 0.0) if CPSAT_KIND.get(rec.get('arm')) in ('ref', 'full') else 0.0
        final = rec['objective']
        pts = [(p[0], p[1]) for p in rec.get('anytime', [])]
    rows = []
    for t in times:
        lim = budget if origin == 'search' else budget + pre
        if t > lim + 1e-6:
            continue
        if runner == 'memetic':
            s_ms = incumbent_at(rec, t, 'wall', 'search' if origin == 'search' else 'process')
        else:
            tt = t if origin == 'search' else t - pre
            vals = [v for x, v in pts if x <= tt + 1e-9]
            s_ms = min(vals) if vals else None
        cand = [v for v in (start_ms, s_ms) if v is not None]
        rows.append(dict(t=int(t) if float(t).is_integer() else t, origin=origin, makespan=min(cand) if cand else None,
                         search_makespan=s_ms, start_makespan=start_ms,
                         final_makespan=final, pre_search_s=pre, budget_s=budget))
    return rows


def trajectory_rows(camp, times=TIMES, origin='search'):
    out = []
    for seg in camp.segments:
        cells = camp.cells(seg)
        recs = {c.id: camp.record(c) for c in cells}
        cls = classify_paths(camp, seg.runner, list(recs.values()))
        for c in cells:
            p = recs[c.id]
            if not os.path.exists(p):
                continue
            r = json.load(open(p))
            ok, why = cls[p]
            for row in record_rows(r, seg.runner, times, origin):
                row.update(arm=seg.arm, segment=seg.label,
                           solver='memetic' if seg.runner == 'memetic' else camp.a.cpsat_arm,
                           size=c.size, combination=c.combo, instance=c.idx,
                           valid=ok, reason=why, record=p)
                out.append(row)
    return out


COLUMNS = ('arm', 'segment', 'solver', 'size', 'combination', 'instance', 't', 'origin',
           'makespan', 'search_makespan', 'start_makespan', 'final_makespan',
           'pre_search_s', 'budget_s', 'valid', 'reason', 'record')


def cmd_table(camp, a):
    rows = trajectory_rows(camp, tuple(a.times), a.origin)
    w = csv.DictWriter(sys.stdout, fieldnames=COLUMNS)
    w.writeheader()
    for r in rows:
        w.writerow({k: r[k] for k in COLUMNS})
    return 0


# ---------------------------------------------------------------------------
# the pilot's selection
# ---------------------------------------------------------------------------

def select_pilot(tr_root, sizes, slots, bars, budget=600.0, workers=8, out='pilot',
                 arms=PILOT_ARMS, regimes=PILOT_REGIMES, instances=PILOT_INSTANCES,
                 times=(60, 300)):
    """Selection rule of report_cpsat_arms.md, section 4: v(a,c) the makespan
    returned at the budget (min of hint and best incumbent), v*(c) the best of
    the arms, Score(a) the mean of 100 (v - v*) / v* over the cells; the lowest
    score wins, and an arm within 0.10 points of it that is simpler wins.
    Returns a dict; 'refused' lists the reasons when the records do not allow
    the choice (missing, invalid, pending, or another protocol)."""
    cells = [(s, r, i) for s in sizes for r in regimes for i in instances]
    paths = {(a, c): os.path.join(tr_root, f'{c[0]}_cparms', out, a, c[1], f'instance_{c[2]:03d}.json')
             for a in arms for c in cells}
    refused = [f'missing {p}' for p in paths.values() if not os.path.exists(p)]
    if refused:
        return {'refused': refused}
    cls = T.classify(list(paths.values()), slots=slots, thresh=bars, key=T.key_cparms)
    recs = {k: json.load(open(p)) for k, p in paths.items()}
    for k, p in paths.items():
        ok, why = cls.get(p, (False, 'not a p85 record'))
        if ok is not True:
            refused.append(f'{"pending" if ok is None else "invalid"} {p}: {why}')
        r = recs[k]
        if abs(float(r['budget_s']) - budget) > 1e-9 or r.get('workers') != workers:
            refused.append(f'protocol {p}: budget {r["budget_s"]}, workers {r.get("workers")}')
        if r.get('development_run'):
            refused.append(f'development run {p}')
    for s in sizes:
        hs = {recs[a, c].get('hint_source') for a in arms for c in cells if c[0] == s}
        if len(hs) > 1:
            refused.append(f'{s}: hints from {sorted(map(str, hs))}')
    if refused:
        return {'refused': refused}

    def scores(val):
        sc = {}
        for a in arms:
            sc[a] = {}
            for part, keep in [('all', lambda c: True)] + \
                    [(s, lambda c, s=s: c[0] == s) for s in sizes] + \
                    [('setup', lambda c: c[1] in SETUP), ('nosetup', lambda c: c[1] not in SETUP)]:
                cs = [c for c in cells if keep(c)]
                if cs:
                    sc[a][part] = sum(100.0 * (val[a, c] - min(val[b, c] for b in arms))
                                      / min(val[b, c] for b in arms) for c in cs) / len(cs)
        return sc

    v = {(a, c): float(recs[a, c]['objective']) for a in arms for c in cells}
    sc = scores(v)
    best = min(sc[a]['all'] for a in arms)
    near = [a for a in arms if sc[a]['all'] <= best + TIE_PP + 1e-9]
    chosen = near[0]                      # arms are listed simplest first
    at = {}
    for t in times:
        vt = {}
        for a in arms:
            for c in cells:
                row = [x for x in record_rows(recs[a, c], 'cpsat', (t,)) if x['t'] == t]
                vt[a, c] = row[0]['makespan'] if row else float('nan')
        at[t] = scores(vt)
    return {'chosen': chosen, 'scores': sc, 'best_score': best, 'within_tie': near,
            'at': at, 'cells': len(cells), 'values': v, 'refused': []}


def cmd_select(camp, a):
    slots = [cs for _, cs, _ in camp.slots]
    res = select_pilot(camp.tr_root(), a.pilot_sizes, slots, a.bars, budget=a.pilot_budget,
                       workers=a.pilot_workers, out=a.pilot_out)
    if res['refused']:
        print(f'REFUSING the selection: {len(res["refused"])} problem(s)')
        for r in res['refused'][:40]:
            print(f'  {r}')
        return 1
    parts = ['all'] + list(a.pilot_sizes) + ['setup', 'nosetup']
    print(f'pilot selection over {res["cells"]} cells ({", ".join(a.pilot_sizes)}; '
          f'{" ".join(PILOT_REGIMES)}; instances 000-002); score = mean % above the best arm')
    print(f'{"arm":30s}' + ''.join(f'{p:>10s}' for p in parts)
          + ''.join(f'{"all@" + str(t) + "s":>10s}' for t in res['at']))
    for arm in PILOT_ARMS:
        s = res['scores'][arm]
        print(f'{arm:30s}' + ''.join(f'{s[p]:10.3f}' for p in parts)
              + ''.join(f'{res["at"][t][arm]["all"]:10.3f}' for t in res['at']))
    print(f'lowest score {res["best_score"]:.3f}; within {TIE_PP} points: '
          f'{", ".join(res["within_tie"])}')
    print(f'CHOSEN: {res["chosen"]} (the scores at 60 s and 300 s and per part are reported '
          f'only; they do not change the choice)')
    return 0


def cmd_bars(camp, a):
    paths = []
    for f in a.folders:
        paths += glob.glob(os.path.join(f, '*', '*', '*.json'))
    bars = T.type_bars(paths, key=T.key_cparms)
    for k, v in sorted(bars.items()):
        print(f'{k:60s} {v:.4f}')
    if a.write:
        old = json.load(open(a.write)) if os.path.exists(a.write) else {}
        clash = [k for k in bars if k in old and abs(old[k] - bars[k]) > 1e-12]
        if clash:
            print(f'REFUSING: frozen bars would change: {clash}')
            return 1
        old.update({k: v for k, v in bars.items() if k not in old})
        tmp = a.write + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(old, f, indent=1, sort_keys=True)
        os.replace(tmp, a.write)
        print(f'written {a.write} ({len(old)} types)')
    return 0


# ---------------------------------------------------------------------------

def build_parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--arms', default='memetic_rule cpsat_rule')
    common.add_argument('--budget', type=float, default=1800.0)
    common.add_argument('--instances', default=None,
                        help='for every size; default: each size\'s evidence instances '
                             '(3-29 at 50x25n30 and 80x25n30, 0-29 at 10x25)')
    common.add_argument('--sizes', nargs='+', default=list(SIZES),
                        help=f'default {" ".join(SIZES)}; the training size is {TRAINING_SIZE}')
    common.add_argument('--combos', nargs='+', default=list(COMBOS))
    common.add_argument('--slots', nargs='+', default=['8-15', '16-23'])
    common.add_argument('--cpsat-arm', default='ref_fullhint',
                        help='the arm the pilot selects (ref_fullhint until then)')
    common.add_argument('--policy-seed', type=int, default=None)
    common.add_argument('--rule-cache', default='or_solution/FAMILY/warmstart_cache_xmask_{size}')
    common.add_argument('--policy-cache',
                        default='or_solution/FAMILY/pactwarm_cache_xmask_s{seed}_{size}')
    common.add_argument('--result-root', default='',
                        help='put both record trees under this folder (smoke runs)')
    common.add_argument('--workers', type=int, default=8,
                        help='CP-SAT workers; 8 in the campaign, fewer only for smoke runs')
    common.add_argument('--memetic-seed', type=int, default=1)
    common.add_argument('--cpsat-seed', type=int, default=1)
    common.add_argument('--mem-budget', type=float, default=40.0)
    common.add_argument('--mem-memetic', type=float, default=0.75,
                        help='GB per memetic process (bound from the smoke runs)')
    common.add_argument('--bars', default='test_results/FAMILY/timed_campaign_share_bars.json',
                        help='frozen CPU-share bars of the CP-SAT cell types (read only)')
    common.add_argument('--max-invalid', type=int, default=2)
    common.add_argument('--own-state', default='train_log/PACT_MEMORY_STATE.txt',
                        help='PACT_MEMORY_STATE of the CP-SAT commands')
    common.add_argument('--peer-state', default='',
                        help="PACT_PEER_MEMORY_STATE of the CP-SAT commands ('none': no hold)")
    common.add_argument('--control-core', type=int, default=None)
    common.add_argument('--py', default=os.path.expanduser('~/miniconda3/envs/fjsp/bin/python'))
    common.add_argument('--logdir', default='train_log')
    common.add_argument('--state', default='')
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('plan', parents=[common])
    p.add_argument('--strict', action='store_true')
    p.add_argument('--all-commands', action='store_true')
    p.add_argument('--layouts', nargs='*', default=list(LAYOUTS),
                   help="other slot layouts to compare, each 'coresA coresB'")
    p = sub.add_parser('next', parents=[common])
    p.add_argument('--slot', required=True)
    p = sub.add_parser('prelaunch', parents=[common])
    p.add_argument('--batch', required=True)
    p.add_argument('--notes', required=True)
    p.add_argument('--window', type=float, default=2.0)
    p.add_argument('--force', action='store_true',
                   help='do not refuse a busy slot (only with --result-root; not a measurement)')
    p = sub.add_parser('classify', parents=[common])
    p.add_argument('--batch', required=True)
    p.add_argument('--notes', required=True)
    p = sub.add_parser('table', parents=[common])
    p.add_argument('--times', nargs='+', type=float, default=list(TIMES))
    p.add_argument('--origin', choices=('search', 'start'), default='search')
    p = sub.add_parser('select', parents=[common])
    p.add_argument('--pilot-sizes', nargs='+', default=list(SIZES))
    p.add_argument('--pilot-budget', type=float, default=600.0)
    p.add_argument('--pilot-workers', type=int, default=8)
    p.add_argument('--pilot-out', default='pilot')
    p = sub.add_parser('bars', parents=[common])
    p.add_argument('folders', nargs='+', help='<root>/<size>_cparms/<out> folders')
    p.add_argument('--write', default=None)
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    os.chdir(ROOT)
    camp = Campaign(a)
    return {'plan': cmd_plan, 'next': cmd_next, 'prelaunch': cmd_prelaunch,
            'classify': cmd_classify, 'table': cmd_table, 'select': cmd_select,
            'bars': cmd_bars}[a.cmd](camp, a)


if __name__ == '__main__':
    sys.exit(main())
