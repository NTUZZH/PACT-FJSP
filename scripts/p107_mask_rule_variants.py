r"""P107: three admissibility rules compared in one environment, without training.

Subcommands plan, run and infer. Evaluation only, CPU only, exact mask options (first_mask
and mask_ties exact). No solver runs, nothing runs on the GPU, and no network is loaded.

The environment of the paper (constraint_family/fjsp_env_family.py, FJSPEnvFamily) builds
every schedule of the policy and of the dispatching rules. At every decision it admits only
the pairs of an operation and a machine that its admissibility rule allows. This script
compares three such rules in that environment for every schedule that needs no training:
the four classical dispatching rules FIFO, MOR, SPT and MWKR, the random control of
scripts/p101_mask_controls.py and the replay of the reference schedules of
scripts/p105_reference_replay.py.

THE THREE RULES
  In one state (the candidate of every unfinished job, the free time and the last routing
  class of every machine) the environment gives every compatible pair (j, k) of a candidate
  and a machine
    block(j, k) = push(max(job ready, machine free), setup + processing, outages of k),
    start(j, k) = block(j, k) + setup(last class of k, class of the candidate),
  the attached-setup convention of FJSPEnvFamily._compute_pair_free_time: the block of setup
  and processing is pushed past every outage it overlaps, and processing starts after the
  setup. Without setups block = start.
  nondelay  the rule of the paper (mask_mode='nondelay', mask_ties='exact'): (j, k) is
            admissible when start(j, k) is within half an hour (half a time quantum) of the
            decision instant t = min start over the compatible pairs.
  block     the classical non-delay rule on block starts: (j, k) is admissible when
            block(j, k) is within half an hour of the smallest block start over the
            compatible pairs. BlockStartEnv below is a subclass of FJSPEnvFamily that
            overrides _compute_unavailable_pairs only; it computes block(j, k) with the
            lines of _compute_pair_free_time that precede the addition of the setup.
  active    the Giffler-Thompson condition the environment implements (mask_mode=
            'active'): (j, k) is admissible when start(j, k) is earlier than C*, the
            earliest completion start + processing over the compatible pairs, by more than
            half an hour.
  Under every rule the chosen pair starts processing at start(j, k) (the step of the
  environment), so under block and active a machine may wait for a later job. Without
  setups block admits exactly the pairs nondelay admits. nondelay is contained in active.

LAUNCH NOTE (the four questions, answered before the run)

1. Purpose. The non-delay mask of the paper compares processing starts. With attached setups
   the classical non-delay definition compares the starts of the blocks, and a wider rule
   (active) lets a machine wait for a later job. Do these rules give other or shorter
   schedules for everything that needs no training (the four rules, the random control, the
   replay of reference schedules)? The numbers land in the supplement section on the
   controls for the mask and in one sentence of the main text. The learned policy under the
   active mask is the training arm xmask-active, outside this script.
2. Expected result. Unknown. If block or active gives shorter rule schedules under setups,
   the paper reports by how much and states that the processing-start rule costs the rules
   that much. If the differences lie within their intervals or favour nondelay, the paper
   states that the rule costs the dispatching rules nothing measurable.
3. Contamination. Instances, descriptors, reference records, the stored rule arrays and the
   cells of p101 and p105 are read only. Cases go to a new folder,
   test_results/FAMILY/maskrule_p107 (--root), one npz and one JSON per case, the npz first
   and the JSON last, each through a temporary file and a rename, so an interrupted run
   leaves no case that looks finished. A stored case of another design or of another
   reference record (sha256) is refused, never overwritten (unless --force). Rollouts are
   deterministic and run on one thread, so a case is not a timing measurement and the run
   may share cores; the recorded times are planning figures.
4. Data accuracy. Two checks are built in and every case must pass them, or run refuses
   (and infer repeats the first two): (a) under nondelay the rollouts of FIFO, MOR, SPT and
   MWKR reproduce the stored exact-mask rule arrays of the instance and combination bit
   for bit; (b) without setups (N, L, W, LW) the block rule admits exactly the pairs of the
   nondelay rule at every decision of every rule rollout, and every schedule of the case
   (four rules, ten random rollouts, every replay) is identical under the two rules. Every
   schedule under every rule passes the validator of p101 (reference_sim.validate_schedule
   and the makespan against the last completion) and has one decision per operation. Under
   nondelay, the admissible set that this script computes from the state equals the
   environment's own mask at every decision of every rule rollout. Where the cells exist,
   the nondelay random control equals the cell of p101 and the nondelay replay equals the
   case of p105 with the same reference record.

SETS (the instances in the row order of the stored rule arrays)
  pub10x5   public benchmark, SD2 test set at 10x5, 100 instances; replay of the p105
            references cpsat150 (CP-SAT, 150 s) on all 100.
  10x25     factory benchmark at the training size, the 100 test instances with stored rule
            arrays; replay of memetic_rule600 on instances 000-029 (the p105 set).
  50x25n30, 80x25n30  production batches, instances 003-029; replay of memetic_rule600 (and of
            cpsat_rule600 when requested with --refs and its configuration is determined).
  Only references that p105 counts are replayed (P105.reference_states; a pending or
  invalid record counts as absent).

PER CASE (one instance, one combination) AND RULE X in nondelay, block, active
  rules     makespan of FIFO, MOR, SPT and MWKR restricted to the pairs X admits
            (common_utils.masked_rule_action after np.random.seed(50), as p101), and their
            per-instance best (best rule);
  random    ten rollouts that draw one admissible pair uniformly at every decision, draw k
            with default_rng(SeedSequence([20261113, code, instance index, k])), code 10,
            50, 80 (the stream of p101) and 105 for the public set; the same seeds under the
            three rules; their mean (rand_mean) and their shortest (rand_best);
  replay    per reference R, the replay of p105 by orders A, B and C under X and the
            shortest of the three; per order the decisions off R and whether order A
            reproduces R;
  pairs     along the SPT rollout under X: the number of admissible pairs before every
            decision, the decisions with exactly one admissible pair, and the decisions at
            which the admissible set differs from the nondelay set of the same state;
  same      for X in block, active, which schedules equal their nondelay counterpart (same
            machine and same completion for every operation).

OUTCOMES (infer), per set and combination, pooled over the eight combinations (All) and
pooled over the four combinations with setups S, LS, SW, LSW (Setup); pooled, a drawn
instance brings its combinations
  diff      for X in block, active and each quantity q in best rule, FIFO, MOR, SPT, MWKR,
            rand_mean, rand_best and the replay of each reference:
            100 (q_X - q_nondelay) / q_nondelay per case (negative = X gives the shorter
            schedule), mean and 95% percentile bootstrap over instances
            (stats_family.instance_boot, 20,000 resamples, fresh default_rng(7), the
            settings of p105);
  gap       for every rule X: 100 (replay_X - reference) / reference per case, mean and
            interval (the upper bound on what restricting the schedules to rule X costs);
  exact     for every rule X: % of references that order A reproduces exactly;
  pairs     for every rule X along the SPT rollout: mean admissible pairs per decision and %
            of decisions with one admissible pair, over all decisions of all cases of the
            row; % of decisions at which the set differs from the nondelay set;
  blockdiff number of cases in which block differs from nondelay in any schedule (rule,
            random or replay), and per kind.
  Outputs: results/mask_rule_variants.json, results/mask_rule_variants.md and
  manuscript/macros_maskrule.tex (macro prefix Mrul), or the same paths under --out.

FOLDERS
  <root>/<set>/<combination>/<instance>.json, .npz   one case; npz per rule X:
      X_rule_amch [4, N], X_rule_ct [4, N] (FIFO, MOR, SPT, MWKR), X_random_ms [10],
      X_spt_pairs [N] (admissible pairs before every decision of SPT)

USAGE (repository root; CPU only; one thread)
  python scripts/p107_mask_rule_variants.py plan  [--parts K]
  python scripts/p107_mask_rule_variants.py run   [--sets ...] [--combos ...] [--n k]
                                                  [--part i --parts K] [--budget_s S]
  python scripts/p107_mask_rule_variants.py infer [--sets ...] [--partial --out <folder>]
  Without --partial infer needs every set, combination, planned instance and valid
  reference, and writes the repository outputs; --partial reports the cases present, under
  --out only. --budget_s stops run before a case whose estimated cost would end the command
  after S seconds; the cases make the run resumable.
  Smoke: run --root <folder> --n 3 --combos N,S,LSW, then infer --partial --root <folder>
  --out <folder> --n 3 --combos N,S,LSW.
"""
import argparse
import json
import os
import platform
import re
import sys
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(ROOT_DIR, 'scripts')
sys.path.insert(0, ROOT_DIR)
if SCRIPTS_DIR not in sys.path:
    sys.path.append(SCRIPTS_DIR)

import arm_registry as REG  # noqa: E402,F401  (removes --pact-* options from sys.argv)

_ARGV = sys.argv[1:]
# params parses argv at import time and EnvState reads the device then: CPU only
sys.argv = sys.argv[:1] + ['--device', 'cpu']

import numpy as np  # noqa: E402

import p101_mask_controls as P101  # noqa: E402
import p105_reference_replay as P105  # noqa: E402
import stats_family as SF  # noqa: E402
from constraint_family.fjsp_env_family import FJSPEnvFamily  # noqa: E402

# ---------------------------------------------------------------------------
# design constants
# ---------------------------------------------------------------------------

R8 = P101.R8
SETUP_COMBOS = P101.SETUP_COMBOS                     # S, LS, SW, LSW
NOSETUP_COMBOS = tuple(c for c in R8 if c not in SETUP_COMBOS)   # N, L, W, LW
RULES = P101.RULES                                   # FIFO, MOR, SPT, MWKR
ADM = ('nondelay', 'block', 'active')
ALT = ('block', 'active')
ORDERS = P105.ORDERS
N_DRAWS = P101.N_DRAWS
RULE_SEED = P101.RULE_SEED
RANDOM_SEED = P101.RANDOM_SEED
EPS = 1e-6
B = SF.B_DEFAULT                                     # 20,000 bootstrap resamples
CI_SEED = 7
DEFAULT_ROOT = 'test_results/FAMILY/maskrule_p107'
OUT_FILES = {'json': 'results/mask_rule_variants.json', 'md': 'results/mask_rule_variants.md',
             'macros': 'manuscript/macros_maskrule.tex'}
GENERATOR = 'scripts/p107_mask_rule_variants.py infer'
PREFIX = 'Mrul'
VERSION = 1
SETS = {
    'pub10x5': dict(code=105, word='Pub',
                    text='public benchmark, SD2 test set 10x5, 100 instances'),
    '10x25': dict(code=10, word='Ten',
                  text='factory benchmark, 10 modules, the 100 test instances'),
    '50x25n30': dict(code=50, word='Fifty',
                     text='factory benchmark, 50 modules, instances 003-029'),
    '80x25n30': dict(code=80, word='Eighty',
                     text='factory benchmark, 80 modules, instances 003-029'),
}
DEFAULT_REFS = ('cpsat150', 'memetic_rule600')
REF_WORD = {'cpsat150': 'Cp', 'memetic_rule600': 'Mem', 'cpsat_rule600': 'Cp'}
ADM_WORD = {'nondelay': 'Nd', 'block': 'Blk', 'active': 'Act'}
QUANTS = ('best_rule',) + RULES + ('rand_mean', 'rand_best')
Q_WORD = {'best_rule': 'BestRule', 'FIFO': 'Fifo', 'MOR': 'Mor', 'SPT': 'Spt', 'MWKR': 'Mwkr',
          'rand_mean': 'RandMean', 'rand_best': 'RandBest'}
Q_TEXT = {'best_rule': 'best classical rule (per-instance best of FIFO, MOR, SPT, MWKR)',
          'FIFO': 'FIFO', 'MOR': 'MOR', 'SPT': 'SPT', 'MWKR': 'MWKR',
          'rand_mean': 'mean of the ten random rollouts',
          'rand_best': 'shortest of the ten random rollouts'}
ADM_TEXT = {
    'nondelay': 'admissible when the processing start is within half an hour of the earliest '
                'processing start over the compatible pairs (the rule of the paper)',
    'block': 'admissible when the block start (after the push past outages, before the '
             'setup) is within half an hour of the earliest block start over the compatible '
             'pairs (classical non-delay on block starts)',
    'active': 'admissible when the processing start is earlier than the earliest completion '
              'over the compatible pairs by more than half an hour (Giffler-Thompson)'}
# planning cost, process CPU seconds per case (the three rules, each with four rule
# rollouts and ten random rollouts, plus three replays per reference where one exists),
# from the smoke cases (2026-11-13, one shared core): N, S and LSW of three instances of the
# public set and of 10x25, and of one instance of 50x25n30 and of 80x25n30 (LSW the slowest)
COST_S = {'pub10x5': 0.7, '10x25': 4.5, '50x25n30': 62.0, '80x25n30': 160.0}
ROWS_EXTRA = ('All', 'Setup')


def stop(msg):
    """A precondition of plan or run failed."""
    raise SystemExit(f'REFUSING: {msg}')


Refused = P101.Refused
refuse = P101.refuse          # infer: raises Refused('refused: ...'), outputs removed


# ---------------------------------------------------------------------------
# the block rule
# ---------------------------------------------------------------------------

class BlockStartEnv(FJSPEnvFamily):
    """FJSPEnvFamily with the classical non-delay rule on block starts.

    Only _compute_unavailable_pairs differs. The environment calls it after every step and
    for the first decision (_first_mask_tables, first_mask='exact'), each time after
    pair_free_time and next_schedule_time of the same state. The block start of every pair
    is computed by the lines of _compute_pair_free_time that precede the addition of the
    setup: block = max(job ready, machine free), pushed past every outage that the block of
    setup and processing would overlap (the same setup from the machine's last class, the
    same durations, the same half-quantum guard). A pair is unavailable when its block
    start exceeds the smallest block start over the compatible pairs by more than half a
    time quantum (1 h on the true axis, self._slope on the normalized one), the tolerance
    of the environment's own comparisons. The smallest block start is attained by some
    compatible pair, so the admissible set is never empty while an operation remains. The
    chosen pair starts processing at block + setup, the transition the environment's step
    applies to every pair."""

    def _compute_unavailable_pairs(self):
        candFT = np.expand_dims(self.candidate_free_time, axis=2)   # [E,J,1]
        mchFT = np.expand_dims(self.mch_free_time, axis=1)          # [E,1,M]
        s = self._setup_into_candidates()                            # [E,J,M]
        block = np.maximum(candFT, mchFT)
        if self.n_win:
            dur = s + self.candidate_pt                              # [E,J,M]
            block = self._push_past_outages(
                block, dur, self.win_s_norm[:, None, :, :],
                self.win_e_norm[:, None, :, :], eps=0.5 * self._slope)
        open_ = np.where(self.candidate_process_relation, np.inf, block)
        bstar = open_.reshape(self.number_of_envs, -1).min(axis=1)   # [E]
        return block > bstar[:, None, None] + 0.5 * self._slope


def make_env(jl, pt, meta, descs, adm):
    """A batch of len(descs) copies of one instance under admissibility rule adm. nondelay
    and active are P101.make_env (the construction of eval_family.py with the exact mask
    options); block is the same construction with BlockStartEnv."""
    if adm in ('nondelay', 'active'):
        return P101.make_env(jl, pt, meta, descs, mask_mode=adm)
    if adm != 'block':
        raise ValueError(f'unknown admissibility rule {adm}')
    E = len(descs)
    env = BlockStartEnv(jl.shape[0], pt.shape[1], family_mode=True, blind_channels=(),
                        mask_mode='nondelay', pair_extra='none', **P101.MASK)
    env.delta_mode = 'min'
    env.reward_delta_env = True
    typed = meta.get('op_type') is not None
    env.set_initial_data([jl] * E, [pt] * E, descriptor_list=list(descs),
                         op_type_list=[meta['op_type']] * E if typed else None,
                         mch_type_list=[meta['mch_type']] * E if typed else None)
    for k, v in P101.MASK.items():
        if env.mask_setting(k) != v:
            raise RuntimeError(f'the environment reads {k}={env.mask_setting(k)}')
    if not env._first_mask:
        raise RuntimeError('the exact first-decision mask was not applied')
    return env


def admitted(env):
    """[E, J, M] bool: the pairs the environment's mask admits now."""
    return ~np.asarray(env.dynamic_pair_mask, dtype=bool)


def nondelay_admitted(env):
    """[E, J, M] bool: the pairs the nondelay rule admits in the environment's current state
    (the tie-tolerant comparison of FJSPEnvFamily._compute_unavailable_pairs with
    mask_ties='exact'), whatever rule the environment applies."""
    compat = ~np.asarray(env.candidate_process_relation, dtype=bool)
    return compat & (env.pair_free_time
                     <= env.next_schedule_time[:, None, None] + 0.5 * env._slope)


# ---------------------------------------------------------------------------
# rollouts (P101.rule_rollout, P101.random_rollouts and P105.replay with the environment
# of the rule; under nondelay each returns what the original returns)
# ---------------------------------------------------------------------------

def rule_rollout(rule, jl, pt, meta, desc, adm):
    """One rollout of a classical rule restricted to the pairs rule adm admits, with the
    admissible-pair count before every decision and the number of decisions at which the
    admissible set differs from the nondelay set of the same state."""
    from common_utils import masked_rule_action
    M = pt.shape[1]
    env = make_env(jl, pt, meta, [desc], adm)
    amch = np.full(pt.shape[0], -1, dtype=int)
    counts, differ = [], 0
    np.random.seed(RULE_SEED)
    while not env.done().all():
        legal = admitted(env)[0]
        c = int(legal.sum())
        if c < 1:
            raise RuntimeError('the mask admits no pair')
        counts.append(c)
        if not np.array_equal(legal, nondelay_admitted(env)[0]):
            differ += 1
        a = int(masked_rule_action(rule, env))
        amch[env.candidate[0, a // M]] = a % M
        env.step(np.array([a]))
    return dict(makespan=float(env.current_makespan[0]), amch=amch,
                op_ct=env.true_op_ct[0].copy(), counts=np.asarray(counts, dtype=np.int32),
                differ=int(differ))


def random_rollouts(jl, pt, meta, desc, seeds, adm):
    """len(seeds) rollouts of the random control in one batch under rule adm: at every
    decision each copy draws one admissible pair uniformly with its own generator. Returns
    makespans [D], machine assignments [D, N] and completion times [D, N]."""
    M = pt.shape[1]
    D = len(seeds)
    env = make_env(jl, pt, meta, [desc] * D, adm)
    rngs = [np.random.default_rng(np.random.SeedSequence(s)) for s in seeds]
    amch = np.full((D, pt.shape[0]), -1, dtype=int)
    rows = np.arange(D)
    while not env.done().all():
        legal = admitted(env).reshape(D, -1)
        acts = np.empty(D, dtype=np.int64)
        for k in range(D):
            idx = np.flatnonzero(legal[k])
            if idx.size == 0:
                raise RuntimeError('the mask admits no pair')
            acts[k] = idx[int(rngs[k].integers(idx.size))]
        amch[rows, env.candidate[rows, acts // M]] = acts % M
        env.step(acts)
    return (env.current_makespan.astype(float).copy(), amch,
            env.true_op_ct.astype(float).copy())


def replay(jl, pt, meta, desc, ref, order, adm):
    """P105.replay under rule adm: at every decision the admissible pair that the order
    picks (P105.choose), with the completion of a pair its processing start in hours (read
    from pair_free_time) plus its processing time."""
    N, M = pt.shape
    env = make_env(jl, pt, meta, [desc], adm)
    seq = P105.machine_sequences(ref['amch'], ref['st'], M)
    ptr = [0] * M
    done = np.zeros(N, dtype=bool)
    amch = np.full(N, -1, dtype=int)
    off, steps = 0, 0
    while not env.done().all():
        legal = admitted(env)[0]
        js, ms = np.nonzero(legal)
        if js.size == 0:
            raise RuntimeError('the mask admits no pair')
        ops = env.candidate[0, js]
        start = np.rint(np.asarray(env.pair_free_time[0], dtype=float)[js, ms] / env._slope)
        compl = start + pt[ops, ms]
        k = P105.choose(order, js, ms, ops, compl, ref)
        j, m, o = int(js[k]), int(ms[k]), int(ops[k])
        mr = int(ref['amch'][o])
        while ptr[mr] < len(seq[mr]) and done[seq[mr][ptr[mr]]]:
            ptr[mr] += 1
        if m != mr or seq[mr][ptr[mr]] != o:
            off += 1
        done[o] = True
        amch[o] = m
        env.step(np.array([j * M + m]))
        steps += 1
    op_ct = np.asarray(env.true_op_ct[0], dtype=float).copy()
    msp = float(env.current_makespan[0])
    op_st = op_ct - pt[np.arange(N), amch]
    exact = bool(np.array_equal(amch, ref['amch']) and np.allclose(op_st, ref['st'], atol=EPS))
    return dict(makespan=msp, amch=amch, op_ct=op_ct, off=off, exact=exact, decisions=steps)


# ---------------------------------------------------------------------------
# one case
# ---------------------------------------------------------------------------

def check_schedule(where, jl, pt, desc, amch, op_ct, makespan, decisions=None):
    v = P101.validate(jl, pt, desc, amch, op_ct, makespan)
    if v:
        stop(f'{where} is infeasible: {v[:2]}')
    if decisions is not None and decisions != pt.shape[0]:
        stop(f'{where} took {decisions} decisions for {pt.shape[0]} operations')


def evaluate_rule(adm, jl, pt, meta, desc, seeds, refs, where='case'):
    """Every schedule of one case under rule adm. refs: {arm: dict(amch, st, ct)}. Returns
    (record, schedules {key: (amch, op_ct)}, admissible-pair counts along SPT)."""
    t0, c0 = time.perf_counter(), time.process_time()
    rec = dict(rules={}, replay={})
    sched = {}
    spt_counts = None
    for r in RULES:
        x = rule_rollout(r, jl, pt, meta, desc, adm)
        check_schedule(f'{where}: {r} under {adm}', jl, pt, desc, x['amch'], x['op_ct'],
                       x['makespan'], len(x['counts']))
        cnt = x['counts']
        rec['rules'][r] = dict(makespan=x['makespan'], decisions=int(len(cnt)),
                               one_pair=int((cnt == 1).sum()), pairs_sum=int(cnt.sum()),
                               differ_nondelay=int(x['differ']))
        sched[('rule', r)] = (x['amch'], x['op_ct'])
        if r == 'SPT':
            spt_counts = cnt
    rec['best_rule'] = float(min(rec['rules'][r]['makespan'] for r in RULES))
    rec['best_rule_name'] = min(RULES, key=lambda r: (rec['rules'][r]['makespan'],
                                                      RULES.index(r)))
    ms, amch, ct = random_rollouts(jl, pt, meta, desc, seeds, adm)
    for k in range(len(seeds)):
        check_schedule(f'{where}: random rollout {k} under {adm}', jl, pt, desc, amch[k],
                       ct[k], ms[k])
        sched[('random', k)] = (amch[k], ct[k])
    rec['random'] = dict(makespan=[float(v) for v in ms], mean=float(ms.mean()),
                         best=float(ms.min()))
    for arm, ref in refs.items():
        R = {}
        for od in ORDERS:
            x = replay(jl, pt, meta, desc, ref, od, adm)
            check_schedule(f'{where}: replay of {arm} by order {od} under {adm}', jl, pt,
                           desc, x['amch'], x['op_ct'], x['makespan'], x['decisions'])
            R[od] = dict(makespan=x['makespan'], off=int(x['off']), exact=bool(x['exact']))
            sched[('replay', arm, od)] = (x['amch'], x['op_ct'])
        best = min(R[od]['makespan'] for od in ORDERS)
        R['best'] = best
        R['best_order'] = [od for od in ORDERS if R[od]['makespan'] == best][0]
        rec['replay'][arm] = R
    rec['cpu_s'] = time.process_time() - c0
    rec['wall_s'] = time.perf_counter() - t0
    return rec, sched, spt_counts


def key_text(k):
    return '/'.join(str(x) for x in k)


def schedule_differences(sx, s0):
    """Keys of the schedules of sx that differ from s0 (another machine or another
    completion for some operation)."""
    if set(sx) != set(s0):
        raise ValueError('the two rules hold different schedule sets')
    return [k for k in s0 if not (np.array_equal(sx[k][0], s0[k][0])
                                  and np.array_equal(sx[k][1], s0[k][1]))]


def has_setups(desc, combo):
    """Whether the case carries sequence-dependent setups; the descriptor and the name of
    the combination must agree."""
    s = bool(desc.has_setup and desc.setup is not None)
    if s != ('S' in combo):
        stop(f'combination {combo}: the descriptor says setups {"on" if s else "off"}')
    return s


def evaluate_case(jl, pt, meta, desc, combo, seeds, refs, stored, where='case'):
    """The three rules on one case and the two built-in checks. stored: {rule: makespan}
    of the stored exact-mask arrays (the nondelay rollouts must reproduce them). Returns
    (record fields, arrays for the npz)."""
    per, sch, cnt = {}, {}, {}
    for adm in ADM:
        per[adm], sch[adm], cnt[adm] = evaluate_rule(adm, jl, pt, meta, desc, seeds, refs,
                                                     where)
    nd = per['nondelay']
    # check (a): the nondelay rollouts reproduce the stored arrays
    for r in RULES:
        if r not in stored:
            stop(f'{where}: no stored exact-mask array of {r}')
        if nd['rules'][r]['makespan'] != float(stored[r]):
            stop(f'{where}: the {r} rollout under nondelay ends at '
                 f'{nd["rules"][r]["makespan"]}, the stored array holds {stored[r]}; the '
                 f'environments diverge')
    # the set computed from the state equals the environment's own nondelay mask
    bad = [r for r in RULES if nd['rules'][r]['differ_nondelay']]
    if bad:
        stop(f'{where}: under nondelay the admissible set computed from the state differs '
             f'from the environment\'s mask ({bad})')
    same = {}
    for x in ALT:
        diff = schedule_differences(sch[x], sch['nondelay'])
        same[x] = dict(
            any=bool(diff), rules=[k[1] for k in diff if k[0] == 'rule'],
            random=int(sum(k[0] == 'random' for k in diff)),
            replay=sorted(f'{k[1]}/{k[2]}' for k in diff if k[0] == 'replay'),
            differ_decisions_spt=int(per[x]['rules']['SPT']['differ_nondelay']))
    setups = has_setups(desc, combo)
    # check (b): without setups the block rule is the nondelay rule
    if not setups:
        dec = {r: per['block']['rules'][r]['differ_nondelay'] for r in RULES}
        if any(dec.values()) or same['block']['any']:
            stop(f'{where}: without setups the block rule differs from nondelay (decisions '
                 f'with another admissible set per rule {dec}; schedules that differ: '
                 f'{same["block"]}); the block start is not computed as the environment '
                 f'computes it')
    checks = dict(
        stored_rules='the four nondelay rule rollouts equal the stored arrays',
        nondelay_mask='the admissible set computed from the state equals the environment\'s '
                      'mask at every decision of the four nondelay rule rollouts',
        block_vs_nondelay=('no setups: the block rule admits the nondelay pairs at every '
                           'decision of the four rule rollouts and every schedule is identical'
                           if not setups else
                           'setups: block may differ, not required to equal nondelay'))
    arrays = {}
    for adm in ADM:
        arrays[f'{adm}_rule_amch'] = np.stack([sch[adm][('rule', r)][0]
                                               for r in RULES]).astype(np.int16)
        arrays[f'{adm}_rule_ct'] = np.stack([sch[adm][('rule', r)][1] for r in RULES])
        arrays[f'{adm}_random_ms'] = np.asarray(per[adm]['random']['makespan'], dtype=float)
        arrays[f'{adm}_spt_pairs'] = np.asarray(cnt[adm], dtype=np.int32)
    rec = dict(adm=per, same_as_nondelay=same, setups=setups, checks=checks)
    return rec, arrays


# ---------------------------------------------------------------------------
# instances, stored arrays and the cells of p101 and p105
# ---------------------------------------------------------------------------

def set_names(set_key, n=None):
    """The instances of a set, in the row order of the stored arrays, optionally the
    first n."""
    if set_key == 'pub10x5':
        names = list(P105.public_names())
    elif set_key == '10x25':
        names = list(P101.names_of('10x25'))
    else:
        names = list(P105.planned_names(set_key))
    return names[:n] if n else names


def replay_set(set_key):
    """The instances of a set whose references are replayed (the planned instances of
    p105)."""
    return set(P105.planned_names(set_key))


def draw_seeds(set_key, name):
    """SeedSequence entropy of the ten random rollouts of one instance (P101.draw_seeds for
    the factory sets, code 105 for the public set)."""
    return [[RANDOM_SEED, SETS[set_key]['code'], P101.instance_index(name), k]
            for k in range(N_DRAWS)]


def stored_rules(set_key, combo, name):
    """{rule: makespan} of the stored exact-mask rule arrays of one instance."""
    arr = P105.rule_arrays(set_key, combo)
    row = P105.all_names(set_key).index(name)
    return {r: float(a[row]) for r, a in arr.items()}


def arms_of(set_key, refs):
    """The reference arms of p105 that cover the set, among refs."""
    return [r for r in refs if set_key in P105.REFS[r]['sets']]


def case_references(states, set_key, combo, name, refs):
    """{arm: (state, reason, record path, proved)} of the references of one case (empty
    outside the replay set)."""
    if name not in replay_set(set_key):
        return {}
    return {arm: P105.ref_state(states, set_key, arm, combo, name)
            for arm in arms_of(set_key, refs)}


def p101_random_row(set_key, combo, name):
    """The makespans [10] of p101's random control of one instance, or None without a
    cell of p101 that covers it."""
    if set_key not in P101.SIZES:
        return None
    npz, js = P101.cell_paths(os.path.join(ROOT_DIR, P101.DEFAULT_ROOT), set_key, combo,
                              'random')
    if not (os.path.exists(npz) and os.path.exists(js)):
        return None
    with open(js) as f:
        rec = json.load(f)
    inst = rec.get('instances', [])
    if name not in inst or rec.get('design', {}).get('kind') != 'random':
        return None
    with np.load(npz) as z:
        return np.asarray(z['makespan'], dtype=float)[inst.index(name)]


def p105_case(set_key, arm, combo, name, files):
    """p105's stored case of one reference when it exists with the same design and the
    same reference record, else None."""
    js, npz = P105.case_paths(os.path.join(ROOT_DIR, P105.DEFAULT_ROOT), set_key, arm, combo,
                              name)
    if not (os.path.exists(js) and os.path.exists(npz)):
        return None
    with open(js) as f:
        rec = json.load(f)
    if (rec.get('design') != P101.jsonable(P105.design(set_key, arm, combo))
            or rec.get('reference', {}).get('files') != files):
        return None
    return rec


# ---------------------------------------------------------------------------
# cases: design record, paths, state
# ---------------------------------------------------------------------------

def design(set_key, combo, refs):
    """The fields that define a case's content; a stored case with other fields is
    refused."""
    return dict(kind='admissibility rules', version=VERSION, set=set_key, combo=combo,
                rules=ADM_TEXT, mask=P101.MASK, dispatching=list(RULES),
                rule_set='common_utils.masked_rule_action after np.random.seed(50)',
                draws=N_DRAWS,
                stream=(f'default_rng(SeedSequence([{RANDOM_SEED}, {SETS[set_key]["code"]}, '
                        f'instance index, draw]))'),
                orders=P105.ORDER_TEXT, refs=list(arms_of(set_key, refs)),
                validator='reference_sim.validate_schedule (P101.validate)')


def case_paths(root, set_key, combo, name):
    stem = os.path.join(root, set_key, combo, name)
    return stem + '.json', stem + '.npz'


def reference_files(set_key, refinfo):
    """{arm: files {relative path: sha256} or None} of the references of one case."""
    out = {}
    for arm, (st, _, path, _) in refinfo.items():
        out[arm] = P105.read_reference(set_key, arm, path)[4] if st == 'valid' else None
    return out


def case_state(root, set_key, combo, name, refs, files):
    """'done' when the case holds this design and these reference records, 'absent' when
    missing, 'other' otherwise."""
    js, npz = case_paths(root, set_key, combo, name)
    if not (os.path.exists(js) and os.path.exists(npz)):
        return 'absent', None
    with open(js) as f:
        rec = json.load(f)
    if rec.get('design') != P101.jsonable(design(set_key, combo, refs)):
        return 'other', rec
    got = {a: (r.get('files') if r.get('state') == 'valid' else None)
           for a, r in rec.get('references', {}).items()}
    if got != files:
        return 'other', rec
    return 'done', rec


def timing_record(t0, c0):
    wall, cpu = time.perf_counter() - t0, time.process_time() - c0
    return dict(wall_s=wall, cpu_s=cpu, cpu_share=cpu / max(wall, 1e-9),
                affinity=sorted(os.sched_getaffinity(0)), loadavg=list(os.getloadavg()),
                python=platform.python_version(), omp=os.environ.get('OMP_NUM_THREADS'),
                finished=time.strftime('%Y-%m-%d %H:%M:%S'),
                timing_note='planning figures only; the rollouts are deterministic')


def run_case(root, set_key, combo, name, refinfo, refs, force):
    """One case: load, check the references, the three rules, the checks, write. Returns
    ('done' | 'skip', cpu s)."""
    files = reference_files(set_key, refinfo)
    state, _ = case_state(root, set_key, combo, name, refs, files)
    if state == 'done' and not force:
        return 'skip', 0.0
    if state == 'other' and not force:
        stop(f'the case {set_key}/{combo}/{name} under --root holds another design or '
             f'reference record; use --force or another --root')
    t0, c0 = time.perf_counter(), time.process_time()
    where = f'{set_key}/{combo}/{name}'
    jl, pt, meta, desc = P105.load_instance_case(set_key, combo, name)
    ref_arrays, references = {}, {}
    for arm, (st, why, path, proved) in refinfo.items():
        if st != 'valid':
            references[arm] = dict(state=st, reason=why)
            continue
        amch, rst, rct, obj, fl, extra = P105.read_reference(set_key, arm, path)
        ms_ref = P105.check_reference(jl, pt, desc, amch, rst, rct, obj, f'{where}/{arm}')
        ref_arrays[arm] = dict(amch=amch, st=rst, ct=rct)
        references[arm] = dict(state='valid', files=fl, objective=obj, makespan=ms_ref,
                               proved_optimal=bool(proved if proved is not None
                                                   else extra['proved']),
                               status=extra.get('status'))
    stored = stored_rules(set_key, combo, name)
    seeds = draw_seeds(set_key, name)
    rec, arrays = evaluate_case(jl, pt, meta, desc, combo, seeds, ref_arrays, stored, where)
    # cross-checks with the cells of p101 and p105 where they exist
    nd = rec['adm']['nondelay']
    row = p101_random_row(set_key, combo, name)
    if row is not None:
        if not np.array_equal(row, np.asarray(nd['random']['makespan'], dtype=float)):
            stop(f'{where}: the nondelay random rollouts differ from the cell of p101 '
                 f'({row.tolist()} against {nd["random"]["makespan"]})')
        rec['checks']['p101_random'] = 'equal to the cell of p101'
    else:
        rec['checks']['p101_random'] = 'no cell of p101 covers this instance'
    rec['checks']['p105_replay'] = {}
    for arm in ref_arrays:
        c = p105_case(set_key, arm, combo, name, references[arm]['files'])
        if c is None:
            rec['checks']['p105_replay'][arm] = 'no case of p105 with this reference record'
            continue
        for od in ORDERS:
            a, b = nd['replay'][arm][od], c['replay'][od]
            if (a['makespan'] != b['makespan'] or a['off'] != b['off']
                    or a['exact'] != b['exact']):
                stop(f'{where}: the nondelay replay of {arm} by order {od} differs from the '
                     f'case of p105 ({a} against {b})')
        rec['checks']['p105_replay'][arm] = 'equal to the case of p105 (orders A, B, C)'
    rec.update(design=P101.jsonable(design(set_key, combo, refs)), set=set_key, combo=combo,
               instance=name, row=P105.all_names(set_key).index(name),
               n_ops=int(pt.shape[0]), n_jobs=int(len(jl)), n_machines=int(pt.shape[1]),
               stored_rules=stored, references=references, seeds=seeds)
    rec['timing'] = timing_record(t0, c0)
    js, npz = case_paths(root, set_key, combo, name)
    P101.save_npz_atomic(npz, **arrays)
    P101.write_json_atomic(js, P101.jsonable(rec))
    return 'done', rec['timing']['cpu_s']


# ---------------------------------------------------------------------------
# plan and run
# ---------------------------------------------------------------------------

def case_list(args):
    """(set, combo, name) of every planned case of the selection, in a fixed order."""
    return [(s, c, nm) for s in args.sets for c in args.combos
            for nm in set_names(s, args.n)]


def assign_parts(cases, parts):
    """Longest-processing-time assignment of the cases to parts."""
    order = sorted(range(len(cases)), key=lambda i: (-COST_S[cases[i][0]], i))
    loads, out = [0.0] * parts, [[] for _ in range(parts)]
    for i in order:
        p = int(np.argmin(loads))
        out[p].append(i)
        loads[p] += COST_S[cases[i][0]]
    return [sorted(x) for x in out], loads


def states_for(args):
    """P105's reference states for the arms that cover the selected sets."""
    need = sorted({a for s in args.sets for a in arms_of(s, args.refs)})
    a = argparse.Namespace(refs=need, cpsat_arm=args.cpsat_arm)
    return P105.reference_states(a) if need else dict(factory={}, public={}, cpsat_arm=None,
                                                      cpsat_why='not requested')


def plan(args):
    states = states_for(args)
    cases = case_list(args)
    parts, loads = assign_parts(cases, args.parts)
    summary = {}
    for (s, c, nm) in cases:
        k = summary.setdefault(s, dict(planned=0, done=0, est_left_s=0.0, refs={}))
        k['planned'] += 1
        ri = case_references(states, s, c, nm, args.refs)
        for arm, (st, _, _, _) in ri.items():
            k['refs'].setdefault(arm, {}).setdefault(st, 0)
            k['refs'][arm][st] += 1
        if case_state(args.root, s, c, nm, args.refs, reference_files(s, ri))[0] == 'done':
            k['done'] += 1
        else:
            k['est_left_s'] += COST_S[s]
    total = sum(COST_S[x[0]] for x in cases)
    left = sum(k['est_left_s'] for k in summary.values())
    print(f'[p107 plan] {len(cases)} planned cases, about {total / 3600:.2f} core-hours of CPU '
          f'time in all and {left / 3600:.2f} left (COST_S per case, measured on smoke cases), '
          f'{args.parts} parts of {min(loads) / 60:.1f} to {max(loads) / 60:.1f} min')
    for s, k in summary.items():
        refs = '; '.join(f'{a}: ' + ', '.join(f'{v} {st}' for st, v in sorted(d.items()))
                         for a, d in k['refs'].items()) or 'none'
        print(f'  {s:9s} planned {k["planned"]:4d}  done {k["done"]:4d}  left about '
              f'{k["est_left_s"] / 60:6.1f} min  references: {refs}')
    py = sys.executable + ' -B'
    sel = f' --sets {",".join(args.sets)}' if args.sets != list(SETS) else ''
    for i in range(args.parts):
        print(f'CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c '
              f'<CORE> {py} scripts/p107_mask_rule_variants.py run{sel} --part {i} '
              f'--parts {args.parts}')
    return dict(cases=len(cases), core_h=total / 3600, left_h=left / 3600,
                part_min=[x / 60 for x in loads], summary=summary)


def run(args):
    P101._cpu_torch()
    states = states_for(args)
    cases = case_list(args)
    if args.part is not None:
        parts, _ = assign_parts(cases, args.parts)
        cases = [cases[i] for i in parts[args.part]]
    t_all = time.perf_counter()
    counts = {}
    for k, (s, c, nm) in enumerate(cases):
        ri = case_references(states, s, c, nm, args.refs)
        if args.budget_s is not None:
            el = time.perf_counter() - t_all
            if (el + COST_S[s] > args.budget_s and case_state(
                    args.root, s, c, nm, args.refs, reference_files(s, ri))[0] != 'done'):
                counts['not reached (budget)'] = len(cases) - k
                print(f'[p107 run] budget of {args.budget_s:.0f} s reached after {el:.0f} s',
                      flush=True)
                break
        res, cpu = run_case(args.root, s, c, nm, ri, args.refs, args.force)
        counts[res] = counts.get(res, 0) + 1
        if res == 'done':
            print(f'[p107 run {k + 1}/{len(cases)}] {s:9s} {c:4s} {nm} cpu={cpu:6.1f}s '
                  f'elapsed={time.perf_counter() - t_all:7.0f}s', flush=True)
    print(f'[p107 run] {len(cases)} cases in this command: '
          + ', '.join(f'{v} {k}' for k, v in sorted(counts.items())), flush=True)
    return counts


# ---------------------------------------------------------------------------
# infer
# ---------------------------------------------------------------------------

def gap(a, b):
    return SF.rel_gap(a, b)


boot = P105.boot


def pct(x):
    x = np.asarray(x, dtype=bool)
    return 100.0 * float(x.mean())


def qvals(recs, adm, q):
    """Per-case values of quantity q under rule adm; recs [n][c] of case records."""
    def one(r):
        A = r['adm'][adm]
        if q == 'best_rule':
            return A['best_rule']
        if q in RULES:
            return A['rules'][q]['makespan']
        if q == 'rand_mean':
            return A['random']['mean']
        if q == 'rand_best':
            return A['random']['best']
        raise ValueError(q)
    return np.array([[one(r) for r in row] for row in recs], dtype=float)


def replay_vals(recs, adm, arm):
    return np.array([[r['adm'][adm]['replay'][arm]['best'] for r in row] for row in recs],
                    dtype=float)


def block_stats(recs, arms):
    """Outcomes of one row: recs [n][c] case records (an instance brings its c
    combinations). Every quantity is an array [n, c]."""
    n, c = len(recs), len(recs[0])
    flat = [r for row in recs for r in row]
    out = dict(n=n, combos=c, cases=n * c, diff={}, ms={}, replay={}, pairs={})
    vals = {adm: {q: qvals(recs, adm, q) for q in QUANTS} for adm in ADM}
    for adm in ADM:
        out['ms'][adm] = {q: float(vals[adm][q].mean()) for q in QUANTS}
    for x in ALT:
        out['diff'][x] = {q: boot(gap(vals[x][q], vals['nondelay'][q])) for q in QUANTS}
        out['diff'][x]['shorter_cases'] = {
            q: int((vals[x][q] < vals['nondelay'][q] - EPS).sum()) for q in QUANTS}
        out['diff'][x]['longer_cases'] = {
            q: int((vals[x][q] > vals['nondelay'][q] + EPS).sum()) for q in QUANTS}
    for adm in ADM:
        R = [r['adm'][adm]['rules']['SPT'] for r in flat]
        dec = sum(x['decisions'] for x in R)
        out['pairs'][adm] = dict(
            decisions=int(dec), mean=sum(x['pairs_sum'] for x in R) / dec,
            one_share=100.0 * sum(x['one_pair'] for x in R) / dec,
            differ_share=100.0 * sum(x['differ_nondelay'] for x in R) / dec)
    bd = [r['same_as_nondelay']['block'] for r in flat]
    out['block_differs'] = dict(
        any=int(sum(x['any'] for x in bd)), cases=len(bd),
        rules=int(sum(bool(x['rules']) for x in bd)),
        random=int(sum(x['random'] > 0 for x in bd)),
        replay=int(sum(bool(x['replay']) for x in bd)),
        rule_schedules=int(sum(len(x['rules']) for x in bd)),
        random_schedules=int(sum(x['random'] for x in bd)))
    for arm in arms:
        rec_ok = [[r for r in row] for row in recs
                  if all(r.get('references', {}).get(arm, {}).get('state') == 'valid'
                         for r in row)]
        if not rec_ok:
            continue
        ref = np.array([[r['references'][arm]['makespan'] for r in row] for row in rec_ok],
                       dtype=float)
        rv = {adm: replay_vals(rec_ok, adm, arm) for adm in ADM}
        P = dict(n=len(rec_ok), cases=int(ref.size), ref_ms=float(ref.mean()),
                 ms={adm: float(rv[adm].mean()) for adm in ADM},
                 gap={adm: boot(gap(rv[adm], ref)) for adm in ADM},
                 diff={x: boot(gap(rv[x], rv['nondelay'])) for x in ALT},
                 exact={adm: pct([[r['adm'][adm]['replay'][arm]['A']['exact'] for r in row]
                                  for row in rec_ok]) for adm in ADM},
                 proved=int(sum(r['references'][arm].get('proved_optimal', False)
                                for row in rec_ok for r in row)))
        P['shorter_cases'] = {x: int((rv[x] < rv['nondelay'] - EPS).sum()) for x in ALT}
        P['longer_cases'] = {x: int((rv[x] > rv['nondelay'] + EPS).sum()) for x in ALT}
        out['replay'][arm] = P
    return out


def read_case(root, set_key, combo, name, refs, refinfo):
    """(record, None) of a case that counts, or (None, reason)."""
    files = reference_files(set_key, refinfo)
    st, rec = case_state(root, set_key, combo, name, refs, files)
    if st != 'done':
        return None, (f'{set_key}/{combo}/{name}: case '
                      + ('not run' if st == 'absent' else 'of another design or reference '
                         'record'))
    if rec.get('instance') != name or rec.get('combo') != combo or rec.get('set') != set_key:
        refuse(f'{set_key}/{combo}/{name}: the case file names {rec.get("set")}/'
               f'{rec.get("combo")}/{rec.get("instance")}')
    for arm, (s, why, _, _) in refinfo.items():
        if s != 'valid':
            return None, f'{set_key}/{combo}/{name}: reference {arm} {s}' + (
                f' ({why})' if why else '')
    return rec, None


def recheck(rec, set_key, combo, name):
    """The built-in checks again, from the case record and the stored arrays."""
    stored = stored_rules(set_key, combo, name)
    for r in RULES:
        if rec['adm']['nondelay']['rules'][r]['makespan'] != stored.get(r):
            refuse(f'{set_key}/{combo}/{name}: the nondelay {r} rollout '
                   f'({rec["adm"]["nondelay"]["rules"][r]["makespan"]}) differs from the '
                   f'stored array ({stored.get(r)}); the pipelines diverge')
    if combo in NOSETUP_COMBOS and (rec['same_as_nondelay']['block']['any'] or rec['setups']):
        refuse(f'{set_key}/{combo}/{name}: without setups the case records a block schedule '
               f'that differs from nondelay')


def set_stats(args, states, set_key):
    names = set_names(set_key, args.n)
    arms = arms_of(set_key, args.refs)
    S = dict(set=set_key, word=SETS[set_key]['word'], text=SETS[set_key]['text'],
             planned=len(names), arms=arms, rows={}, missing=[])
    got = {}
    for c in args.combos:
        got[c] = {}
        for nm in names:
            ri = case_references(states, set_key, c, nm, args.refs)
            rec, why = read_case(args.root, set_key, c, nm, args.refs, ri)
            if rec is None:
                S['missing'].append(why)
                continue
            recheck(rec, set_key, c, nm)
            got[c][nm] = rec
    for c in args.combos:
        nm = [x for x in names if x in got[c]]
        if not nm:
            S['rows'][c] = dict(n=0)
            continue
        S['rows'][c] = block_stats([[got[c][x]] for x in nm], arms)
        S['rows'][c]['label'] = c
    for lab, cs in (('All', list(args.combos)),
                    ('Setup', [c for c in args.combos if c in SETUP_COMBOS])):
        cs = [c for c in cs if got[c]]
        common = [x for x in names if cs and all(x in got[c] for c in cs)]
        if len(cs) < 2 or not common:
            continue
        R = block_stats([[got[c][x] for c in cs] for x in common], arms)
        R['label'] = lab
        R['combo_list'] = cs
        S['rows'][lab] = R
    return S


def infer_outputs(out_root):
    return {k: os.path.join(out_root, v) for k, v in OUT_FILES.items()}


def infer(args):
    """Every outcome of the module doc from the cases of --root. Returns the result
    dictionary."""
    partial = bool(args.partial)
    if args.out is None:
        if partial:
            refuse('--partial writes only to --out <folder>')
        out_root = ROOT_DIR
    else:
        out_root = os.path.abspath(args.out)
        P101.guard_out(out_root)
    paths = infer_outputs(out_root)
    for p in paths.values():                # a refusal leaves no stale output
        if os.path.exists(p):
            os.remove(p)
    if not partial:
        if args.n or args.combos != list(R8) or args.sets != list(SETS):
            refuse('every set, combination and planned instance is required; a subset needs '
                   '--partial')
    states = states_for(args)
    res = {'generator': GENERATOR, 'partial': partial, 'mask': P101.MASK, 'rules': ADM_TEXT,
           'orders': P105.ORDER_TEXT, 'draws': N_DRAWS,
           'random_stream': (f'default_rng(SeedSequence([{RANDOM_SEED}, code, instance index, '
                             f'draw])), codes ' + ', '.join(f'{s} {v["code"]}'
                                                           for s, v in SETS.items())),
           'diff': '100 (x_X - x_nondelay) / x_nondelay per case; negative means rule X '
                   'gives the shorter schedule',
           'gap': '100 (replay_X - reference) / reference per case',
           'bootstrap': f'stats_family.instance_boot, {B} resamples, fresh '
                        f'default_rng({CI_SEED}); pooled, a drawn instance brings all its '
                        f'combinations',
           'pairs': 'along the SPT rollout under each rule: admissible pairs before every '
                    'decision, over all decisions of all cases of the row',
           'rows': 'per combination; All: pooled over the selected combinations; Setup: '
                   'pooled over S, LS, SW, LSW',
           'refs': {r: P105.REFS[r]['text'] for r in args.refs},
           'cpsat_configuration': states.get('cpsat_arm'),
           'combos': list(args.combos), 'n': args.n, 'sets': {}}
    missing = []
    for s in args.sets:
        S = set_stats(args, states, s)
        missing += S['missing']
        res['sets'][s] = S
    if missing and not partial:
        refuse(f'{len(missing)} planned cases do not count, e.g. ' + '; '.join(missing[:4]))
    res['complete'] = not missing
    res['missing_count'] = len(missing)
    res['missing_examples'] = missing[:20]
    if not any(R.get('n') for S in res['sets'].values() for R in S['rows'].values()):
        refuse('no case counts; nothing to report')
    lines = macro_lines(res)
    names_m = re.findall(r'\\newcommand\{\\([^}]*)\}', '\n'.join(lines))
    res['macro_files_checked'] = check_macro_names(names_m, paths)
    from macro_io import emit
    emit(paths['macros'], lines)
    P101.write_json_atomic(paths['json'], P101.jsonable(res))
    os.makedirs(os.path.dirname(paths['md']), exist_ok=True)
    with open(paths['md'], 'w') as f:
        f.write('\n'.join(markdown(res)) + '\n')
    for k in ('json', 'md'):
        print('wrote', os.path.relpath(paths[k], out_root))
    print(f'{len(names_m)} macros' + ('; PARTIAL' if partial else '')
          + (f'; {len(missing)} planned cases do not count yet' if missing else ''))
    return res


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------

f2, u1, u2 = P101.f2, P101.u1, P101.u2


def row_items(S):
    rows = [(c, R) for c, R in S['rows'].items() if R.get('n') and c in R8]
    rows += [(c, S['rows'][c]) for c in ROWS_EXTRA if c in S['rows']]
    return rows


def row_text(lab, R):
    if lab == 'All':
        return f'pooled over {",".join(R["combo_list"])}'
    if lab == 'Setup':
        return f'pooled over the combinations with setups {",".join(R["combo_list"])}'
    return lab


def macro_lines(res):
    mark = 'PARTIAL ' if res['partial'] else ''
    if not res['complete']:
        mark += 'INCOMPLETE '
    L = [f'% Generated by {GENERATOR} -- do not hand-edit',
         '% Three admissibility rules in the environment of the paper, without training: '
         'nondelay (processing starts, the rule of the paper), block (classical non-delay on '
         'block starts) and active (Giffler-Thompson).',
         '% Differences in %: 100 (x_X - x_nondelay) / x_nondelay per case, mean over cases, '
         '95% bootstrap over instances (pooled: an instance brings its combinations); '
         'negative means rule X gives the shorter schedule.']
    if res['partial']:
        L.insert(1, '% PARTIAL: subset of sets, combinations or instances; not for the '
                    'manuscript.')
    if not res['complete']:
        L.insert(1, f'% INCOMPLETE: {res["missing_count"]} planned cases do not count yet.')
    L.append(f'\\newcommand{{\\{PREFIX}Boot}}{{{B}}}\\newcommand{{\\{PREFIX}DrawN}}{{{N_DRAWS}}}'
             f'  % {mark}bootstrap resamples; random rollouts per instance and rule')
    for s, S in res['sets'].items():
        w = S['word']
        for lab, R in row_items(S):
            t = f'{s}, {row_text(lab, R)}'
            L.append(f'\\newcommand{{\\{PREFIX}N{w}{lab}}}{{{R["n"]}}}'
                     f'\\newcommand{{\\{PREFIX}Cases{w}{lab}}}{{{R["cases"]}}}'
                     f'  % {mark}{t}: instances, and cases (instances times combinations)')
            for x in ALT:
                X = ADM_WORD[x]
                for q in QUANTS:
                    g = R['diff'][x][q]
                    a = f'Diff{X}{Q_WORD[q]}'
                    L.append(f'\\newcommand{{\\{PREFIX}{a}{w}{lab}}}{{{f2(g["mean"])}}}'
                             f'\\newcommand{{\\{PREFIX}{a}Lo{w}{lab}}}{{{f2(g["lo"])}}}'
                             f'\\newcommand{{\\{PREFIX}{a}Hi{w}{lab}}}{{{f2(g["hi"])}}}'
                             f'  % {mark}{t}: {Q_TEXT[q]} under the {x} rule against '
                             f'nondelay, %, mean and 95% interval ([{g["lo"]:+.4f}, '
                             f'{g["hi"]:+.4f}]); negative = {x} shorter')
            for arm, P in R['replay'].items():
                rw = REF_WORD[arm]
                for x in ALT:
                    g = P['diff'][x]
                    a = f'Diff{ADM_WORD[x]}Replay{rw}'
                    L.append(f'\\newcommand{{\\{PREFIX}{a}{w}{lab}}}{{{f2(g["mean"])}}}'
                             f'\\newcommand{{\\{PREFIX}{a}Lo{w}{lab}}}{{{f2(g["lo"])}}}'
                             f'\\newcommand{{\\{PREFIX}{a}Hi{w}{lab}}}{{{f2(g["hi"])}}}'
                             f'  % {mark}{t}: replay of the {arm} references (shortest of '
                             f'orders A, B, C) under the {x} rule against nondelay, %, mean and '
                             f'95% interval over {P["n"]} instances ([{g["lo"]:+.4f}, '
                             f'{g["hi"]:+.4f}]); negative = {x} shorter')
                for adm in ADM:
                    g = P['gap'][adm]
                    a = f'Gap{ADM_WORD[adm]}{rw}'
                    L.append(f'\\newcommand{{\\{PREFIX}{a}{w}{lab}}}{{{f2(g["mean"])}}}'
                             f'\\newcommand{{\\{PREFIX}{a}Lo{w}{lab}}}{{{f2(g["lo"])}}}'
                             f'\\newcommand{{\\{PREFIX}{a}Hi{w}{lab}}}{{{f2(g["hi"])}}}'
                             f'\\newcommand{{\\{PREFIX}Exact{ADM_WORD[adm]}{rw}{w}{lab}}}'
                             f'{{{u1(P["exact"][adm])}}}'
                             f'  % {mark}{t}: gap of the replay under the {adm} rule to the '
                             f'{arm} reference, %, mean and 95% interval ([{g["lo"]:+.4f}, '
                             f'{g["hi"]:+.4f}]), an upper bound on what the rule costs; share '
                             f'(%) of references order A reproduces exactly under the rule')
            pp = ''.join(f'\\newcommand{{\\{PREFIX}Pairs{ADM_WORD[a]}{w}{lab}}}'
                         f'{{{u2(R["pairs"][a]["mean"])}}}'
                         f'\\newcommand{{\\{PREFIX}One{ADM_WORD[a]}{w}{lab}}}'
                         f'{{{u1(R["pairs"][a]["one_share"])}}}' for a in ADM)
            L.append(pp + f'  % {mark}{t}: along the SPT rollout under nondelay, block and '
                          f'active, mean admissible pairs per decision and share (%) of '
                          f'decisions with one admissible pair')
            L.append(''.join(f'\\newcommand{{\\{PREFIX}DiffSet{ADM_WORD[a]}{w}{lab}}}'
                             f'{{{u1(R["pairs"][a]["differ_share"])}}}' for a in ALT)
                     + f'  % {mark}{t}: share (%) of the decisions of the SPT rollout under '
                       f'block and under active at which the admissible set differs from the '
                       f'nondelay set of the same state')
            bd = R['block_differs']
            L.append(f'\\newcommand{{\\{PREFIX}BlkDiffCases{w}{lab}}}{{{bd["any"]}}}'
                     f'\\newcommand{{\\{PREFIX}BlkDiffRuleCases{w}{lab}}}{{{bd["rules"]}}}'
                     f'  % {mark}{t}: cases (of {bd["cases"]}) in which the block rule gives '
                     f'a schedule other than nondelay in any rollout or replay, and in a rule '
                     f'rollout')
    return L


def check_macro_names(names, out_paths):
    """Every name is the prefix plus letters, unique, and absent from every other macro
    file of the repository's paper folders (the check of p101)."""
    bad = [x for x in names if not re.fullmatch(PREFIX + r'[A-Za-z]+', x)]
    if bad:
        refuse(f'macro names that are not {PREFIX} plus letters: {bad[:5]}')
    if len(names) != len(set(names)):
        refuse(f'duplicate macro names: {sorted({x for x in names if names.count(x) > 1})[:5]}')
    import glob
    own = {os.path.abspath(out_paths['macros']),
           os.path.abspath(os.path.join(ROOT_DIR, OUT_FILES['macros']))}
    files = sorted(set(glob.glob(os.path.join(ROOT_DIR, 'paper*', 'macros*.tex'))
                       + glob.glob(os.path.join(ROOT_DIR, 'paper*', '*', 'macros*.tex'))
                       + glob.glob(os.path.join(ROOT_DIR, 'manuscript', 'sections_new', '*.tex'))))
    rx = re.compile(r'\\(?:newcommand|renewcommand|providecommand|def)\s*\{?\\([A-Za-z]+)')
    for f in files:
        if os.path.abspath(f) in own:
            continue
        with open(f, errors='replace') as fh:
            clash = set(names) & set(rx.findall(fh.read()))
        if clash:
            refuse(f'macro name collision with {os.path.relpath(f, ROOT_DIR)}: '
                   f'{sorted(clash)[:5]}')
    return len(files)


def markdown(res):
    L = [f'# Three admissibility rules without training ({GENERATOR})', '']
    if res['partial']:
        L += ['PARTIAL: a subset of sets, combinations or instances. Not for the manuscript.',
              '']
    if not res['complete']:
        L += [f'INCOMPLETE: {res["missing_count"]} planned cases do not count yet, e.g.:', '']
        L += [f'- {x}' for x in res['missing_examples'][:8]] + ['']
    L += ['Rules: ' + '; '.join(f'{a}: {t}' for a, t in res['rules'].items()) + '. '
          f'Difference: {res["diff"]}. Gap: {res["gap"]}. Interval: {res["bootstrap"]}. '
          f'Pairs: {res["pairs"]}. Rows: {res["rows"]}. Random control: {N_DRAWS} rollouts, '
          f'{res["random_stream"]}.', '']

    def g(x):
        return f'{f2(x["mean"])} [{f2(x["lo"])}, {f2(x["hi"])}]'
    for s, S in res['sets'].items():
        rows = row_items(S)
        L += [f'## {s}: {S["text"]}', '']
        if not rows:
            L += ['No case counts yet.', '']
            continue
        for x in ALT:
            L += [f'{x} against nondelay, difference (%) and cases shorter / longer:', '',
                  '| row | instances | cases | best rule | FIFO | MOR | SPT | MWKR | random mean '
                  '| random shortest |' + ''.join(f' replay {a} |' for a in S['arms']),
                  '|' + '---|' * (10 + len(S['arms']))]
            for lab, R in rows:
                D = R['diff'][x]
                cells = [f'{g(D[q])} ({D["shorter_cases"][q]} / {D["longer_cases"][q]})'
                         for q in QUANTS]
                rp = [(f'{g(R["replay"][a]["diff"][x])} ({R["replay"][a]["shorter_cases"][x]} '
                       f'/ {R["replay"][a]["longer_cases"][x]})') if a in R['replay'] else '--'
                      for a in S['arms']]
                L.append(f'| {row_text(lab, R)} | {R["n"]} | {R["cases"]} | '
                         + ' | '.join(cells + rp) + ' |')
            L.append('')
        L += ['Mean makespan (h) under nondelay / block / active:', '',
              '| row | best rule | random mean | random shortest |'
              + ''.join(f' replay {a} | reference {a} |' for a in S['arms']),
              '|' + '---|' * (4 + 2 * len(S['arms']))]
        for lab, R in rows:
            ms = R['ms']
            cells = [' / '.join(u1(ms[a][q]) for a in ADM)
                     for q in ('best_rule', 'rand_mean', 'rand_best')]
            for a in S['arms']:
                if a in R['replay']:
                    P = R['replay'][a]
                    cells += [' / '.join(u1(P['ms'][d]) for d in ADM), u1(P['ref_ms'])]
                else:
                    cells += ['--', '--']
            L.append(f'| {row_text(lab, R)} | ' + ' | '.join(cells) + ' |')
        L.append('')
        for a in S['arms']:
            L += [f'Replay of the {a} references: gap to the reference (%) and references '
                  f'reproduced exactly by order A (%), per rule:', '',
                  '| row | instances | gap nondelay | gap block | gap active | exact nondelay '
                  '/ block / active |', '|' + '---|' * 6]
            for lab, R in rows:
                if a not in R['replay']:
                    continue
                P = R['replay'][a]
                L.append(f'| {row_text(lab, R)} | {P["n"]} | {g(P["gap"]["nondelay"])} | '
                         f'{g(P["gap"]["block"])} | {g(P["gap"]["active"])} | '
                         + ' / '.join(u1(P['exact'][d]) for d in ADM) + ' |')
            L.append('')
        L += ['Along the SPT rollout: admissible pairs per decision, decisions with one pair '
              '(%) and decisions whose set differs from the nondelay set (%); cases in which '
              'block differs from nondelay in any schedule (in a rule rollout, a random '
              'rollout, a replay):', '',
              '| row | decisions | pairs nondelay / block / active | one pair nondelay / '
              'block / active | set differs block / active | block differs: any (rules, '
              'random, replay) of cases |', '|' + '---|' * 6]
        for lab, R in rows:
            pp = R['pairs']
            bd = R['block_differs']
            L.append(f'| {row_text(lab, R)} | {pp["nondelay"]["decisions"]} | '
                     + ' / '.join(u2(pp[a]['mean']) for a in ADM) + ' | '
                     + ' / '.join(u1(pp[a]['one_share']) for a in ADM) + ' | '
                     + ' / '.join(u1(pp[a]['differ_share']) for a in ALT) + ' | '
                     f'{bd["any"]} ({bd["rules"]}, {bd["random"]}, {bd["replay"]}) of '
                     f'{bd["cases"]} |')
        L.append('')
    return L


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('phase', choices=['run', 'infer', 'plan'])
    ap.add_argument('--root', default=DEFAULT_ROOT, help='folder of the cases')
    ap.add_argument('--sets', default=','.join(SETS))
    ap.add_argument('--refs', default=','.join(DEFAULT_REFS),
                    help='reference arms of p105 to replay (cpsat150, memetic_rule600, '
                         'cpsat_rule600)')
    ap.add_argument('--combos', default=','.join(R8))
    ap.add_argument('--n', type=int, default=0,
                    help='the first n instances of each set (smoke only)')
    ap.add_argument('--cpsat_arm', default=None,
                    help='CP-SAT configuration of cpsat_rule600; must equal the pilot\'s choice')
    ap.add_argument('--part', type=int, default=None)
    ap.add_argument('--parts', type=int, default=1)
    ap.add_argument('--budget_s', type=float, default=None,
                    help='run: start no case whose estimated cost would end the command after '
                         'this many seconds')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--partial', action='store_true',
                    help='infer: the cases present of a selection, under --out')
    ap.add_argument('--out', default=None, help='infer: root of the outputs')
    args = ap.parse_args(_ARGV)
    P101.check_mask_options()
    args.sets = P101.parse_list(args.sets, SETS, 'sets')
    args.refs = P101.parse_list(args.refs, P105.REFS, 'reference arms')
    args.combos = P101.parse_list(args.combos, R8, 'combinations')
    args.n = args.n or None
    if args.part is not None and not 0 <= args.part < args.parts:
        sys.exit(f'--part {args.part} outside 0..{args.parts - 1}')
    os.chdir(ROOT_DIR)
    {'run': run, 'infer': infer, 'plan': plan}[args.phase](args)


if __name__ == '__main__':
    main()
