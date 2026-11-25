r"""P104: does the batch stepped together change the policy's greedy schedule of an instance?

Subcommands run, infer and plan. Evaluation only, CPU only, exact mask.

LAUNCH NOTE (the four questions, answered before the run)

1. Purpose. The environment divides its time features by the largest processing time of
   the instances it steps together: fjsp_env_same_op_nums.py, set_initial_data, takes
   pt_upper_bound = np.max(self.op_pt) over the whole batch [E, N, M] and normalizes
   op_pt and the lags by it; constraint_family/fjsp_env_family.py, set_initial_data,
   scales the setups, the window calendars, the horizon and the release dates by the
   same slope 1 / (raw_pt.max() - raw_pt.min() + 1e-8) of the stacked batch. Training
   steps several instances together, so it sees the batch maximum, while evaluation
   steps one instance and sees that instance's own maximum. The question is whether the
   composition of the batch changes the policy's greedy schedule, and whether an
   instance stepped alone is systematically longer or shorter than stepped with others.
   The numbers land in the supplement section on the environment (S2.6).
2. Expected result. Some cases differ. A smoke run with a stored checkpoint of the same
   architecture (10 instances, N and LSW) gave different makespans in 10 of 20 cases,
   by up to 4.6%, and every case that differed had a near tie along its trajectory
   alone. The paper states the count, the mean signed difference with its 95% interval
   and the largest difference; the stored arrays, a batch of one, stay the reference of
   every evaluation.
3. Contamination. Checkpoints, instances, descriptors and stored arrays are read only.
   Cells go to their own folder, test_results/FAMILY/divisor_p104 (--root), one file per
   combination, written through a temporary file and a rename; a cell of another
   design, run or instance list is refused, never overwritten (unless --force). Greedy
   rollouts on one torch thread are deterministic, so the cells are not timing
   measurements and the run may share cores.
4. Data accuracy. The run is the selected run of the exact-mask environment
   (arm_registry.select_seed over --seeds, the lowest validation score), resolved by
   scripts/p97_policy_anytime.py resolve_run, whose training configuration must carry
   the exact mask; a run named by --model is a development run and its cells are
   marked so (infer refuses them unless --partial). Case (i) of every instance must
   equal the stored greedy array of the run (test_results/FAMILY/10x25_xmask/
   Result_greedy+<model>_<combination>.npy) where it exists, or run refuses. Every
   schedule passes the validator of scripts/p101_mask_controls.py.

DESIGN (10x25, the 100 test instances, all eight combinations)
  Each instance is rolled out greedily (argmax at every decision) in three batches:
    (i)   alone, a batch of one (p97_policy_anytime.PolicyRoller.rollout_batch, the code
          of the stored greedy arrays);
    (ii)  twice in one batch of two (PolicyRoller.new_env(2)), so the divisor is the
          instance's own maximum and only the batch size changes;
    (iii) together with a companion in one batch of two: the first test instance, in the
          sorted order, whose largest processing time equals the maximum over the test
          set, other than the instance itself, under the same combination. The divisor
          of the batch is then that maximum.
  (ii) and (iii) step the same network and the same greedy choice as (i) on a batch of
  two (greedy_batch below; PolicyRoller.rollout_batch rolls a greedy schedule out in a
  batch of one only). Case (i) is rolled out a second time through greedy_batch on a
  batch of one, which must give the same schedule (run refuses otherwise) and counts the
  decisions of the trajectory alone whose two most probable actions lie within
  TIE_TOL = 1e-6 in probability: at such a near tie a rounding difference of a batched
  forward pass can change the argmax although the inputs of the row are identical.
  Recorded per instance: the three makespans, the makespan of the second row of (ii)
  and of the companion in (iii), whether the machine assignments and completion times
  of (ii) and (iii) equal those of (i), the instance's own largest processing time, the
  divisor of each batch (pt_upper_bound of its environment), the largest absolute
  difference of the network's inputs at the first decision between (i) and (iii) (job,
  machine and pair features), which shows that (iii) changes what the policy sees, and
  the near ties and the smallest top-two gap along (i).

OUTCOMES (infer)
  Per combination and in total: the instances whose own largest processing time is
  below the maximum, the cases (instance and combination) with identical makespans
  across (i), (ii) and (iii), across (i) and (ii) (the batch size alone), and across (i)
  and (iii) for the instances below the maximum (divisor and batch) and at it (batch and
  companion, same divisor); the cases with identical schedules; where makespans differ,
  the largest relative difference 100 |x - x_(i)| / x_(i) over (ii) and (iii); the cases
  below the maximum whose first-decision input differs between (i) and (iii); the cases
  with a near tie along (i), and of the cases that differ, those with one. The signed
  differences d_pair = 100 (x_(iii) - x_(i)) / x_(i) and d_twin = 100 (x_(ii) - x_(i)) /
  x_(i), separately for the cases below the maximum and at it: the mean with a 95%
  percentile bootstrap over instances (stats_family.instance_boot, 20,000 resamples,
  fresh default_rng(7); in the total over combinations an instance brings its eight
  combinations), the mean absolute difference, and the shares of cases shorter than,
  equal to and longer than the instance alone. The
  distribution of the largest processing time over the test instances of 10x25,
  50x25n30 and 80x25n30, which needs no policy. Outputs: results/divisor_probe.json,
  results/divisor_probe.md and manuscript/macros_divisor.tex (prefix Divp), in place, or
  under --out with --partial.

USAGE (repository root; CPU only; PACT_ENV=exact)
  PACT_ENV=exact python scripts/p104_divisor_probe.py plan  [--parts K]
  PACT_ENV=exact python scripts/p104_divisor_probe.py run   [--seeds 301-305] [--part i --parts K]
  PACT_ENV=exact python scripts/p104_divisor_probe.py infer [--seeds 301-305]
  Development (a stored run of another environment): run --model <model> --development
  --root <folder> [--n 3] [--combos N,LSW], then infer --partial --model <model> --root
  <folder> --out <folder> [--n 3] [--combos N,LSW].
"""
import argparse
import glob
import json
import os
import re
import sys
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(ROOT_DIR, 'scripts')
sys.path.insert(0, ROOT_DIR)
if SCRIPTS_DIR not in sys.path:
    sys.path.append(SCRIPTS_DIR)

import arm_registry as REG  # noqa: E402  (removes --pact-* options from sys.argv)

_ARGV = sys.argv[1:]
# params parses argv at import time and EnvState reads the device then: CPU only
sys.argv = sys.argv[:1] + ['--device', 'cpu']

import numpy as np  # noqa: E402

import p101_mask_controls as P101  # noqa: E402

SIZE = '10x25'
R8 = P101.R8
DIST_SIZES = tuple(P101.SIZES)           # 10x25, 50x25n30, 80x25n30
DEFAULT_ROOT = 'test_results/FAMILY/divisor_p104'
OUT_FILES = {'json': 'results/divisor_probe.json', 'md': 'results/divisor_probe.md',
             'macros': 'manuscript/macros_divisor.tex'}
GENERATOR = 'scripts/p104_divisor_probe.py infer'
PREFIX = 'Divp'
SEEDS = (301, 302, 303, 304, 305)
CASES = ('alone', 'twin', 'pair')
TIE_TOL = 1e-6          # two most probable actions this close: a rounding difference of a
#                         batched forward pass can change the greedy choice
# planning cost: process CPU seconds per instance and combination (three greedy rollouts,
# two batches of two, and the load of the checkpoint), from the smoke cells
COST_S = 2.3
COST_NOTE = 'smoke cells N and LSW, ten instances each, 2026-11-13, one shared core'
DIVISOR_LINES = ('fjsp_env_same_op_nums.py, set_initial_data: self.pt_upper_bound = '
                 'np.max(self.op_pt) over the batch [E, N, M], then op_pt and op_lag divided '
                 'by (pt_upper_bound - pt_lower_bound + 1e-8)',
                 'constraint_family/fjsp_env_family.py, set_initial_data: raw_pt = '
                 'np.array(op_pt_list); slope = 1.0 / (raw_pt.max() - raw_pt.min() + 1e-8), '
                 'which scales setups, windows, horizon and release dates')

# signed differences against the instance alone: case (iii) and case (ii), and the two
# groups of cases by the instance's own largest processing time
SIGNED = {'pair': 'with the companion (iii)', 'twin': 'twice in one batch (ii)'}
GROUPS = {'below': 'below the test-set maximum', 'at_max': 'at the test-set maximum'}
SIGNED_WORDS = {'pair': 'Pair', 'twin': 'Twin'}
GROUP_WORDS = {'below': 'Below', 'at_max': 'AtMax'}

Refused = P101.Refused
refuse = P101.refuse


# ---------------------------------------------------------------------------
# instances and the companion
# ---------------------------------------------------------------------------

def own_max(size):
    """{instance name: largest processing time} of the test instances of a size (the
    base instances of the stored descriptors, read as load_cell reads them)."""
    from ppvc_instance_generator import load_instance
    out = {}
    for name in P101.names_of(size):
        _, pt, _ = load_instance(os.path.join(ROOT_DIR, P101.SIZES[size]['base'], name))
        out[name] = float(np.max(pt))
    return out


def companion_of(name, pts):
    """The first test instance in sorted order whose largest processing time equals the
    maximum over the test set, other than the instance itself."""
    vmax = max(pts.values())
    for nm in sorted(pts):
        if pts[nm] == vmax and nm != name:
            return nm
    sys.exit('REFUSING: the test set holds no second instance at its largest processing time')


def distribution():
    """Largest processing time over the test instances of each size: counts per value,
    minimum, maximum, the count at the maximum and below it."""
    out = {}
    for size in DIST_SIZES:
        pts = own_max(size)
        v = np.array(list(pts.values()))
        vals, cnt = np.unique(v, return_counts=True)
        out[size] = dict(n=int(v.size), min=float(v.min()), max=float(v.max()),
                         at_max=int((v == v.max()).sum()), below_max=int((v < v.max()).sum()),
                         counts={f'{x:g}': int(c) for x, c in zip(vals, cnt)})
    return out


# ---------------------------------------------------------------------------
# run selection
# ---------------------------------------------------------------------------

def require_exact():
    if REG.env() != 'exact':
        sys.exit(f'REFUSING: PACT_ENV={REG.env()}; the probe runs in the exact-mask '
                 f'environment (set PACT_ENV=exact)')
    P101.check_mask_options()


def selected(seeds):
    """(seed, model, scores) of the run the selection rule picks among the exact-mask
    runs of the released recipe (arm_registry.select_seed)."""
    try:
        s, sc = REG.select_seed('released', seed_list=tuple(seeds))
    except REG.MissingResult as e:
        raise Refused(f'refused: the selected run among seeds {",".join(map(str, seeds))} '
                      f'cannot be determined: {e}')
    return int(s), REG.PREFIX + REG.model('released', None, s, 'exact'), \
        {str(k): {'score': float(v[0]), 'update': int(v[1])} for k, v in sc.items()}


def resolve(args):
    """The run of the probe: dict(run=resolve_run record, seed, development, selection)."""
    P97 = P101._p97()
    if args.model:
        if not args.development:
            sys.exit('REFUSING: --model names a run other than the selected one; add '
                     '--development (its cells are marked and kept out of the paper)')
        run = P97.resolve_run(args.model, development=True)
        seed, sel = P101.model_seed(args.model), dict(rule='named by --model (development)')
    else:
        seed, model, scores = selected(args.seeds)
        run = P97.resolve_run(model)
        sel = dict(rule='arm_registry.select_seed over the exact-mask runs of the released '
                        'recipe: the lowest validation score of the kept checkpoint, ties to '
                        'the lower seed', seeds=list(args.seeds), scores=scores)
    if run['mask'] != P101.MASK:
        sys.exit(f'REFUSING: the run is rolled out with {run["mask"]}, the probe needs '
                 f'{P101.MASK}')
    return dict(run=run, seed=seed, development=bool(args.model), selection=sel)


# ---------------------------------------------------------------------------
# rollouts
# ---------------------------------------------------------------------------

def roller_of(run, combo, name):
    P97 = P101._p97()
    r = P97.PolicyRoller(dict(run, size=SIZE, combination=combo,
                              instance=P101.instance_index(name), first_batch=1))
    if r.n_param != P101.EXPECT_PARAMS:
        sys.exit(f'REFUSING: {run["model"]} has {r.n_param} parameters, expected '
                 f'{P101.EXPECT_PARAMS}')
    return r


def check_env(env):
    for k, v in P101.MASK.items():
        if env.mask_setting(k) != v:
            raise RuntimeError(f'the environment reads {k}={env.mask_setting(k)}')
    if not env._first_mask:
        raise RuntimeError('the exact first-decision mask was not applied')
    if abs(float(env.pt_lower_bound)) > 1e-12:
        raise RuntimeError('the processing-time normalization is not a pure slope')
    return float(env.pt_upper_bound)


def pair_env(r, c):
    """One environment of the instance of roller r and the instance of roller c, built
    as PolicyRoller.new_env builds its copies (options, envelope, type lists, token)."""
    if r.pt.shape != c.pt.shape or not np.array_equal(r.jl, c.jl):
        sys.exit('REFUSING: the companion has another number of jobs, operations or machines')
    env = r.FJSPEnvFamily(r.jl.shape[0], r.M, family_mode=True, mask_mode=r.mm,
                          pair_extra=r.pe, first_mask=r.fm, mask_ties=r.mt)
    env.delta_mode = r.dm
    env.reward_delta_env = (r.dm != 'zero')
    env.set_initial_data([r.jl, c.jl], [r.pt, c.pt], descriptor_list=[r.desc, c.desc],
                         op_type_list=None if r.otl is None else r.otl + c.otl,
                         mch_type_list=None if r.mtl is None else r.mtl + c.mtl)
    env._attach_token()
    return env


def greedy_batch(r, env):
    """Greedy rollout of every row of env with the network of roller r: the steps of
    PolicyRoller.rollout_batch with greedy=True (argmax at every decision), for a batch
    of any size. Returns makespans [B], machine per operation [B, N], completion times
    [B, N], and per row the number of decisions whose two most probable actions differ
    in probability by at most TIE_TOL and the smallest such difference over the
    decisions (the decisions at which a rounding difference can change the argmax)."""
    B = int(env.number_of_envs)
    state = env.state
    rows = np.arange(B)
    amch = np.full((B, r.N), -1, dtype=int)
    ties, gap = np.zeros(B, dtype=np.int64), np.full(B, np.inf)
    steps = 0
    while not env.done().all():
        pi = r._forward(state)
        top = r.torch.topk(pi, 2, dim=1).values.cpu().numpy()
        g = top[:, 0] - top[:, 1]
        ties += (g <= TIE_TOL)
        gap = np.minimum(gap, g)
        a = r.greedy_select_action(pi).cpu().numpy()
        amch[rows, env.candidate[rows, a // r.M]] = a % r.M
        state, _, _ = env.step(a)
        steps += 1
    if steps != r.N:
        raise RuntimeError(f'{steps} decisions for {r.N} operations')
    ms = np.asarray(env.current_makespan, dtype=float).reshape(-1)[:B].copy()
    ct = np.stack([np.asarray(env.true_op_ct[k], dtype=float) for k in range(B)])
    return ms, amch, ct, ties, gap


def input_diff(s1, s2):
    """Largest absolute difference of the job, machine and pair features of row 0 of two
    first states."""
    out = 0.0
    for k in ('fea_j_tensor', 'fea_m_tensor', 'fea_pairs_tensor'):
        a, b = getattr(s1, k, None), getattr(s2, k, None)
        if a is None or b is None:
            continue
        out = max(out, float((a[0] - b[0]).abs().max()))
    return out


def probe_instance(r, c):
    """The three cases of one instance (roller r) with its companion (roller c)."""
    env1 = r._env                           # the environment rollout_batch steps for (i)
    d1 = check_env(env1)
    env3 = pair_env(r, c)
    d3 = check_env(env3)
    diff = input_diff(env1.state, env3.state)          # before either is stepped
    m1, a1, t1 = r.rollout_batch(True, None, 1)
    # (i) again through the loop of (ii) and (iii): the same schedule, and the near ties
    # along the trajectory alone
    env1b = r.new_env(1)
    check_env(env1b)
    m1b, a1b, t1b, tie1, gap1 = greedy_batch(r, env1b)
    if not (m1b[0] == m1[0] and np.array_equal(a1b, a1) and np.array_equal(t1b, t1)):
        sys.exit('REFUSING: the loop of this script and PolicyRoller.rollout_batch give '
                 'different schedules for a batch of one')
    env2 = r.new_env(2)
    d2 = check_env(env2)
    m2, a2, t2, _, _ = greedy_batch(r, env2)
    m3, a3, t3, _, _ = greedy_batch(r, env3)
    sched = [(r, a1[0], t1[0], m1[0]), (r, a2[0], t2[0], m2[0]), (r, a2[1], t2[1], m2[1]),
             (r, a3[0], t3[0], m3[0]), (c, a3[1], t3[1], m3[1])]
    for k, (who, am, ct, ms) in enumerate(sched):
        v = P101.validate(who.jl, who.pt, who.desc, am, ct, ms)
        if v:
            sys.exit(f'REFUSING: schedule {k} of the probe is infeasible: {v[:2]}')
    return dict(ms_alone=float(m1[0]), ms_twin=float(m2[0]), ms_twin_copy=float(m2[1]),
                ms_pair=float(m3[0]), ms_pair_companion=float(m3[1]),
                same_twin=bool(np.array_equal(a2[0], a1[0]) and np.array_equal(t2[0], t1[0])),
                same_pair=bool(np.array_equal(a3[0], a1[0]) and np.array_equal(t3[0], t1[0])),
                own_max=float(np.max(r.pt)), div_alone=d1, div_twin=d2, div_pair=d3,
                input_diff=diff, ties_alone=int(tie1[0]), min_gap_alone=float(gap1[0]))


# ---------------------------------------------------------------------------
# cells
# ---------------------------------------------------------------------------

def label_of(model):
    return f'divisor_{model}'


def design(combo, spec):
    run = spec['run']
    return dict(kind='divisor', size=SIZE, combo=combo, mask=P101.MASK,
                data=P101.SIZES[SIZE]['data'], base=P101.SIZES[SIZE]['base'],
                model=run['model'],
                checkpoint_sha256=P101.sha256_file(os.path.join(ROOT_DIR, run['checkpoint'])),
                trained_seed=int(spec['seed']), development=bool(spec['development']),
                cases='(i) alone, a batch of one; (ii) twice in one batch; (iii) with the '
                      'companion in one batch of two',
                companion='the first test instance in sorted order whose largest processing '
                          'time equals the maximum over the test set, other than the '
                          'instance itself',
                greedy='common_utils.greedy_select_action at every decision; (i) '
                       'p97_policy_anytime.PolicyRoller.rollout_batch, (ii) and (iii) the '
                       'same steps on a batch of two')


def run_combo(root, combo, names, spec, pts, force):
    run = spec['run']
    label = label_of(run['model'])
    want = design(combo, spec)
    st = P101.cell_state(root, SIZE, combo, label, want, names)
    if st == 'done' and not force:
        return 'skip', 0.0
    if st == 'other' and not force:
        sys.exit(f'REFUSING: {P101.cell_paths(root, SIZE, combo, label)[1]} holds another '
                 f'design, run or instance list; use --force or another --root')
    t0, c0 = time.perf_counter(), time.process_time()
    stored = P101.greedy_references(root, SIZE, combo, run['model'], names,
                                    stop=lambda m: sys.exit(f'REFUSING: {m}'))
    full = P101.names_of(SIZE)
    rows, comps, cache = [], [], {}
    for i, name in enumerate(names):
        cn = companion_of(name, pts)
        if cn not in cache:
            cache.clear()
            cache[cn] = roller_of(run, combo, cn)
        r = roller_of(run, combo, name)
        x = probe_instance(r, cache[cn])
        if x['own_max'] != pts[name]:
            sys.exit(f'REFUSING: {name} has largest processing time {x["own_max"]}, the test '
                     f'set read {pts[name]}')
        for src, a in stored.items():
            if a[i] != x['ms_alone']:
                sys.exit(f'REFUSING: case (i) of {combo}/{name} has makespan {x["ms_alone"]}, '
                         f'{src} holds {a[i]}')
        rows.append(x)
        comps.append(full.index(cn))
    arr = {k: np.array([x[k] for x in rows]) for k in rows[0]}
    arr['companion'] = np.array(comps, dtype=np.int64)
    npz, js = P101.cell_paths(root, SIZE, combo, label)
    P101.save_npz_atomic(npz, **arr)
    rec = P101.run_record(want, names, t0, c0)
    rec['run'] = {k: v for k, v in run.items() if k != 'mask'}
    rec['selection'] = spec['selection']
    rec['companions'] = {nm: full[c] for nm, c in zip(names, comps)}
    rec['greedy_check'] = ({src: 'equal' for src in stored} if stored else
                           'no stored greedy array of this run')
    P101.write_json_atomic(js, rec)
    return 'done', rec['cpu_s']


def parts_of(combos, parts):
    """Combinations of each part, round robin in the order of the list."""
    return [[c for j, c in enumerate(combos) if j % parts == p] for p in range(parts)]


def plan(args):
    n = args.n or P101.SIZES[SIZE]['n']
    total = COST_S * n * len(args.combos)
    groups = parts_of(args.combos, args.parts)
    print(f'[p104 plan] {len(args.combos)} cells of {n} instances, estimated '
          f'{total / 60:.1f} core-minutes of CPU time (COST_S {COST_S:g} s per instance and '
          f'combination; {COST_NOTE}), {args.parts} parts of '
          f'{min(len(g) for g in groups) * COST_S * n / 60:.1f} to '
          f'{max(len(g) for g in groups) * COST_S * n / 60:.1f} min')
    py = sys.executable
    for i in range(args.parts):
        print(f'CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PACT_ENV=exact '
              f'taskset -c <CORE_{i}> {py} -B scripts/p104_divisor_probe.py run --seeds '
              f'{",".join(map(str, args.seeds))} --part {i} --parts {args.parts}')
    return dict(cells=len(args.combos), core_min=total / 60)


def run(args):
    require_exact()
    P101._cpu_torch()
    spec = resolve(args)
    names = P101.names_of(SIZE, args.n)
    pts = own_max(SIZE)
    combos = args.combos if args.part is None else parts_of(args.combos, args.parts)[args.part]
    print(f'[p104 run] {spec["run"]["model"]} (seed {spec["seed"]}'
          + (', development' if spec['development'] else ', selected run') + f'); '
          f'{len(names)} instances, combinations {",".join(combos)}', flush=True)
    t_all = time.perf_counter()
    for k, combo in enumerate(combos):
        st, c = run_combo(args.root, combo, names, spec, pts, args.force)
        print(f'[p104 run {k + 1}/{len(combos)}] {combo:4s} {st} cpu={c:7.1f}s '
              f'elapsed={time.perf_counter() - t_all:7.0f}s', flush=True)


# ---------------------------------------------------------------------------
# infer
# ---------------------------------------------------------------------------

def combo_stats(z, vmax):
    """Counts of one combination (arrays z of a cell)."""
    alone, twin, pair = z['ms_alone'], z['ms_twin'], z['ms_pair']
    below = z['own_max'] < vmax
    same = (alone == twin) & (alone == pair)
    sched = z['same_twin'].astype(bool) & z['same_pair'].astype(bool)
    rel = np.maximum(np.abs(twin - alone), np.abs(pair - alone)) / alone * 100.0
    tie = z['ties_alone'] > 0
    return dict(n=int(alone.size), below=int(below.sum()), same=int(same.sum()),
                same_below=int((same & below).sum()), same_schedule=int(sched.sum()),
                same_twin=int((alone == twin).sum()),
                same_pair_below=int(((alone == pair) & below).sum()),
                same_pair_at_max=int(((alone == pair) & ~below).sum()),
                differ=int((~same).sum()), differ_below=int((~same & below).sum()),
                max_rel_diff=float(rel.max()) if (~same).any() else 0.0,
                input_changed_below=int((below & (z['input_diff'] > 0)).sum()),
                max_input_diff=float(z['input_diff'].max()),
                twin_copy_equal=int((z['ms_twin_copy'] == twin).sum()),
                tie_cases=int(tie.sum()), differ_with_tie=int((~same & tie).sum()),
                mean_ms_alone=float(alone.mean()))


def signed_diff(z, case):
    """100 (x - x_alone) / x_alone per instance, x the makespan of case (ii) or (iii)."""
    return 100.0 * (np.asarray(z[f'ms_{case}'], float) - np.asarray(z['ms_alone'], float)) \
        / np.asarray(z['ms_alone'], float)


def signed_stats(d):
    """Mean of the signed differences d ([instances] or [instances, combinations]) with
    the 95% percentile bootstrap over instances of stats_family.instance_boot (an instance
    brings all its combinations; B and the seed of the instance-level intervals of p101),
    the mean absolute difference, and the cases shorter than, equal to and longer than
    the instance alone. None for an empty group."""
    d = np.asarray(d, dtype=float)
    if d.size == 0:
        return None
    ib = P101._stats_family().instance_boot(d, B=P101.B, seed=P101.CI_SEED)
    flat = d.ravel()
    c = dict(shorter=int((flat < 0).sum()), equal=int((flat == 0).sum()),
             longer=int((flat > 0).sum()))
    return dict(mean=ib['mean'], lo=ib['lo'], hi=ib['hi'], instances=int(ib['n']),
                combinations=int(ib['c']), cases=int(flat.size),
                mean_abs=float(np.abs(flat).mean()), **c,
                **{f'{k}_pct': 100.0 * v / flat.size for k, v in c.items()})


def signed_block(D, below):
    """{case: {group: signed_stats}} of D {case: [instances] or [instances,
    combinations]}, with `below` the instances below the test-set maximum."""
    masks = {'below': below, 'at_max': ~below}
    return {case: {g: signed_stats(D[case][m]) for g, m in masks.items()} for case in SIGNED}


def check_cell(z, names, vmax, combo, pts):
    """The premises of the design: the stored largest processing times are those of the
    instances, the divisors of (i) and (ii) are the instance's own maximum, and the
    divisor of (iii) is the maximum of the test set."""
    own = np.array([pts[nm] for nm in names])
    if not np.array_equal(z['own_max'], own):
        refuse(f'{combo}: the stored largest processing times differ from the instances')
    for k in ('div_alone', 'div_twin'):
        if not np.array_equal(z[k], own):
            refuse(f'{combo}: the divisor of case {k[4:]} is not the instance\'s own maximum')
    if not np.all(z['div_pair'] == vmax):
        refuse(f'{combo}: the divisor of case (iii) is not the maximum of the test set')


def infer(args):
    require_exact()
    partial = bool(args.partial)
    if args.n and not partial:
        refuse('--n (a subset of the instances) needs --partial')
    if args.model and not partial:
        refuse('--model (a run other than the selected one) needs --partial')
    if not partial and args.combos != list(R8):
        refuse('every combination is required; a subset needs --partial')
    if args.out is None:
        if partial:
            refuse('--partial writes only to --out <folder>')
        out_root = ROOT_DIR
    else:
        out_root = os.path.abspath(args.out)
        P101.guard_out(out_root)
    paths = {k: os.path.join(out_root, v) for k, v in OUT_FILES.items()}
    for p in paths.values():                # a refusal leaves no stale output
        if os.path.exists(p):
            os.remove(p)
    if args.model:
        model, seed, sel = args.model, P101.model_seed(args.model), \
            dict(rule='named by --model (development)')
    else:
        seed, model, scores = selected(args.seeds)
        sel = dict(rule='arm_registry.select_seed over the exact-mask runs of the released '
                        'recipe', seeds=list(args.seeds), scores=scores)
    names = P101.names_of(SIZE, args.n)
    pts = own_max(SIZE)
    vmax = max(pts.values())
    res = {'generator': GENERATOR, 'partial': partial, 'model': model, 'seed': seed,
           'selection': sel, 'mask': P101.MASK, 'size': SIZE, 'n': len(names),
           'test_set_max': vmax, 'divisor_lines': list(DIVISOR_LINES),
           'cases': {'alone': 'a batch of one', 'twin': 'the instance twice in one batch',
                     'pair': 'the instance and its companion in one batch'},
           'companion': 'the first test instance in sorted order at the maximum of the test '
                        'set, other than the instance itself',
           'relative_difference': '100 |x - x_alone| / x_alone, the larger over twin and pair',
           'signed_difference': {
               'pair': '100 (x_pair - x_alone) / x_alone per case, negative = shorter with the '
                       'companion than alone',
               'twin': '100 (x_twin - x_alone) / x_alone per case, negative = shorter twice in '
                       'one batch than alone',
               'groups': GROUPS,
               'interval': f'95% percentile bootstrap over instances, '
                           f'stats_family.instance_boot ({P101.B} resamples, fresh '
                           f'default_rng({P101.CI_SEED})); in the total an instance brings '
                           f'all its combinations',
               'shares': 'cases shorter than, equal to and longer than the instance alone'},
           'combos': {}}
    ckpt = os.path.join(ROOT_DIR, 'trained_network', 'FAMILY', f'{model}.pth')
    diffs = {}
    for combo in args.combos:
        z, rec = P101.read_cell(args.root, SIZE, combo, label_of(model), names)
        dz = rec.get('design', {})
        if dz.get('model') != model or dz.get('kind') != 'divisor':
            refuse(f'{combo}: the cell records the run {dz.get("model")}, not {model}')
        if dz.get('development') and not partial:
            refuse(f'{combo}: the cell holds a development run; the paper needs the selected run')
        if os.path.exists(ckpt) and dz.get('checkpoint_sha256') != P101.sha256_file(ckpt):
            refuse(f'{combo}: the cell was written by another checkpoint of {model}')
        check_cell(z, names, vmax, combo, pts)
        stored = P101.greedy_references(args.root, SIZE, combo, model, names, stop=refuse)
        for src, a in stored.items():
            if not np.array_equal(a, z['ms_alone']):
                refuse(f'{combo}: case (i) differs from {src} on '
                       f'{int((a != z["ms_alone"]).sum())} instances')
        C = combo_stats(z, vmax)
        C['greedy_check'] = sorted(stored) or 'no stored greedy array of this run'
        diffs[combo] = {case: signed_diff(z, case) for case in SIGNED}
        C['signed'] = signed_block(diffs[combo], np.asarray(z['own_max']) < vmax)
        res['combos'][combo] = C
    T = {k: sum(C[k] for C in res['combos'].values())
         for k in ('n', 'below', 'same', 'same_below', 'same_schedule', 'same_twin',
                   'same_pair_below', 'same_pair_at_max', 'differ', 'differ_below',
                   'input_changed_below', 'twin_copy_equal', 'tie_cases', 'differ_with_tie')}
    T['max_rel_diff'] = max(C['max_rel_diff'] for C in res['combos'].values())
    T['max_input_diff'] = max(C['max_input_diff'] for C in res['combos'].values())
    T['instances_below'] = int(sum(pts[nm] < vmax for nm in names))
    below = np.array([pts[nm] < vmax for nm in names])
    T['signed'] = signed_block({case: np.stack([diffs[c][case] for c in args.combos], axis=1)
                                for case in SIGNED}, below)
    res['total'] = T
    res['distribution'] = distribution()
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
        print('wrote', paths[k])
    print(f'{len(names_m)} macros' + ('; PARTIAL' if partial else ''))
    return res


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------

SIZE_WORDS = {s: P101.SIZES[s]['word'] for s in DIST_SIZES}


def macro_lines(res):
    mark = 'PARTIAL ' if res['partial'] else ''
    T = res['total']
    L = [f'% Generated by {GENERATOR} -- do not hand-edit',
         f'% Batch composition and the greedy schedule of the selected run ({res["model"]}), '
         f'exact mask, {res["size"]} test set: each instance rolled out alone, twice in one '
         f'batch, and with a companion at the largest processing time of the test set.']
    if res['partial']:
        L.insert(1, '% PARTIAL: a subset of combinations or instances, or a development run; '
                    'not for the manuscript.')
    L.append(f'\\newcommand{{\\{PREFIX}InstN}}{{{res["n"]}}}'
             f'\\newcommand{{\\{PREFIX}CombN}}{{{len(res["combos"])}}}'
             f'\\newcommand{{\\{PREFIX}CaseN}}{{{T["n"]}}}'
             f'\\newcommand{{\\{PREFIX}SelSeed}}{{{res["seed"]}}}'
             f'\\newcommand{{\\{PREFIX}MaxPt}}{{{res["test_set_max"]:g}}}'
             f'  % {mark}test instances, combinations, cases (instance and combination), '
             f'training seed of the run, largest processing time of the test set (h)')
    L.append(f'\\newcommand{{\\{PREFIX}BelowN}}{{{T["instances_below"]}}}'
             f'\\newcommand{{\\{PREFIX}BelowCaseN}}{{{T["below"]}}}'
             f'\\newcommand{{\\{PREFIX}InputChangedN}}{{{T["input_changed_below"]}}}'
             f'  % {mark}instances whose own largest processing time is below the maximum of '
             f'the test set, the cases they give over the combinations, and of those cases '
             f'the number whose network input at the first decision differs between the '
             f'instance alone and with the companion')
    L.append(f'\\newcommand{{\\{PREFIX}SameN}}{{{T["same"]}}}'
             f'\\newcommand{{\\{PREFIX}SameBelowN}}{{{T["same_below"]}}}'
             f'\\newcommand{{\\{PREFIX}SameSchedN}}{{{T["same_schedule"]}}}'
             f'\\newcommand{{\\{PREFIX}DiffN}}{{{T["differ"]}}}'
             f'\\newcommand{{\\{PREFIX}MaxRelDiff}}{{{T["max_rel_diff"]:.2f}}}'
             f'  % {mark}cases with the same greedy makespan alone, twice in one batch and with '
             f'the companion; the same among the cases below the maximum; cases with the same '
             f'schedule (machines and completion times) in all three; cases whose makespans '
             f'differ; the largest relative difference of the makespan over those cases (%)')
    L.append(f'\\newcommand{{\\{PREFIX}SameTwinN}}{{{T["same_twin"]}}}'
             f'\\newcommand{{\\{PREFIX}SamePairBelowN}}{{{T["same_pair_below"]}}}'
             f'\\newcommand{{\\{PREFIX}SamePairAtMaxN}}{{{T["same_pair_at_max"]}}}'
             f'\\newcommand{{\\{PREFIX}AtMaxCaseN}}{{{T["n"] - T["below"]}}}'
             f'  % {mark}cases with the same makespan alone and twice in one batch (only the '
             f'batch size changes); cases below the maximum with the same makespan alone and '
             f'with the companion (divisor and batch change); the same for the cases at the '
             f'maximum (batch and companion change, the divisor does not), and the number of '
             f'cases at the maximum')
    L.append(f'\\newcommand{{\\{PREFIX}TieCaseN}}{{{T["tie_cases"]}}}'
             f'\\newcommand{{\\{PREFIX}DiffTieN}}{{{T["differ_with_tie"]}}}'
             f'  % {mark}cases whose greedy trajectory alone has a decision with the two most '
             f'probable actions within {TIE_TOL:g} in probability, and of the cases whose '
             f'makespans differ, the number with such a decision')
    for combo, C in res['combos'].items():
        L.append(f'\\newcommand{{\\{PREFIX}SameIn{combo}}}{{{C["same"]}}}'
                 f'\\newcommand{{\\{PREFIX}DiffIn{combo}}}{{{C["differ"]}}}'
                 f'\\newcommand{{\\{PREFIX}MaxRelDiffIn{combo}}}{{{C["max_rel_diff"]:.2f}}}'
                 f'  % {mark}{combo}: instances with the same greedy makespan in the three '
                 f'batches, instances whose makespans differ, largest relative difference (%)')
    for scope, C in [('', T)] + [(f'In{c}', C) for c, C in res['combos'].items()]:
        where = 'all combinations' if not scope else scope[2:]
        for case, cw in SIGNED_WORDS.items():
            for g, gw in GROUP_WORDS.items():
                X = C['signed'][case][g]
                if X is None:
                    continue
                a = f'{PREFIX}{cw}{gw}'
                L.append(f'\\newcommand{{\\{a}Mean{scope}}}{{{P101.f2(X["mean"])}}}'
                         f'\\newcommand{{\\{a}Lo{scope}}}{{{P101.f2(X["lo"])}}}'
                         f'\\newcommand{{\\{a}Hi{scope}}}{{{P101.f2(X["hi"])}}}'
                         f'\\newcommand{{\\{a}Abs{scope}}}{{{X["mean_abs"]:.2f}}}'
                         f'\\newcommand{{\\{a}Shorter{scope}}}{{{P101.u1(X["shorter_pct"])}}}'
                         f'\\newcommand{{\\{a}Equal{scope}}}{{{P101.u1(X["equal_pct"])}}}'
                         f'\\newcommand{{\\{a}Longer{scope}}}{{{P101.u1(X["longer_pct"])}}}'
                         f'  % {mark}{where}, the {X["cases"]} cases {GROUPS[g]} '
                         f'({X["instances"]} instances): signed difference of the makespan '
                         f'{SIGNED[case]} against alone, 100 (x - x_alone) / x_alone (%), '
                         f'negative = shorter than alone: mean and 95% interval (bootstrap over '
                         f'instances' + (', an instance with all its combinations' if not scope
                                         else '') +
                         f'), mean absolute difference (%), and the shares (%) of the cases '
                         f'shorter than, equal to and longer than alone')
    for size, D in res['distribution'].items():
        w = SIZE_WORDS[size]
        L.append(f'\\newcommand{{\\{PREFIX}PtN{w}}}{{{D["n"]}}}'
                 f'\\newcommand{{\\{PREFIX}PtMin{w}}}{{{D["min"]:g}}}'
                 f'\\newcommand{{\\{PREFIX}PtMax{w}}}{{{D["max"]:g}}}'
                 f'\\newcommand{{\\{PREFIX}PtAtMax{w}}}{{{D["at_max"]}}}'
                 f'\\newcommand{{\\{PREFIX}PtBelowMax{w}}}{{{D["below_max"]}}}'
                 f'  % {mark}{size}: test instances, smallest and largest value over them of '
                 f'the largest processing time of an instance (h), instances at that largest '
                 f'value and below it (counts per value: '
                 + ', '.join(f'{k} h {v}' for k, v in D['counts'].items()) + ')')
    return L


def check_macro_names(names, out_paths):
    """Every name is the prefix plus letters, unique, and absent from every other macro
    file of the repository's paper folders (the check of p101)."""
    bad = [x for x in names if not re.fullmatch(PREFIX + r'[A-Za-z]+', x)]
    if bad:
        refuse(f'macro names that are not {PREFIX} plus letters: {bad[:5]}')
    if len(names) != len(set(names)):
        refuse(f'duplicate macro names: {sorted({x for x in names if names.count(x) > 1})[:5]}')
    own = {os.path.abspath(out_paths['macros']),
           os.path.abspath(os.path.join(ROOT_DIR, OUT_FILES['macros']))}
    files = sorted(set(glob.glob(os.path.join(ROOT_DIR, 'paper*', 'macros*.tex'))
                       + glob.glob(os.path.join(ROOT_DIR, 'paper*', '*', 'macros*.tex'))
                       + glob.glob(os.path.join(ROOT_DIR, 'manuscript', 'sections_new', '*.tex'))
                       + glob.glob(os.path.join(ROOT_DIR, 'manuscript', 'supp_new', '*.tex'))))
    rx = re.compile(r'\\(?:newcommand|renewcommand|providecommand|def)\s*\*?\s*\{?\\([A-Za-z]+)')
    for f in files:
        if os.path.abspath(f) in own:
            continue
        with open(f, errors='replace') as fh:
            clash = sorted(x for x in rx.findall(fh.read()) if x.startswith(PREFIX))
        if clash:
            refuse(f'the prefix {PREFIX} is in use in {os.path.relpath(f, ROOT_DIR)}: '
                   f'{clash[:5]}')
    return len(files)


def markdown(res):
    T = res['total']
    L = [f'# Batch composition and the greedy schedule ({GENERATOR})', '']
    if res['partial']:
        L += ['PARTIAL: a subset of combinations or instances, or a development run. Not for '
              'the manuscript.', '']
    L += [f'Run {res["model"]} (training seed {res["seed"]}; {res["selection"]["rule"]}), '
          f'exact mask, {res["size"]} test set, {res["n"]} instances. Each instance is rolled '
          f'out greedily alone, twice in one batch, and with a companion whose largest '
          f'processing time is the maximum of the test set ({res["test_set_max"]:g} h), so '
          f'that the divisor of the time features is that maximum. The divisor: '
          + '; '.join(res['divisor_lines']) + '.', '',
          '| combination | instances | own maximum below the test-set maximum | same makespan '
          'in all three | same schedule in all three | same alone and twice | same alone and '
          'with the companion, below / at the maximum | makespans differ | largest relative '
          'difference (%) | input changed (below the maximum) | largest input difference | '
          f'near tie alone (gap <= {TIE_TOL:g}) | differ with a near tie |',
          '|' + '---|' * 13]
    for combo, C in list(res['combos'].items()) + [('total', T)]:
        L.append(f'| {combo} | {C["n"]} | {C["below"]} | {C["same"]} | {C["same_schedule"]} | '
                 f'{C["same_twin"]} | {C["same_pair_below"]} / {C["same_pair_at_max"]} | '
                 f'{C["differ"]} | {C["max_rel_diff"]:.2f} | {C["input_changed_below"]} | '
                 f'{C["max_input_diff"]:.4f} | {C["tie_cases"]} | {C["differ_with_tie"]} |')
    L += ['', 'Signed difference of the makespan against the instance alone, 100 (x - x_alone) '
          '/ x_alone (%), negative = shorter than alone: mean [95% bootstrap interval over '
          'instances; in the total an instance brings all its combinations], mean absolute '
          'difference, and the shares of the cases shorter / equal / longer than alone (%).',
          '', '| combination | cases of | cases | with the companion, mean [interval] | mean '
          'absolute | shorter / equal / longer | twice in one batch, mean [interval] | mean '
          'absolute | shorter / equal / longer |', '|' + '---|' * 9]
    for combo, C in list(res['combos'].items()) + [('total', T)]:
        for g in GROUPS:
            P, W = C['signed']['pair'][g], C['signed']['twin'][g]
            if P is None:
                L.append(f'| {combo} | {GROUPS[g]} | 0 | -- | -- | -- | -- | -- | -- |')
                continue
            L.append(f'| {combo} | {GROUPS[g]} | {P["cases"]} | '
                     + ' | '.join(f'{P101.f2(X["mean"])} [{P101.f2(X["lo"])}, '
                                  f'{P101.f2(X["hi"])}] | {X["mean_abs"]:.2f} | '
                                  f'{P101.u1(X["shorter_pct"])} / {P101.u1(X["equal_pct"])} / '
                                  f'{P101.u1(X["longer_pct"])}' for X in (P, W)) + ' |')
    L += ['', 'Largest processing time of an instance over the test instances (no policy):', '',
          '| size | instances | minimum (h) | maximum (h) | at the maximum | below it | counts '
          'per value |', '|' + '---|' * 7]
    for size, D in res['distribution'].items():
        L.append(f'| {size} | {D["n"]} | {D["min"]:g} | {D["max"]:g} | {D["at_max"]} | '
                 f'{D["below_max"]} | ' + ', '.join(f'{k} h: {v}' for k, v in D['counts'].items())
                 + ' |')
    L.append('')
    return L


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('phase', choices=['run', 'infer', 'plan'])
    ap.add_argument('--root', default=DEFAULT_ROOT, help='folder of the cells')
    ap.add_argument('--combos', default=','.join(R8))
    ap.add_argument('--seeds', default='301-305',
                    help='trained runs among which the selection rule picks the run')
    ap.add_argument('--model', default=None,
                    help='development: a run other than the selected one (with --development)')
    ap.add_argument('--development', action='store_true')
    ap.add_argument('--n', type=int, default=0, help='first n instances (smoke only)')
    ap.add_argument('--part', type=int, default=None)
    ap.add_argument('--parts', type=int, default=1)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--partial', action='store_true',
                    help='infer: a subset of combinations or instances, or --model, under --out')
    ap.add_argument('--out', default=None, help='infer: root of the outputs')
    args = ap.parse_args(_ARGV)
    args.combos = P101.parse_list(args.combos, R8, 'combinations')
    args.seeds = list(REG.parse_seeds(args.seeds))
    for s in args.seeds:
        if s not in REG.PLANNED_SEEDS['exact']:
            sys.exit(f'REFUSING: seed {s} is not planned in the exact environment')
    args.n = args.n or None
    if args.part is not None and not 0 <= args.part < args.parts:
        sys.exit(f'--part {args.part} outside 0..{args.parts - 1}')
    os.chdir(ROOT_DIR)
    {'run': run, 'infer': infer, 'plan': plan}[args.phase](args)


if __name__ == '__main__':
    main()
