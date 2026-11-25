r"""P102: two sensitivity levels at the production sizes, evaluation only.

Subcommands build, plan, evaluate, infer and classes. CPU only, exact mask.

LAUNCH NOTE (the four questions, answered before the run)

1. Purpose. The stored unavailability windows lie on the job-chain horizon H of each
   instance (213 h at 50 modules and 217 h at 80 modules on average), while a
   production schedule runs 360 to 890 h, so the windows act only on its first part.
   Level win_full draws
   the windows again on a horizon equal to the schedule length and re-measures the
   comparisons of the sensitivity study (scripts/p87_sensitivity.py) at 50 and 80
   modules. The numbers land in the sensitivity table of the supplement (two rows)
   and in one sentence of the benchmark section and of the limitations.
2. Expected result. With windows over the whole schedule the verdicts of the default
   level hold: the policy stays within the equivalence margin of the specialists and
   ahead of the best dispatching rule on W, LW, SW and LSW. If a verdict changes, the
   production-size claims for the window rule hold only for the coverage of the
   stored calendars, and the paper states that coverage as a number next to each
   claim.
3. Contamination. The frozen folders data/FAMILY/50x25n30 and 80x25n30, the base
   instances and the stored result folders are read only. New descriptors and results
   go under --root (default test_results/FAMILY/sens_p102), a folder no existing
   script reads (tests/test_p101_mask_controls.py checks every glob over
   test_results/FAMILY). build refuses to overwrite a file whose content differs.
   Rollouts are deterministic, so results are not timing measurements and may share
   cores; the recorded times are planning figures.
4. Data accuracy. The default level is the stored descriptors, evaluated by the same
   command as the new level. infer refuses when a default-level rule array, or a
   default-level array of the released policy, differs from the stored exact-mask
   array of the same method (test_results/FAMILY/<size>_xmask, where it exists).
   build checks that the generator call with the stored seed
   reproduces every stored window field, and replays SPT on the first --verify_n
   instances of each window-free combination to confirm the stored makespans it uses
   as horizons (scripts/p101_mask_controls.py infer checks all of them).

LEVELS (sizes 50x25n30 and 80x25n30, 30 instances each; combinations W, LW, SW, LSW)

  default   the stored descriptors of data/FAMILY/<size>/desc/<R>.
  win_full  per instance and combination R, the stored descriptor with lags, setups,
            classes and every other field unchanged, and the windows drawn again by
            descriptor.draw_windows, the generator's rule, with its default density
            range U[0.05, 0.15] of the horizon and durations U{2..8} h, on the horizon
            H' = the makespan of the stored SPT schedule of the same instance under
            the same combination without windows (N for W, L for LW, S for SW, LS for
            LSW), read from test_results/FAMILY/<size>_xmask/Result_SPT_<R0>.npy. The
            descriptor's horizon field becomes H', so the outage density of the
            constraint token keeps the training range, while the outage count grows
            with the horizon. Window stream of instance i: numpy
            default_rng(SeedSequence([20261113 + i, 13])), the generator's window tag
            13 with a new seed; the stream does not depend on the combination, so the
            calendars of W, LW, SW and LSW differ only through H'.
  five classes (training size and both production sizes, S, LS, SW, LSW) is the
            level of scripts/p103_five_classes.py, which sets the class count for its
            own process. The shared modules hold four classes, and subcommand classes
            of this script reports the sites that fix that count.

ARMS (as scripts/p87_sensitivity.py, exact-mask names of scripts/arm_registry.py)
  pact    released policy xmask-v1, training seeds --seeds (default 301-305)
  spec    specialist of the combination, xmask-solo-<R>, the same seeds
  single  single-rule control xmask-ctl, the same seeds
  rules   FIFO, MOR, SPT, MWKR restricted to the legal pairs (common_utils.
          masked_rule_action) and ECT, SSPT, LLF (scripts/p72_aware_rules.py), each
          with np.random.seed(50) after the data are set, as the stored arrays
  Every policy run is loaded with the configuration of its training snapshot and its
  parameter count is asserted (32786).

OUTCOMES (infer)
  Rules (available today): mean makespan of each rule; the best of the four classical
  rules and the best of all seven, per instance; the change of SPT and of the best
  classical rule from the default level to win_full, 100 (x_full - x_default) /
  x_default per instance with a 95% bootstrap over instances (20,000 resamples, fresh
  default_rng(7)); the window coverage of the SPT schedule (last window end over the
  makespan, window time before the makespan over machines times makespan).
  Policy arms (once trained): per level, size and combination, and pooled over the
  four combinations of a size, the gap of the policy to the specialist and of the
  control to the specialist with verdicts at the margins 0.5% and 1%, and of the
  policy to the best classical rule and to the best of the seven rules with the
  combined verdict (Holm over the four combinations of a size); d = 100 (a - b) / b,
  seed s against seed s, crossed hierarchical bootstrap over training runs and base
  instances (stats_family.hier_boot, B = 20000, fresh default_rng(11)), the
  estimators of p87 infer. The shift of each gap from the default level to win_full,
  paired by seed and instance, with the same bootstrap; the verdicts that change.
  Outputs: results/scale_sensitivity.json and .md, manuscript/macros_scalesens.tex
  (prefix Psens), in place for --arms rules or for every arm, otherwise under --out.

USAGE (repository root; CPU only)
  python scripts/p102_scale_sensitivity.py build    [--verify_n 3]
  python scripts/p102_scale_sensitivity.py plan     --parts K [--arms rules]
  python scripts/p102_scale_sensitivity.py evaluate --arms rules [--part i --parts K]
  python scripts/p102_scale_sensitivity.py evaluate --arms pact,spec,single [--seeds 301-305]
  python scripts/p102_scale_sensitivity.py infer    --arms rules
  python scripts/p102_scale_sensitivity.py infer    [--seeds 301-305]       (every arm)
  python scripts/p102_scale_sensitivity.py classes                          (five classes)
  Smoke: --root <folder> --n 2, and infer --partial --out <folder>.
"""
import argparse
import glob
import hashlib
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
sys.argv = sys.argv[:1] + ['--device', 'cpu']

import numpy as np  # noqa: E402

import p101_mask_controls as P101  # noqa: E402

# ---------------------------------------------------------------------------
# design constants
# ---------------------------------------------------------------------------

SIZES = ('50x25n30', '80x25n30')
LEVELS = ('default', 'win_full')
WIN_COMBOS = ('W', 'LW', 'SW', 'LSW')
NO_WINDOW = {'W': 'N', 'LW': 'L', 'SW': 'S', 'LSW': 'LS'}
WINDOW_SEED0 = 20261113
WINDOW_TAG = 13                     # descriptor.make_descriptor's window stream tag
RULES4 = ('FIFO', 'MOR', 'SPT', 'MWKR')
AWARE = ('ECT', 'SSPT', 'LLF')
RULES7 = RULES4 + AWARE
POLICY_ARMS = ('pact', 'spec', 'single')
ARM_CHOICES = POLICY_ARMS + ('rules',)
SEEDS = (301, 302, 303, 304, 305)
DEFAULT_ROOT = 'test_results/FAMILY/sens_p102'
OUT_FILES = {'json': 'results/scale_sensitivity.json', 'md': 'results/scale_sensitivity.md',
             'macros': 'manuscript/macros_scalesens.tex'}
GENERATOR = 'scripts/p102_scale_sensitivity.py infer'
PREFIX = 'Psens'
MARGINS = (0.5, 1.0)
B = 20000
HIER_SEED, INST_SEED, ALPHA = 11, 7, 0.05
LEVEL_WORDS = {'default': 'Def', 'win_full': 'Full'}
SIZE_WORDS = {'50x25n30': 'Fifty', '80x25n30': 'Eighty'}
MW = {0.5: 'Half', 1.0: 'One'}
# comparison key -> (first-named method, second, macro word)
COMPARISONS = {'pol_spec': ('policy', 'specialist', 'PolSpec'),
               'ctl_spec': ('single-rule control', 'specialist', 'CtlSpec'),
               'pol_rule': ('policy', 'best classical rule', 'PolRule'),
               'pol_rule7': ('policy', 'best of the seven rules', 'PolRuleSeven')}
# planning cost, CPU seconds per instance. Rules are the process CPU times of the LSW
# smoke cells of this script (2026-11-13, shared core); policies are the exact
# campaign's rollout times (scripts/eval_newseeds_plan.py SPEED).
COST_S = {'50x25n30': {'rule': 2.0, 'policy': 16.3},
          '80x25n30': {'rule': 4.5, 'policy': 33.3}}


def window_ranges():
    """Density and duration ranges of the generator's rule (draw_windows defaults, as
    check C4 reads them)."""
    from constraint_family import checks
    return tuple(checks.WINDOW_DENSITY_RANGE), tuple(checks.WINDOW_DUR_RANGE)


def window_seed(name):
    return WINDOW_SEED0 + P101.instance_index(name)


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------

sha256_file = P101.sha256_file
write_json_atomic = P101.write_json_atomic
jsonable = P101.jsonable


def write_once(path, text):
    """Write text, or verify that an existing file holds exactly that text."""
    if os.path.exists(path):
        with open(path) as f:
            if f.read() != text:
                sys.exit(f'REFUSING: {path} exists with different content; the build is '
                         f'deterministic, so the design or the inputs changed. Use a new '
                         f'--root.')
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        f.write(text)
    os.replace(tmp, path)


def stored_desc_path(size, combo, name):
    return os.path.join(ROOT_DIR, P101.SIZES[size]['data'], 'desc', combo,
                        f'{name}.desc.json')


def level_desc_path(root, level, size, combo, name):
    if level == 'default':
        return stored_desc_path(size, combo, name)
    return os.path.join(root, 'data', level, size, 'desc', combo, f'{name}.desc.json')


def stored_spt(size, combo):
    """Makespans of the stored SPT schedules of one combination, in instance order."""
    p = os.path.join(ROOT_DIR, P101.RESULTS_ROOT, f'{size}_xmask', f'Result_SPT_{combo}.npy')
    a = np.load(p)
    return (a[:, 0] if a.ndim > 1 else a).astype(float), p


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def full_windows(horizon, n_machines, seed):
    """Windows of level win_full: draw_windows on the new horizon with the generator's
    ranges, from the stream of the instance's new seed."""
    from constraint_family.descriptor import draw_windows
    dens, dur = window_ranges()
    rng = np.random.default_rng(np.random.SeedSequence([seed, WINDOW_TAG]))
    return draw_windows(n_machines, horizon, rng, density_range=dens, dur_range=dur)


def full_descriptor(size, combo, name, horizon):
    """(ConstraintDescriptor, record) of level win_full for one cell entry."""
    from constraint_family.descriptor import ConstraintDescriptor
    with open(stored_desc_path(size, combo, name)) as f:
        rec = json.load(f)
    desc = ConstraintDescriptor.from_json(rec, setup_convention='attached')
    n_m = len(desc.windows)
    desc.windows = full_windows(int(horizon), n_m, window_seed(name))
    desc.horizon = int(horizon)
    return desc, rec


def guard_generator(size, names):
    """The generator call with the stored descriptor seed reproduces the stored window
    field and horizon of every W combination."""
    from constraint_family.descriptor import make_descriptor
    from ppvc_instance_generator import load_instance
    k = 0
    for name in names:
        jl, pt, meta = load_instance(os.path.join(ROOT_DIR, P101.SIZES[size]['base'], name))
        for combo in WIN_COMBOS:
            with open(stored_desc_path(size, combo, name)) as f:
                rec = json.load(f)
            d = make_descriptor(combo, jl, pt, meta, pt.shape[1], seed=rec['desc_seed']).to_json()
            for key in ('windows', 'horizon', 'lag', 'op_class', 'setup', 'active'):
                if json.dumps(d[key]) != json.dumps(rec[key]):
                    sys.exit(f'REFUSING: the generator no longer reproduces {size}/{combo}/'
                             f'{name} field {key}')
            k += 1
    return k


def verify_horizon_source(size, n):
    """Replay SPT on the first n instances of each window-free combination and compare
    with the stored makespans used as horizons."""
    out = {}
    for combo in WIN_COMBOS:
        c0 = NO_WINDOW[combo]
        ref, _ = stored_spt(size, c0)
        cell = P101.load_cell(size, c0, n)
        got = [P101.rule_rollout('SPT', jl, pt, meta, desc)['makespan']
               for (_, jl, pt, meta, desc) in cell]
        if not np.array_equal(np.asarray(got), ref[:len(got)]):
            sys.exit(f'REFUSING: SPT on {size}/{c0} gives {got}, the stored array '
                     f'{ref[:len(got)].tolist()}; the horizon source is not reproduced')
        out[c0] = len(got)
    return out


def build(args):
    from constraint_family import checks
    from ppvc_instance_generator import load_instance
    P101._cpu_torch()
    dens, dur = window_ranges()
    for size in args.sizes:
        names = P101.names_of(size, args.n)
        row = {nm: i for i, nm in enumerate(P101.names_of(size))}
        n_guard = guard_generator(size, names)
        verified = verify_horizon_source(size, args.verify_n) if args.verify_n else {}
        digest = hashlib.sha256()
        stats = {c: dict(h_new=[], h_old=[], dens=[], n_win=[], n_win_old=[], c4=0)
                 for c in WIN_COMBOS}
        for combo in WIN_COMBOS:
            spt, spt_path = stored_spt(size, NO_WINDOW[combo])
            if len(spt) != len(row):
                sys.exit(f'REFUSING: {spt_path} has {len(spt)} rows, {size} has '
                         f'{len(row)} instances')
            for name in names:
                h = spt[row[name]]
                if h != int(h) or h <= 0:
                    sys.exit(f'REFUSING: stored SPT makespan {h} of {size}/{name} is not a '
                             f'positive integer')
                desc, rec = full_descriptor(size, combo, name, h)
                out = desc.to_json()
                out['base_instance'] = rec['base_instance']
                out['regime'] = combo
                out['desc_seed'] = rec['desc_seed']
                out['window_seed'] = window_seed(name)
                out['level'] = {
                    'level': 'win_full',
                    'rule': (f'windows redrawn by descriptor.draw_windows, density '
                             f'U[{dens[0]}, {dens[1]}] of the horizon, durations '
                             f'U{{{dur[0]}..{dur[1]}}} h, on the horizon of the stored SPT '
                             f'makespan without windows'),
                    'horizon_source': {'file': os.path.relpath(spt_path, ROOT_DIR),
                                       'combination': NO_WINDOW[combo], 'value': int(h)},
                    'window_stream': [window_seed(name), WINDOW_TAG],
                    'source_desc': os.path.relpath(stored_desc_path(size, combo, name),
                                                   ROOT_DIR),
                    'source_sha256': sha256_file(stored_desc_path(size, combo, name)),
                    'stored_horizon': rec['horizon']}
                txt = json.dumps(out)
                write_once(level_desc_path(args.root, 'win_full', size, combo, name), txt)
                digest.update(txt.encode())
                jl, pt, _ = load_instance(os.path.join(ROOT_DIR, P101.SIZES[size]['base'],
                                                       name))
                rep = {r.id: r for r in checks.run_all(desc, pt, False, job_length=jl)}
                bad = [c for c in ('C1', 'C2', 'C3', 'C5') if rep[c].status == checks.FAIL]
                if bad:
                    sys.exit(f'REFUSING: {size}/{combo}/{name} fails {bad}: '
                             f'{[rep[c].messages[:1] for c in bad]}')
                s = stats[combo]
                s['c4'] += rep['C4'].status == checks.WARN
                s['h_new'].append(int(h))
                s['h_old'].append(int(rec['horizon']))
                s['dens'].append(float(np.mean([sum(e - a for a, e in w) / h
                                                for w in desc.windows])))
                s['n_win'].append(float(np.mean([len(w) for w in desc.windows])))
                s['n_win_old'].append(float(np.mean([len(w) for w in rec['windows']])))
        man = {'level': 'win_full', 'size': size, 'n_instances': len(names),
               'instances': names, 'combinations': list(WIN_COMBOS),
               'density_range': list(dens), 'duration_range_h': list(dur),
               'window_seed': f'{WINDOW_SEED0} + instance index, tag {WINDOW_TAG}',
               'generator_guard_checked': n_guard, 'horizon_source_verified': verified,
               'per_combination': {c: {
                   'horizon_new_mean_h': float(np.mean(s['h_new'])),
                   'horizon_stored_mean_h': float(np.mean(s['h_old'])),
                   'window_density_mean': float(np.mean(s['dens'])),
                   'windows_per_machine_mean': float(np.mean(s['n_win'])),
                   'windows_per_machine_stored_mean': float(np.mean(s['n_win_old'])),
                   'c4_warn_share': s['c4'] / len(names)} for c, s in stats.items()},
               'content_sha256': digest.hexdigest(), 'commit': P101.git_commit()}
        mp = os.path.join(args.root, 'data', 'win_full', size, 'manifest.json')
        if os.path.exists(mp):
            with open(mp) as f:
                if json.load(f).get('content_sha256') != man['content_sha256']:
                    sys.exit(f'REFUSING: {mp} records different content')
        write_json_atomic(mp, man)
        print(f'[p102 build] {size}: {len(names)} instances x {len(WIN_COMBOS)} '
              f'combinations; guard {n_guard} stored descriptors reproduced; horizon '
              f'source verified on {verified}; ' + '; '.join(
                  f'{c} H {v["horizon_stored_mean_h"]:.0f}->{v["horizon_new_mean_h"]:.0f} h, '
                  f'density {v["window_density_mean"]:.3f}, windows/machine '
                  f'{v["windows_per_machine_stored_mean"]:.1f}->{v["windows_per_machine_mean"]:.1f}, '
                  f'C4 warn {v["c4_warn_share"]:.2f}'
                  for c, v in man['per_combination'].items()), flush=True)


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

def model_name(arm, combo, seed):
    reg_arm = {'pact': 'released', 'spec': 'specialist', 'single': 'control'}[arm]
    return REG.PREFIX + REG.model(reg_arm, combo if arm == 'spec' else None, seed, 'exact')


def units_for(level, size, combo, arms, seeds):
    out = []
    for arm in arms:
        if arm == 'rules':
            out += [(level, size, combo, 'rule', None, r) for r in RULES7]
        else:
            out += [(level, size, combo, arm, s, 'greedy+' + model_name(arm, combo, s))
                    for s in seeds]
    return out


def all_units(levels, sizes, combos, arms, seeds):
    return [u for lv in levels for sz in sizes for c in combos
            for u in units_for(lv, sz, c, arms, seeds)]


def result_path(root, level, size, combo, label):
    return os.path.join(root, 'results', 'exact', level, size, combo, f'{label}.npz')


def load_level_cell(root, level, size, combo, n):
    """[(name, jl, pt, meta, desc)] of one cell of a level."""
    from constraint_family.descriptor import ConstraintDescriptor
    from ppvc_instance_generator import load_instance
    out = []
    for name in P101.names_of(size, n):
        p = level_desc_path(root, level, size, combo, name)
        if not os.path.exists(p):
            sys.exit(f'REFUSING: {p} is missing; run the build phase first')
        with open(p) as f:
            rec = json.load(f)
        jl, pt, meta = load_instance(os.path.join(ROOT_DIR, P101.SIZES[size]['base'], name))
        out.append((name, jl, pt, meta,
                    ConstraintDescriptor.from_json(rec, setup_convention='attached')))
    return out


def aware_rollout(rule, jl, pt, meta, desc):
    """One rollout of ECT, SSPT or LLF (scripts/p72_aware_rules.py select_action) in the
    environment of P101.make_env, np.random.seed(50) after the data are set."""
    import params  # noqa: F401  (parsed with --device cpu before p72 resets argv)
    import p72_aware_rules as P72
    M = pt.shape[1]
    env = P101.make_env(jl, pt, meta, [desc])
    lb0 = float(P101.lb0_hours(env)[0])
    amch = np.full(pt.shape[0], -1, dtype=int)
    np.random.seed(P101.RULE_SEED)
    while not env.done().all():
        a = int(P72.select_action(rule, env))
        amch[env.candidate[0, a // M]] = a % M
        env.step(np.array([a]))
    return dict(makespan=float(env.current_makespan[0]), amch=amch,
                op_ct=env.true_op_ct[0].copy(), lb0=lb0)


def evaluate_unit(root, unit, cell, runner, force):
    level, size, combo, arm, seed, label = unit
    path = result_path(root, level, size, combo, label)
    names = [c[0] for c in cell]
    if os.path.exists(path) and os.path.exists(path[:-4] + '.json') and not force:
        with open(path[:-4] + '.json') as f:
            old = json.load(f)
        if old.get('instances') != names:
            sys.exit(f'REFUSING: {path} covers other instances; use --force or another '
                     f'--root')
        if arm != 'rule':
            ckpt = os.path.join(ROOT_DIR, 'trained_network', 'FAMILY',
                                f'{label[len("greedy+"):]}.pth')
            if old.get('policy', {}).get('checkpoint_sha256') != sha256_file(ckpt):
                sys.exit(f'REFUSING: {path} was written by another checkpoint of the same '
                         f'run; use --force or another --root')
        return 'skip', 0.0
    t0, c0 = time.perf_counter(), time.process_time()
    n, N, M = len(cell), cell[0][2].shape[0], cell[0][2].shape[1]
    arr = {k: np.zeros(n) for k in ('makespan', 'lb0', 'util', 'ms_over_lb0', 'cov_end',
                                    'cov_time', 'last_window_end', 'horizon')}
    arr['amch'] = np.zeros((n, N), dtype=np.int16)
    rec = {'level': level, 'size': size, 'regime': combo, 'arm': arm, 'seed': seed,
           'label': label, 'mask': P101.MASK}
    for i, (name, jl, pt, meta, desc) in enumerate(cell):
        if arm == 'rule' and label in RULES4:
            r = P101.rule_rollout(label, jl, pt, meta, desc)
        elif arm == 'rule':
            r = aware_rollout(label, jl, pt, meta, desc)
        else:
            r = runner.rollout(label[len('greedy+'):], jl, pt, meta, desc)
        v = P101.validate(jl, pt, desc, r['amch'], r['op_ct'], r['makespan'])
        if v:
            sys.exit(f'REFUSING: {label} on {level}/{size}/{combo}/{name} is infeasible: '
                     f'{v[:2]}')
        ce, ct, le = P101.window_coverage(desc, r['makespan'], M)
        u, ml, _ = P101.shop_load(pt, r['amch'], r['makespan'], r['lb0'])
        for k, val in (('makespan', r['makespan']), ('lb0', r['lb0']), ('util', u),
                       ('ms_over_lb0', ml), ('cov_end', ce), ('cov_time', ct),
                       ('last_window_end', le), ('horizon', float(desc.horizon or 0))):
            arr[k][i] = val
        arr['amch'][i] = r['amch']
    P101.save_npz_atomic(path, **arr)
    rr = P101.run_record({'unit': list(unit)}, names, t0, c0)
    rr.update(rec)
    if runner is not None and arm != 'rule':
        rr['policy'] = runner.load(label[len('greedy+'):])[1]
    write_json_atomic(path[:-4] + '.json', rr)
    return 'done', rr['cpu_s']


def unit_cost(u, n):
    return COST_S[u[1]]['rule' if u[3] == 'rule' else 'policy'] * (n or 30)


def assign_parts(units, parts, n):
    order = sorted(range(len(units)), key=lambda i: (-unit_cost(units[i], n), i))
    loads, out = [0.0] * parts, [[] for _ in range(parts)]
    for i in order:
        p = int(np.argmin(loads))
        out[p].append(i)
        loads[p] += unit_cost(units[i], n)
    return [sorted(x) for x in out], loads


def plan(args):
    units = all_units(args.levels, args.sizes, args.combos, args.arms, args.seeds)
    parts, loads = assign_parts(units, args.parts, args.n)
    total = sum(unit_cost(u, args.n) for u in units)
    print(f'[p102 plan] {len(units)} result files, estimated {total / 3600:.2f} core-hours '
          f'of CPU time (COST_S), {args.parts} parts of {min(loads) / 3600:.2f} to '
          f'{max(loads) / 3600:.2f} h')
    py = sys.executable
    for i in range(args.parts):
        print(f'CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c '
              f'<CORE_{i}> {py} scripts/p102_scale_sensitivity.py evaluate --arms '
              f'{",".join(args.arms)} --seeds {",".join(map(str, args.seeds))} '
              f'--part {i} --parts {args.parts}')
    return dict(units=len(units), core_h=total / 3600)


def evaluate(args):
    units = all_units(args.levels, args.sizes, args.combos, args.arms, args.seeds)
    if args.part is not None:
        parts, _ = assign_parts(units, args.parts, args.n)
        units = [units[i] for i in parts[args.part]]
    runner = P101.PolicyRunner() if any(a in POLICY_ARMS for a in args.arms) else None
    if runner is None:
        P101._cpu_torch()
    cache = {}
    t_all = time.perf_counter()
    for k, u in enumerate(units):
        key = u[:3]
        if key not in cache:
            cache.clear()
            cache[key] = load_level_cell(args.root, *key, args.n)
        st, c = evaluate_unit(args.root, u, cache[key], runner, args.force)
        print(f'[p102 eval {k + 1}/{len(units)}] {u[0]:8s} {u[1]:9s} {u[2]:4s} {u[5]:40s} '
              f'{st} cpu={c:7.1f}s elapsed={time.perf_counter() - t_all:7.0f}s', flush=True)


# ---------------------------------------------------------------------------
# infer
# ---------------------------------------------------------------------------

Refused = P101.Refused
refuse = P101.refuse


def _stats():
    """stats_family and the combined verdict rule of p89_public_family (the estimators
    of p87 infer)."""
    import stats_family as SF
    from p89_public_family import VERDICT_WORDS, superiority_verdict
    return SF, VERDICT_WORDS, superiority_verdict


def read_unit(root, level, size, combo, label, names):
    p = result_path(root, level, size, combo, label)
    if not (os.path.exists(p) and os.path.exists(p[:-4] + '.json')):
        refuse(f'result {level}/{size}/{combo}/{label} is missing ({p})')
    with open(p[:-4] + '.json') as f:
        rec = json.load(f)
    want = {'level': level, 'size': size, 'regime': combo, 'label': label}
    bad = {k: rec.get(k) for k, v in want.items() if rec.get(k) != v}
    if bad or rec.get('instances') != names:
        refuse(f'{p}: record {bad or "instance list"} does not match the study')
    with np.load(p) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def component(SF, d, nboot):
    """Interval package of one comparison d [k, n] or [k, n, c]: crossed hierarchical
    bootstrap (fresh default_rng(HIER_SEED)) and the seed-level t interval; with one
    training run the bootstrap over base instances (fresh default_rng(INST_SEED))."""
    x = np.asarray(d, dtype=float)
    k, n = x.shape[0], x.shape[1]
    c = x.shape[2] if x.ndim == 3 else 1
    if k >= 2:
        hb = SF.hier_boot(x, B=nboot, seed=HIER_SEED, alpha=ALPHA, return_draws=True)
        t = SF.seed_t(x, ALPHA)
        out = dict(mean=hb['mean'], lo=hb['lo'], hi=hb['hi'], t_lo=t['lo'], t_hi=t['hi'],
                   scheme='hierarchical')
        draws = hb['draws']
    else:
        ib = SF.instance_boot(x[0], B=nboot, seed=INST_SEED, alpha=ALPHA, return_draws=True)
        out = dict(mean=ib['mean'], lo=ib['lo'], hi=ib['hi'], t_lo=float('nan'),
                   t_hi=float('nan'), scheme='instance')
        draws = ib['draws']
    out.update(k=int(k), n=int(n), c=int(c), direction=SF.direction(out['lo'], out['hi']),
               **{f'v{mg}': SF.verdict(out['lo'], out['hi'], mg) for mg in MARGINS})
    return out, draws


def superiority(st, D, nboot):
    """Per combination of D the interval, the one-sided p-values in both directions,
    Holm over the combinations of D for each kind, and the combined verdict."""
    SF, _, sv = st
    rows = {}
    for r, d in D.items():
        c, draws = component(SF, d, nboot)
        pl, pg = SF.seed_t_p(d, 'less'), SF.seed_t_p(d, 'greater')
        c.update(p_t=1.0 if np.isnan(pl) else pl, p_boot=SF.boot_p_one_sided(draws, 'less'),
                 p_t_greater=1.0 if np.isnan(pg) else pg,
                 p_boot_greater=SF.boot_p_one_sided(draws, 'greater'))
        rows[r] = c
    keys = list(rows)
    for col in ('p_t', 'p_boot', 'p_t_greater', 'p_boot_greater'):
        for r, a in zip(keys, SF.holm([rows[r][col] for r in keys])):
            rows[r][col + '_holm'] = a
    for r in keys:
        x = rows[r]
        x['verdict'] = sv(x['lo'], x['hi'], x['p_t_holm'], x['p_boot_holm'],
                          x['p_t_greater_holm'], x['p_boot_greater_holm'], ALPHA)
    return rows


def boot_ci(x):
    return P101.ci(x)


def guard_out(out):
    P101.guard_out(out)


def infer(args):
    """Every outcome of the module doc. Returns the result dictionary."""
    partial = bool(args.partial)
    arms = list(args.arms)
    full_set = (args.levels == list(LEVELS) and args.sizes == list(SIZES)
                and args.combos == list(WIN_COMBOS) and not args.n)
    if args.n and not partial:
        refuse('--n (a subset of the instances) needs --partial')
    if not partial and not full_set:
        refuse('every level, size and combination is required; a subset needs --partial')
    if 'default' not in args.levels:
        refuse('the default level is the reference of every shift; it is required')
    in_place_ok = not partial and (arms == ['rules'] or sorted(arms) == sorted(ARM_CHOICES))
    if args.out is None:
        if not in_place_ok:
            refuse('--partial, or arms other than "rules" or every arm, write only to '
                   '--out <folder>')
        out_root = ROOT_DIR
    else:
        out_root = os.path.abspath(args.out)
        guard_out(out_root)
    paths = {k: os.path.join(out_root, v) for k, v in OUT_FILES.items()}
    for p in paths.values():
        if os.path.exists(p):
            os.remove(p)
    with_policy = all(a in arms for a in POLICY_ARMS)
    if any(a in arms for a in POLICY_ARMS) and not with_policy:
        refuse('the policy comparisons need the arms pact, spec and single together')
    if with_policy and len(args.seeds) < 2 and not partial:
        refuse('at least two training seeds are required (the unit of the hierarchical '
               'bootstrap)')
    st = _stats() if with_policy else None
    seeds = list(args.seeds)
    res = {'generator': GENERATOR, 'partial': partial, 'arms': arms, 'seeds': seeds,
           'mask': P101.MASK, 'B': args.boot, 'levels': list(args.levels),
           'convention': 'd = 100 (first - second) / second, seed s against seed s, per '
                         'base instance; negative means the first-named method is shorter',
           'best_rule': 'per-instance minimum of ' + ', '.join(RULES4),
           'best_rule7': 'per-instance minimum of ' + ', '.join(RULES7),
           'sizes': {}}
    for size in args.sizes:
        names = P101.names_of(size, args.n)
        S = {'n': len(names), 'combos': {}}
        mp = os.path.join(args.root, 'data', 'win_full', size, 'manifest.json')
        if 'win_full' in args.levels:
            if not os.path.exists(mp):
                refuse(f'{mp} is missing; run the build phase first')
            with open(mp) as f:
                S['manifest'] = {k: v for k, v in json.load(f).items() if k != 'instances'}
        gaps = {lv: {} for lv in args.levels}
        for combo in args.combos:
            C = {'levels': {}}
            for lv in args.levels:
                R = {r: read_unit(args.root, lv, size, combo, r, names) for r in RULES7} \
                    if 'rules' in arms else {}
                L = {}
                if R:
                    ms = {r: R[r]['makespan'] for r in RULES7}
                    b4 = np.min(np.stack([ms[r] for r in RULES4]), axis=0)
                    b7 = np.min(np.stack([ms[r] for r in RULES7]), axis=0)
                    L.update(rule_ms={r: float(ms[r].mean()) for r in RULES7},
                             best_rule_ms=float(b4.mean()), best_rule7_ms=float(b7.mean()),
                             cov_spt=dict(cov_end=float(R['SPT']['cov_end'].mean()),
                                          cov_end_min=float(R['SPT']['cov_end'].min()),
                                          cov_end_max=float(R['SPT']['cov_end'].max()),
                                          cov_time=100.0 * float(R['SPT']['cov_time'].mean()),
                                          horizon=float(R['SPT']['horizon'].mean()),
                                          last_window_end=float(
                                              R['SPT']['last_window_end'].mean())),
                             util_spt=100.0 * float(R['SPT']['util'].mean()))
                    L['_ms'], L['_b4'], L['_b7'] = ms, b4, b7
                    if lv == 'default':
                        L['stored_check'] = check_stored(size, combo, ms, len(names))
                if with_policy:
                    P = {a: np.stack([read_unit(args.root, lv, size, combo,
                                                'greedy+' + model_name(a, combo, s),
                                                names)['makespan'] for s in seeds])
                         for a in POLICY_ARMS}
                    if not R:
                        refuse('the policy comparisons against the rules need --arms with '
                               'rules')
                    d = {'pol_spec': 100.0 * (P['pact'] - P['spec']) / P['spec'],
                         'ctl_spec': 100.0 * (P['single'] - P['spec']) / P['spec'],
                         'pol_rule': 100.0 * (P['pact'] - L['_b4'][None]) / L['_b4'][None],
                         'pol_rule7': 100.0 * (P['pact'] - L['_b7'][None]) / L['_b7'][None]}
                    gaps[lv][combo] = d
                    L['policy_ms'] = {a: float(P[a].mean()) for a in POLICY_ARMS}
                    if lv == 'default':
                        L['stored_policy_check'] = check_stored_policy(
                            size, combo, seeds, P['pact'])
                    cov = [read_unit(args.root, lv, size, combo,
                                     'greedy+' + model_name('pact', combo, s), names)
                           for s in seeds]
                    L['cov_policy'] = dict(
                        cov_end=float(np.mean([c['cov_end'].mean() for c in cov])),
                        cov_time=100.0 * float(np.mean([c['cov_time'].mean() for c in cov])))
                C['levels'][lv] = L
            if 'rules' in arms and 'win_full' in args.levels:
                D0, D1 = C['levels']['default'], C['levels']['win_full']
                C['rule_shift'] = {
                    'spt': boot_ci(100.0 * (D1['_ms']['SPT'] - D0['_ms']['SPT'])
                                   / D0['_ms']['SPT']),
                    'best_rule': boot_ci(100.0 * (D1['_b4'] - D0['_b4']) / D0['_b4'])}
            for L in C['levels'].values():
                for k in ('_ms', '_b4', '_b7'):
                    L.pop(k, None)
            S['combos'][combo] = C
        if with_policy:
            S['policy'] = policy_block(st, gaps, args.levels, args.combos, args.boot)
        res['sizes'][size] = S
    lines = macro_lines(res)
    names_m = re.findall(r'\\newcommand\{\\([^}]*)\}', '\n'.join(lines))
    check_macro_names(names_m, paths)
    from macro_io import emit
    emit(paths['macros'], lines)
    write_json_atomic(paths['json'], jsonable(res))
    os.makedirs(os.path.dirname(paths['md']), exist_ok=True)
    with open(paths['md'], 'w') as f:
        f.write('\n'.join(markdown(res)) + '\n')
    for k in ('json', 'md'):
        print('wrote', paths[k])
    print(f'{len(names_m)} macros' + ('; PARTIAL' if partial else ''))
    return res


def check_stored(size, combo, ms, n):
    """The default level's rule makespans against the stored exact-mask arrays of the
    same rules; refused on a difference, recorded when a stored array is absent."""
    out = {}
    for r, a in ms.items():
        p = os.path.join(ROOT_DIR, P101.RESULTS_ROOT, f'{size}_xmask', f'Result_{r}_{combo}.npy')
        if not os.path.exists(p):
            out[r] = 'no stored array'
            continue
        s = np.load(p)
        s = (s[:, 0] if s.ndim > 1 else s).astype(float)[:n]
        if not np.array_equal(s, a):
            refuse(f'{size}/{combo}: the default level of {r} differs from {p} on '
                   f'{int((s != a).sum())} instances; the pipelines diverge')
        out[r] = 'equal'
    return out


def check_stored_policy(size, combo, seeds, pact):
    """The default level's released-policy makespans [k, n] against the arrays of the
    exact campaign (test_results/FAMILY/<size>_xmask), seed by seed; refused on a
    difference, recorded when a stored array is absent."""
    out = {}
    for k, s in enumerate(seeds):
        p = os.path.join(ROOT_DIR, P101.RESULTS_ROOT, f'{size}_xmask',
                         f'Result_greedy+{model_name("pact", combo, s)}_{combo}.npy')
        if not os.path.exists(p):
            out[s] = 'no stored array'
            continue
        a = np.load(p)
        a = (a[:, 0] if a.ndim > 1 else a).astype(float)[:pact.shape[1]]
        if not np.array_equal(a, pact[k]):
            refuse(f'{size}/{combo}: the default level of the released policy, seed {s}, '
                   f'differs from {p} on {int((a != pact[k]).sum())} instances')
        out[s] = 'equal'
    return out


def policy_block(st, gaps, levels, combos, nboot):
    """Comparisons per level and combination, pooled over the combinations, and the
    shift from the default level to win_full, with verdict counts and changes."""
    SF = st[0]
    out = {}
    for lv in levels:
        G = gaps[lv]
        rows = {key: {r: component(SF, G[r][key], nboot)[0] for r in combos}
                for key in ('pol_spec', 'ctl_spec')}
        for key in ('pol_rule', 'pol_rule7'):
            rows[key] = superiority(st, {r: G[r][key] for r in combos}, nboot)
        pooled = {key: component(SF, np.stack([G[r][key] for r in combos], axis=2), nboot)[0]
                  for key in COMPARISONS}
        counts = {f'{key}|{mg}': {v: [rows[key][r][f'v{mg}'] for r in combos].count(v)
                                  for v in SF.VERDICTS}
                  for key in ('pol_spec', 'ctl_spec') for mg in MARGINS}
        for key in ('pol_rule', 'pol_rule7'):
            counts[key] = {v: [rows[key][r]['verdict'] for r in combos].count(v)
                           for v in st[1]}
        out[lv] = {'rows': rows, 'pooled': pooled, 'counts': counts}
    if 'win_full' in levels:
        sh = {key: {r: component(SF, gaps['win_full'][r][key] - gaps['default'][r][key],
                                 nboot)[0] for r in combos} for key in COMPARISONS}
        for key in COMPARISONS:
            sh[key]['pooled'] = component(SF, np.stack(
                [gaps['win_full'][r][key] - gaps['default'][r][key] for r in combos], axis=2),
                nboot)[0]
        out['shift'] = sh
        ch = {}
        for key in COMPARISONS:
            cols = [f'v{mg}' for mg in MARGINS] if key in ('pol_spec', 'ctl_spec') else ['verdict']
            for col in cols:
                ch[f'{key}|{col}'] = [r for r in combos
                                      if out['win_full']['rows'][key][r][col]
                                      != out['default']['rows'][key][r][col]]
        out['changed'] = ch
    return out


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------

f2, u1, u2 = P101.f2, P101.u1, P101.u2


def _tex_verdict(v):
    import stats_family as SF
    return SF.TEX.get(v, v)


def macro_lines(res):
    mark = 'PARTIAL ' if res['partial'] else ''
    L = [f'% Generated by {GENERATOR} -- do not hand-edit',
         '% Sensitivity at the production sizes (50 and 80 modules, 25 stations, 30 '
         'instances each), exact mask: the stored windows (level Def) and windows drawn '
         'again over the schedule length (level Full), combinations W, LW, SW, LSW.',
         '% Gaps in %: first-named method minus second over second, negative means the '
         'first-named method is shorter; best classical rule: per-instance minimum of '
         'FIFO, MOR, SPT and MWKR restricted to the legal pairs.']
    if res['partial']:
        L.insert(1, '% PARTIAL: subset of levels, sizes, combinations or instances; not '
                    'for the manuscript.')
    for size, S in res['sizes'].items():
        w = SIZE_WORDS[size]
        L.append(f'\\newcommand{{\\{PREFIX}{w}N}}{{{S["n"]}}}  % {mark}{size}: instances')
        man = S.get('manifest')
        if man:
            for combo, v in man['per_combination'].items():
                if combo not in S['combos']:
                    continue
                L.append(f'\\newcommand{{\\{PREFIX}HorizonOld{w}{combo}}}'
                         f'{{{v["horizon_stored_mean_h"]:.0f}}}'
                         f'\\newcommand{{\\{PREFIX}HorizonNew{w}{combo}}}'
                         f'{{{v["horizon_new_mean_h"]:.0f}}}'
                         f'\\newcommand{{\\{PREFIX}WinDensity{w}{combo}}}'
                         f'{{{u1(100.0 * v["window_density_mean"])}}}'
                         f'\\newcommand{{\\{PREFIX}WinPerMachineOld{w}{combo}}}'
                         f'{{{u1(v["windows_per_machine_stored_mean"])}}}'
                         f'\\newcommand{{\\{PREFIX}WinPerMachineNew{w}{combo}}}'
                         f'{{{u1(v["windows_per_machine_mean"])}}}'
                         f'  % {mark}{size} {combo}: mean horizon of the stored windows and '
                         f'of the new windows (h), new window density (% of the horizon), '
                         f'windows per machine, stored and new')
        for combo, C in S['combos'].items():
            for lv, D in C['levels'].items():
                lw = LEVEL_WORDS[lv]
                if 'rule_ms' in D:
                    cv = D['cov_spt']
                    L.append(f'\\newcommand{{\\{PREFIX}MsSpt{lw}{w}{combo}}}'
                             f'{{{u1(D["rule_ms"]["SPT"])}}}'
                             f'\\newcommand{{\\{PREFIX}MsRule{lw}{w}{combo}}}'
                             f'{{{u1(D["best_rule_ms"])}}}'
                             f'\\newcommand{{\\{PREFIX}MsRuleSeven{lw}{w}{combo}}}'
                             f'{{{u1(D["best_rule7_ms"])}}}'
                             f'\\newcommand{{\\{PREFIX}SptCovEnd{lw}{w}{combo}}}'
                             f'{{{u2(cv["cov_end"])}}}'
                             f'\\newcommand{{\\{PREFIX}SptCovTime{lw}{w}{combo}}}'
                             f'{{{u1(cv["cov_time"])}}}'
                             f'  % {mark}{size} {combo}, level {lv}: mean makespan (h) of SPT, '
                             f'of the best classical rule, of the best of seven rules; SPT '
                             f'schedule: last window end over the makespan, window time '
                             f'before the makespan over machines times makespan (%)')
                if 'cov_policy' in D:
                    L.append(f'\\newcommand{{\\{PREFIX}PolCovEnd{lw}{w}{combo}}}'
                             f'{{{u2(D["cov_policy"]["cov_end"])}}}'
                             f'\\newcommand{{\\{PREFIX}PolCovTime{lw}{w}{combo}}}'
                             f'{{{u1(D["cov_policy"]["cov_time"])}}}'
                             f'  % {mark}{size} {combo}, level {lv}: the same coverage for '
                             f'the released policy, mean over its runs')
            if 'rule_shift' in C:
                for key, word in (('spt', 'Spt'), ('best_rule', 'Rule')):
                    g = C['rule_shift'][key]
                    L.append(f'\\newcommand{{\\{PREFIX}{word}Shift{w}{combo}}}{{{f2(g["mean"])}}}'
                             f'\\newcommand{{\\{PREFIX}{word}ShiftLo{w}{combo}}}{{{f2(g["lo"])}}}'
                             f'\\newcommand{{\\{PREFIX}{word}ShiftHi{w}{combo}}}{{{f2(g["hi"])}}}'
                             f'  % {mark}{size} {combo}: change of the {key.replace("_", " ")} '
                             f'makespan from the stored to the new windows, %, mean and 95% '
                             f'interval over {g["n"]} instances')
        if 'policy' in S:
            P = S['policy']
            for lv in res['levels']:
                lw = LEVEL_WORDS[lv]
                for key, (first, second, cw) in COMPARISONS.items():
                    for r in list(S['combos']) + ['Pool']:
                        c = P[lv]['pooled'][key] if r == 'Pool' else P[lv]['rows'][key][r]
                        s = (f'\\newcommand{{\\{PREFIX}{cw}{lw}{w}{r}}}{{{f2(c["mean"])}}}'
                             f'\\newcommand{{\\{PREFIX}{cw}Lo{lw}{w}{r}}}{{{f2(c["lo"])}}}'
                             f'\\newcommand{{\\{PREFIX}{cw}Hi{lw}{w}{r}}}{{{f2(c["hi"])}}}')
                        if key in ('pol_spec', 'ctl_spec'):
                            s += ''.join(f'\\newcommand{{\\{PREFIX}{cw}V{MW[mg]}{lw}{w}{r}}}'
                                         f'{{{_tex_verdict(c[f"v{mg}"])}}}' for mg in MARGINS)
                        elif r != 'Pool':
                            s += f'\\newcommand{{\\{PREFIX}{cw}Verdict{lw}{w}{r}}}{{{c["verdict"]}}}'
                        s += f'\\newcommand{{\\{PREFIX}{cw}Dir{lw}{w}{r}}}{{{_tex_verdict(c["direction"])}}}'
                        L.append(s + f'  % {mark}{size} {r}, level {lv}: {first} vs {second}, '
                                     f'{c["scheme"]} 95% interval over {c["k"]} runs and '
                                     f'{c["n"]} instances' + (f', {c["c"]} combinations'
                                                              if c['c'] > 1 else ''))
            if 'shift' in P:
                for key, (first, second, cw) in COMPARISONS.items():
                    for r in list(S['combos']) + ['Pool']:
                        c = P['shift'][key]['pooled' if r == 'Pool' else r]
                        L.append(f'\\newcommand{{\\{PREFIX}{cw}Shift{w}{r}}}{{{f2(c["mean"])}}}'
                                 f'\\newcommand{{\\{PREFIX}{cw}ShiftLo{w}{r}}}{{{f2(c["lo"])}}}'
                                 f'\\newcommand{{\\{PREFIX}{cw}ShiftHi{w}{r}}}{{{f2(c["hi"])}}}'
                                 f'  % {mark}{size} {r}: change of the {first} vs {second} gap '
                                 f'from the stored to the new windows, percentage points')
                s, what = '', []
                for k2, lst in P['changed'].items():
                    key, col = k2.split('|')
                    cw = COMPARISONS[key][2]
                    mw = MW[float(col[1:])] if col != 'verdict' else ''
                    s += f'\\newcommand{{\\{PREFIX}{cw}Changed{mw}{w}}}{{{len(lst)}}}'
                    what.append(f'{cw}{mw}: {",".join(lst) or "none"}')
                L.append(s + f'  % {mark}{size}: combinations whose verdict differs between '
                             f'the stored and the new windows ({"; ".join(what)})')
    return L


def check_macro_names(names, out_paths):
    bad = [x for x in names if not re.fullmatch(PREFIX + r'[A-Za-z]+', x)]
    if bad:
        refuse(f'macro names that are not {PREFIX} plus letters: {bad[:5]}')
    if len(names) != len(set(names)):
        refuse(f'duplicate macro names: {sorted({x for x in names if names.count(x) > 1})[:5]}')
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
    L = [f'# Sensitivity at the production sizes ({GENERATOR})', '']
    if res['partial']:
        L += ['PARTIAL: a subset of levels, sizes, combinations or instances. Not for the '
              'manuscript.', '']
    L += [f'Exact mask. Arms {", ".join(res["arms"])}; training seeds '
          f'{",".join(map(str, res["seeds"]))}. Level default: the stored windows. Level '
          'win_full: windows drawn again by the generator\'s rule on the horizon of the stored '
          'SPT makespan without windows. Best rule: ' + res['best_rule'] + '. Shift: 100 '
          '(full - default) / default per instance, 95% bootstrap over instances.', '']

    def g(x):
        return f'{f2(x["mean"])} [{f2(x["lo"])}, {f2(x["hi"])}]'
    for size, S in res['sizes'].items():
        L += [f'## {size} ({S["n"]} instances)', '']
        man = S.get('manifest')
        if man:
            L += ['| combination | stored horizon (h) | new horizon (h) | new density (%) | '
                  'windows per machine, stored | new | C4 warnings |', '|' + '---|' * 7]
            for combo, v in man['per_combination'].items():
                L.append(f'| {combo} | {v["horizon_stored_mean_h"]:.0f} | '
                         f'{v["horizon_new_mean_h"]:.0f} | {u1(100 * v["window_density_mean"])} | '
                         f'{u1(v["windows_per_machine_stored_mean"])} | '
                         f'{u1(v["windows_per_machine_mean"])} | {v["c4_warn_share"]:.2f} |')
            L.append('')
        L += ['| combination | level | SPT (h) | best classical rule (h) | best of seven (h) | '
              'SPT: last window end / makespan [min, max] | SPT: window time (%) | '
              'default level equals stored arrays |', '|' + '---|' * 8]
        for combo, C in S['combos'].items():
            for lv, D in C['levels'].items():
                if 'rule_ms' not in D:
                    continue
                cv = D['cov_spt']
                chk = D.get('stored_check')
                L.append(f'| {combo} | {lv} | {u1(D["rule_ms"]["SPT"])} | '
                         f'{u1(D["best_rule_ms"])} | {u1(D["best_rule7_ms"])} | '
                         f'{u2(cv["cov_end"])} [{u2(cv["cov_end_min"])}, '
                         f'{u2(cv["cov_end_max"])}] | {u1(cv["cov_time"])} | '
                         + (', '.join(f'{r}: {v}' for r, v in chk.items()) if chk else '')
                         + ' |')
        if any('rule_shift' in C for C in S['combos'].values()):
            L += ['', '| combination | SPT shift (%) | best classical rule shift (%) |',
                  '|---|---|---|']
            for combo, C in S['combos'].items():
                if 'rule_shift' in C:
                    L.append(f'| {combo} | {g(C["rule_shift"]["spt"])} | '
                             f'{g(C["rule_shift"]["best_rule"])} |')
        if 'policy' in S:
            P = S['policy']
            L += ['', '| level | combination | policy - specialist | control - specialist | '
                  'policy - best classical rule | policy - best of seven |', '|' + '---|' * 6]
            for lv in res['levels']:
                for r in list(S['combos']) + ['pooled']:
                    def cell(key):
                        c = P[lv]['pooled'][key] if r == 'pooled' else P[lv]['rows'][key][r]
                        tail = (f' {c["v0.5"]} / {c["v1.0"]}' if key in ('pol_spec', 'ctl_spec')
                                else (f' {c["verdict"]}' if 'verdict' in c
                                      else f' direction {c["direction"]}'))
                        return g(c) + tail
                    L.append(f'| {lv} | {r} | ' + ' | '.join(cell(k) for k in COMPARISONS) + ' |')
            if 'shift' in P:
                L += ['', 'Shift of each gap from the stored to the new windows (percentage '
                      'points):', '', '| combination | ' + ' | '.join(
                          f'{a} - {b}' for a, b, _ in COMPARISONS.values()) + ' |',
                      '|' + '---|' * (1 + len(COMPARISONS))]
                for r in list(S['combos']) + ['pooled']:
                    L.append(f'| {r} | ' + ' | '.join(g(P['shift'][k][r]) for k in COMPARISONS)
                             + ' |')
                L += ['', 'Verdicts that change: ' + '; '.join(
                    f'{k}: {",".join(v) or "none"}' for k, v in P['changed'].items())]
        L.append('')
    return L


# ---------------------------------------------------------------------------
# classes: can the environment hold a fifth setup class?
# ---------------------------------------------------------------------------

FIVE_CLASS_SEED = 20261113


def five_class_descriptor(size='10x25', combo='S', name='instance_000'):
    """A five-class variant of one stored descriptor, built in memory only: a fixed
    quarter of the modules (drawn with default_rng(SeedSequence([20261113, 5, index])))
    becomes class 4, and the setup matrix gains the column and row of class 4, drawn
    from the generator's ranges, with the cold-start row kept last."""
    from constraint_family.descriptor import ConstraintDescriptor
    from ppvc_instance_generator import load_instance
    with open(stored_desc_path(size, combo, name)) as f:
        rec = json.load(f)
    jl, pt, meta = load_instance(os.path.join(ROOT_DIR, P101.SIZES[size]['base'], name))
    d = ConstraintDescriptor.from_json(rec, setup_convention='attached')
    rng = np.random.default_rng(np.random.SeedSequence(
        [FIVE_CLASS_SEED, 5, P101.instance_index(name)]))
    J = len(jl)
    pick = np.sort(rng.choice(J, size=J // 4, replace=False))
    job_of_op = np.repeat(np.arange(J), jl)
    oc = d.op_class.copy()
    oc[np.isin(job_of_op, pick)] = 4
    S4 = np.asarray(rec['setup'], float)            # [5, 4]: classes 0-3, cold-start row
    mean_pt = float(pt[pt > 0].mean())
    S5 = np.zeros((6, 5))
    S5[:4, :4] = S4[:4]
    for a in range(4):
        v = max(1, int(round(rng.uniform(0.50 * mean_pt, 1.00 * mean_pt))))
        S5[a, 4] = S5[4, a] = v
    S5[4, 4] = max(1, int(round(rng.uniform(0.25 * mean_pt, 0.50 * mean_pt))))
    S5[5, :4] = S4[4]
    S5[5, 4] = max(1, int(round(rng.uniform(0.25 * mean_pt, 0.50 * mean_pt))))
    d5 = ConstraintDescriptor(active=d.active, lag=d.lag, op_class=oc, setup=S5,
                              windows=d.windows, horizon=d.horizon)
    return d5, jl, pt, meta, pick


def five_class_probe():
    """Each place that reads a descriptor, fed the five-class variant; returns
    {site: outcome}. Nothing is written."""
    from constraint_family import checks
    from constraint_family.descriptor import ConstraintDescriptor, IDLE_CLASS, N_CLASSES
    P101._cpu_torch()
    d5, jl, pt, meta, pick = five_class_descriptor()
    out = {'constants': f'descriptor.N_CLASSES = {N_CLASSES}, IDLE_CLASS = {IDLE_CLASS}',
           'modules_in_class_4': [int(x) for x in pick]}
    try:
        ConstraintDescriptor.from_json(d5.to_json(), setup_convention='attached')
        out['descriptor_json'] = 'accepted (no shape check in to_json or from_json)'
    except Exception as e:  # noqa: BLE001
        out['descriptor_json'] = f'refused: {type(e).__name__}: {e}'
    rep = {r.id: r for r in checks.run_all(d5, pt, False, job_length=jl)}
    out['check_C1'] = f'{rep["C1"].status}: {rep["C1"].messages[:2]}'
    try:
        P101.make_env(jl, pt, meta, [d5])
        out['environment'] = 'accepted'
    except Exception as e:  # noqa: BLE001
        out['environment'] = f'refused: {type(e).__name__}: {e}'
    out['idle_row'] = (f'the environment and constraint_family/reference_sim.py start every '
                       f'machine in class IDLE_CLASS = {IDLE_CLASS}, which is the index the '
                       f'fifth class would take, so the cold-start row and the fifth class '
                       f'would share one row index')
    return out


def classes(args):
    out = five_class_probe()
    for k, v in out.items():
        print(f'{k}: {v}')
    return out


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('phase', choices=['build', 'plan', 'evaluate', 'infer', 'classes'])
    ap.add_argument('--root', default=DEFAULT_ROOT)
    ap.add_argument('--levels', default=','.join(LEVELS))
    ap.add_argument('--sizes', default=','.join(SIZES))
    ap.add_argument('--combos', default=','.join(WIN_COMBOS))
    ap.add_argument('--arms', default=','.join(ARM_CHOICES))
    ap.add_argument('--seeds', default='301-305')
    ap.add_argument('--n', type=int, default=0, help='first n instances (smoke only)')
    ap.add_argument('--verify_n', type=int, default=3,
                    help='build: instances per window-free combination replayed with SPT')
    ap.add_argument('--part', type=int, default=None)
    ap.add_argument('--parts', type=int, default=1)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--partial', action='store_true')
    ap.add_argument('--out', default=None)
    ap.add_argument('--boot', type=int, default=B)
    args = ap.parse_args(_ARGV)
    P101.check_mask_options()
    args.levels = P101.parse_list(args.levels, LEVELS, 'levels')
    args.sizes = P101.parse_list(args.sizes, SIZES, 'sizes')
    args.combos = P101.parse_list(args.combos, WIN_COMBOS, 'combinations')
    args.arms = P101.parse_list(args.arms, ARM_CHOICES, 'arms')
    args.seeds = list(REG.parse_seeds(args.seeds))
    for s in args.seeds:
        if s not in REG.PLANNED_SEEDS['exact']:
            sys.exit(f'REFUSING: seed {s} is not planned in the exact environment')
    args.n = args.n or None
    if args.part is not None and not 0 <= args.part < args.parts:
        sys.exit(f'--part {args.part} outside 0..{args.parts - 1}')
    os.chdir(ROOT_DIR)
    {'build': build, 'plan': plan, 'evaluate': evaluate, 'infer': infer,
     'classes': classes}[args.phase](args)


if __name__ == '__main__':
    main()
