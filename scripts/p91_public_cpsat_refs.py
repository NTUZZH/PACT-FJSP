r"""P91: CP-SAT reference schedules for the public-instance family.

Purpose. Best-known makespans of the public sets exist for combination N only.
This runner gives every combination of a public family set a reference: the
best schedule OR-Tools CP-SAT finds within a cap, its status (proved optimal
or not) and its bound. One solve per (base instance, combination), one JSON
record per solve:

  <out>/<combination>/<base instance>.json        (default <out>:
  or_solution/PUBFAMILY/<set>_cpsat<budget>s_w<workers>)

MODEL. The exact model is cpsat_arms.build_model with every operation free
(the 'compact' arm of scripts/p85_cpsat_arms.py): attached setups through one
circuit per machine, one optional interval per eligible machine sharing the
operation's block, start and end variables, lags on the block, outages inside
each machine's no-overlap, the horizon set to the warm-start makespan. The
horizon is exact: the warm start is a validated feasible schedule, so the
optimum and every schedule at least as good lie inside it. The model of
cpsat_family.family_solver (rebuilt by cpsat_arms.build_reference_model) is
available as --model ref; both are exact and reach the same optimum
(tests/test_public_cpsat_refs.py checks both against exhaustive search on
small public-style instances in all eight combinations). On public 10x5
instances the compact model proves regime-N optima that the family_solver
model does not prove within the same time, so it is the default.

WARM START. The best of the four masked dispatching rules of eval_family.py
(FIFO, MOR, SPT, MWKR restricted to the non-delay mask, first_mask and
mask_ties exact, numpy seed configs.seed_test before each rollout), rolled out
in FJSPEnvFamily. Each rollout must reproduce the stored makespan of its rule
for that instance and combination (--stored), so the environment, the
descriptor and the row order are checked before every solve. The hint passes
both validators and covers every variable of the model (CP-SAT reports it
complete and feasible and takes it as its first solution). The reported
reference is the schedule the solver returns.

TIMING. Every solve runs in a fresh child process pinned to --cores. A record
carries cpu_share = CPU seconds / (wall seconds x workers) over the model
build and the solve, the build's CPU seconds apart, the affinity mask, the
machine load average before and after (information only), the peak memory and
the start and end times. Validity of the value (record_validity):
  * a proved optimum is exact whatever CPU it received: valid by proof. A
    solve that ends at optimality after a few seconds spends most of its wall
    time in the single-threaded build and presolve, so its share is
    structurally below 0.9 (the exception documented in
    scripts/timing_validity.py, rule (b)); its times are marked
    timing_valid = False, its value is valid;
  * a solve stopped by the cap is valid only with cpu_share >= 0.9 on the
    reserved cores; otherwise it is re-run.
Development runs (--dev) are never valid.

CHECKS. Before a record is written, the returned schedule passes
cpsat_arms.validate_all (the strict attached-setup validator and the
repository's C6 check) and its replay through reference_sim.simulate (same
machines, same sequences) is no longer than the solver's makespan; a failure
writes no record and goes to <out>/_failures.log. After each record, the
instance is checked against everything known about it: in N the bound may not
exceed the best-known makespan and a proved optimum must equal a best-known
value marked optimal; for every pair of combinations R inside R' the makespan
of R' is at least the bound of R (switching a rule on can only delay); every
stored makespan of a rule or policy for that instance and combination is at
least the strongest bound of the combination and its sub-combinations. A
violation is appended to _failures.log, written to <out>/_STOPPED and stops the
campaign; the runner refuses to start while _STOPPED exists.

CAMPAIGN CONTROL. --start_when FILE PREFIX waits before the first solve until
the last line of FILE that does not start with NOTE starts with PREFIX, or
such a line was appended while the runner waited (a state that passes between
two polls is not missed; a line already in the file does not count);
--stop_when FILE PREFIX takes no new solve once it does; no new solve starts
while --hold_file (default train_log/PUBREF_HOLD) exists. Solves run
combination by combination in the order N, S, LS, SW, LSW, L, W, LW (--order
changes it), instances ascending; existing records are skipped. A spawned
child imports the solver code from disk, so the runner fingerprints the source
files of a solve (CODE_FILES) at the start, stores the fingerprint in every
record and stops the campaign when a file changes while it runs.

Usage (from the repository root):
  python scripts/p91_public_cpsat_refs.py --phase plan --budget 300 --workers 2 --cores 0-1
  python scripts/p91_public_cpsat_refs.py --budget 300 --workers 2 --cores 0-1 \
      [--set pub_SD2_test_10x5+mix] [--regimes N S] [--instances 0-9] \
      [--start_when FILE PREFIX] [--stop_when FILE PREFIX]
  python scripts/p91_public_cpsat_refs.py --phase check --out <folder>

load_refs(out_dir) returns, per combination, the arrays of objective, bound,
proved flag and validity in the sorted-name order of the result arrays.
"""
import argparse
import csv
import glob
import json
import multiprocessing as mp
import os
import platform
import resource
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

R8 = ('N', 'L', 'S', 'W', 'LS', 'LW', 'SW', 'LSW')
ORDER = ('N', 'S', 'LS', 'SW', 'LSW', 'L', 'W', 'LW')
RULES = ('FIFO', 'MOR', 'SPT', 'MWKR')
SETUP = {'S', 'LS', 'SW', 'LSW'}
SHARE_MIN = 0.9
EPS = 1e-6
DEFAULT_SET = 'pub_SD2_test_10x5+mix'
DATA_ROOT = 'data/PUBLIC/family'
STORED_ROOT = 'test_results/PUBFAMILY'
BEST_KNOWN = {'pub_SD2_test_10x5+mix': 'data/PUBLIC/SD2/best_known_test_10x5+mix.csv',
              'pub_SD2_test_20x10+mix': 'data/PUBLIC/SD2/best_known_test_20x10+mix.csv'}
HOLD_FILE = 'train_log/PUBREF_HOLD'
MODELS = ('compact', 'ref')
STOP_MARK = '_STOPPED'
FAIL_LOG = '_failures.log'
# Source files a solve executes. A spawned child imports them from disk, so
# the campaign stops when one of them changes while it runs.
CODE_FILES = ('scripts/p91_public_cpsat_refs.py', 'constraint_family/cpsat_arms.py',
              'constraint_family/cpsat_family.py', 'constraint_family/reference_sim.py',
              'constraint_family/checks.py', 'schedule_validator.py',
              'constraint_family/descriptor.py', 'constraint_family/public_rules.py',
              'constraint_family/public_adapter.py')


# ---------------------------------------------------------------- paths
def in_root(p):
    """A relative path is relative to the repository root."""
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def show(p):
    """Path as stored in a record: relative to the root when inside it."""
    a = os.path.abspath(p)
    r = os.path.relpath(a, ROOT)
    return r if not r.startswith('..') else os.path.basename(a)


def subsets(regime):
    """Combinations whose rules are a subset of `regime`'s (itself included)."""
    s = set('' if regime == 'N' else regime)
    return [r for r in R8 if set('' if r == 'N' else r) <= s]


def parse_cores(spec):
    out = []
    for part in str(spec).split(','):
        if '-' in part:
            a, b = part.split('-')
            out += list(range(int(a), int(b) + 1))
        elif part.strip():
            out.append(int(part))
    return sorted(set(out))


def parse_indices(spec, n):
    if spec in (None, '', 'all'):
        return list(range(n))
    out = []
    for part in spec.split(','):
        if '-' in part:
            a, b = part.split('-')
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    bad = [i for i in out if not 0 <= i < n]
    if bad:
        raise SystemExit(f'REFUSING: instance indices {bad} outside 0..{n - 1}')
    return sorted(set(out))


def code_fingerprint(files=None):
    """First 16 hex digits of the SHA-256 over the source files of a solve."""
    import hashlib
    h = hashlib.sha256()
    for f in files or CODE_FILES:
        h.update(f.encode())
        with open(in_root(f), 'rb') as g:
            h.update(g.read())
    return h.hexdigest()[:16]


def default_out(set_name, budget, workers, model, seed):
    tag = f'{set_name}_cpsat{budget:g}s_w{workers}'
    if model != 'compact':
        tag += f'_{model}'
    if seed != 1:
        tag += f'_s{seed}'
    return os.path.join('or_solution', 'PUBFAMILY', tag)


# ---------------------------------------------------------------- inputs
def set_names(set_dir):
    """Sorted descriptor names of a family set (the row order of every result
    array); every combination must hold the same names."""
    names = None
    for r in R8:
        ps = sorted(glob.glob(os.path.join(set_dir, 'desc', r, '*.desc.json')))
        nm = [os.path.basename(p)[:-len('.desc.json')] for p in ps]
        if names is None:
            names = nm
        elif nm != names:
            raise SystemExit(f'REFUSING: {set_dir}: combination {r} holds other '
                             f'descriptor names than N')
    if not names:
        raise SystemExit(f'REFUSING: no descriptors under {set_dir}/desc')
    return names


def desc_path(set_dir, regime, name):
    return os.path.join(set_dir, 'desc', regime, f'{name}.desc.json')


def load_case(set_dir, regime, name):
    """(Case, record) of one stored public record."""
    from constraint_family import cpsat_arms as A
    from constraint_family.public_rules import load_family_record
    jl, pt, desc, rec = load_family_record(desc_path(set_dir, regime, name), root=ROOT)
    return A.Case(jl, pt, desc, name, regime), rec


def check_members(set_dir, names):
    """Problems that would break the comparison across combinations: a
    combination whose record activates other rules than its name says, or
    whose lags, classes, setup matrix, outages, horizon, base instance or
    descriptor seed differ from the other combinations of the same base
    instance; a field the models do not implement (setup_mult)."""
    probs = []
    for nm in names:
        recs = {}
        for r in R8:
            with open(desc_path(set_dir, r, nm)) as f:
                recs[r] = json.load(f)
        lsw = recs['LSW']
        for r, d in recs.items():
            want = ('L' in r, 'S' in r, 'W' in r)
            if tuple(bool(x) for x in d['active']) != want:
                probs.append(f'{nm} {r}: active {d["active"]} is not {list(want)}')
            if d.get('setup_mult') is not None:
                probs.append(f'{nm} {r}: setup_mult is not modelled')
            for k in ('op_class', 'horizon', 'base_instance', 'desc_seed'):
                if d.get(k) != lsw.get(k):
                    probs.append(f'{nm} {r}: {k} differs from LSW')
            if want[0] and d['lag'] != lsw['lag']:
                probs.append(f'{nm} {r}: lags differ from LSW')
            if not want[0] and any(d['lag']):
                probs.append(f'{nm} {r}: nonzero lags without the lag rule')
            if want[1] and d['setup'] != lsw['setup']:
                probs.append(f'{nm} {r}: setup matrix differs from LSW')
            if want[2] and d['windows'] != lsw['windows']:
                probs.append(f'{nm} {r}: outages differ from LSW')
            if d.get('release') != lsw.get('release'):
                probs.append(f'{nm} {r}: release dates differ from LSW')
    return probs


def load_stored(stored_dir, regimes, n):
    """{regime: {method: makespans [n]}} of every Result_<method>_<regime>.npy
    (column 0, rows in sorted-name order)."""
    out = {r: {} for r in regimes}
    if stored_dir is None:
        return out
    for p in sorted(glob.glob(os.path.join(stored_dir, 'Result_*.npy'))):
        stem = os.path.basename(p)[len('Result_'):-len('.npy')]
        method, _, r = stem.rpartition('_')
        if r not in out:
            continue
        a = np.load(p)
        if a.ndim != 2 or a.shape[0] != n or not np.all(np.isfinite(a[:, 0])):
            raise SystemExit(f'REFUSING: {show(p)} has shape {a.shape}, want ({n}, 2)')
        out[r][method] = a[:, 0].astype(float)
    return out


def load_best_known(path, names):
    """{name: (best-known makespan, lower bound or nan, marked optimal)}."""
    if path is None:
        return {}
    with open(path) as f:
        rows = {r['instance']: r for r in csv.DictReader(f)}
    out = {}
    for nm in names:
        r = rows.get(nm)
        if r is None or not r['best_known_makespan']:
            continue
        bk = float(r['best_known_makespan'])
        lb = float(r['lower_bound_if_known']) if r['lower_bound_if_known'] else float('nan')
        proved = r['optimal'].startswith('yes') or (np.isfinite(lb) and lb == bk)
        out[nm] = (bk, lb, bool(proved))
    return out


# ---------------------------------------------------------------- warm start
_ENV = {}


def _env_modules():
    """The environment and the masked rules, imported once. params.py parses
    sys.argv at import time, so it sees only a CPU device option."""
    if not _ENV:
        saved = sys.argv
        sys.argv = saved[:1] + ['--device', 'cpu']
        try:
            import torch
            torch.set_num_threads(1)
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass
            from params import configs
            import fjsp_env_same_op_nums
            fjsp_env_same_op_nums.EnvState.device = torch.device('cpu')
            from constraint_family.fjsp_env_family import FJSPEnvFamily, resolve_delta_mode
            from common_utils import masked_rule_action
        finally:
            sys.argv = saved
        configs.device = 'cpu'
        _ENV.update(configs=configs, Env=FJSPEnvFamily, delta=resolve_delta_mode,
                    rule=masked_rule_action)
    return _ENV


def rule_rollouts(case):
    """{rule: (makespan, amch [N], ct [N])} of the four masked rules in
    FJSPEnvFamily, exactly as eval_family.py --rule_set masked --first_mask
    exact --mask_ties exact rolls them out."""
    E = _env_modules()
    jl, pt = case.jl, case.pt
    M = pt.shape[1]
    out = {}
    for rule in RULES:
        env = E['Env'](jl.shape[0], M, family_mode=True, mask_mode='nondelay',
                       pair_extra='none', first_mask='exact', mask_ties='exact')
        dm = E['delta'](E['configs'])
        env.delta_mode = dm
        env.reward_delta_env = (dm != 'zero')
        env.set_initial_data([jl], [pt], descriptor_list=[case.desc],
                             op_type_list=None, mch_type_list=None)
        np.random.seed(E['configs'].seed_test)
        amch = np.full(pt.shape[0], -1, dtype=int)
        while not env.done().all():
            a = int(E['rule'](rule, env))
            amch[env.candidate[0, a // M]] = a % M
            env.step(np.array([a]))
        out[rule] = (float(env.current_makespan[0]), amch,
                     np.asarray(env.true_op_ct[0], dtype=float).copy())
    return out


def make_hint(case, stored=None, row=None):
    """Best masked-rule schedule as an integer (amch, st, ct) hint, with its
    metadata. Refuses (ValueError) when a rollout fails validation or does not
    reproduce the stored makespan of its rule."""
    from constraint_family import cpsat_arms as A
    rolls = rule_rollouts(case)
    meta = {'warm_all_rules': {k: v[0] for k, v in rolls.items()}}
    if stored is not None:
        diff = {k: (v[0], float(stored[k][row])) for k, v in rolls.items()
                if k in stored and abs(v[0] - float(stored[k][row])) > EPS}
        missing = [k for k in RULES if k not in stored]
        if missing:
            raise ValueError(f'no stored array for rules {missing}')
        if diff:
            raise ValueError(f'rule rollouts do not reproduce the stored makespans '
                             f'(rollout, stored): {diff}')
        meta['stored_rules_match'] = True
    best = min(RULES, key=lambda k: (rolls[k][0], RULES.index(k)))
    hint = A.schedule_from_hint(case, rolls[best][1], rolls[best][2])
    bad = A.validate_all(case, *hint)
    if bad:
        raise ValueError(f'hint schedule of {best} fails validation: {bad[:2]}')
    if abs(float(hint[2].max()) - rolls[best][0]) > EPS:
        raise ValueError('hint makespan differs from the rollout makespan')
    meta.update(warm_makespan=rolls[best][0], warm_rule=best,
                warm_source=f'masked rule {best} in FJSPEnvFamily (best of '
                            f'{", ".join(RULES)}; first_mask and mask_ties exact)')
    return hint, meta


# ---------------------------------------------------------------- campaign gates
def state_lines(path):
    """Non-empty lines of a state file that do not start with NOTE (a NOTE
    line is a remark and leaves the state before it), or None without file."""
    try:
        with open(path) as f:
            lines = [ln.strip() for ln in f if ln.strip()]
    except OSError:
        return None
    return [ln for ln in lines if not ln.startswith('NOTE')]


def last_state(path):
    """Last state line of a state file, or None."""
    lines = state_lines(path)
    return lines[-1] if lines else None


def state_is(cond):
    """True when the last state line of cond = (file, prefix) starts with prefix."""
    if not cond:
        return False
    s = last_state(cond[0])
    return s is not None and s.startswith(cond[1])


def wait_for_start(start, stop=None, poll=10.0, log=print, report_every=600.0):
    """Block until the start condition holds: the last state line starts with
    the prefix, or a state line appended while waiting did (so a state that
    passes between two polls is not missed). Returns False (do not start)
    when the stop condition holds first."""
    if not start:
        return not state_is(stop)
    t0 = last = time.time()
    n0 = len(state_lines(start[0]) or [])
    log(f'waiting for "{start[1]}" in {start[0]} (now: {last_state(start[0])!r})')

    def met():
        lines = state_lines(start[0]) or []
        return any(ln.startswith(start[1]) for ln in lines[n0:] + lines[-1:])

    while not met():
        if state_is(stop):
            log('stop condition reached before the start condition')
            return False
        if time.time() - last >= report_every:
            last = time.time()
            log(f'still waiting ({(last - t0) / 60:.0f} min), state now: '
                f'{last_state(start[0])!r}')
        time.sleep(poll)
    log(f'start condition met: {last_state(start[0])!r}')
    return True


def wait_while_hold(hold_file, stop=None, poll=30.0, log=print):
    """Block while the hold file exists. Returns False when the stop
    condition holds (do not take the next solve)."""
    said = False
    while hold_file and os.path.exists(hold_file):
        if state_is(stop):
            return False
        if not said:
            log(f'hold file {hold_file} present: no new solve')
            said = True
        time.sleep(poll)
    if said:
        log('hold file removed')
    return not state_is(stop)


# ---------------------------------------------------------------- records
def write_json_atomic(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f'{path}.tmp{os.getpid()}'
    try:
        with open(tmp, 'w') as f:
            json.dump(obj, f)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def fail(out_dir, msg):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, FAIL_LOG), 'a') as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")


def record_validity(rec):
    """(valid, reason, timing_valid) of a record; see the module docstring."""
    share = rec.get('cpu_share') or 0.0
    cores_ok = sorted(rec.get('affinity') or []) == sorted(rec.get('cores') or [-1])
    timing = bool(rec.get('validated')) and not rec.get('development_run') \
        and cores_ok and share >= SHARE_MIN
    if not rec.get('validated'):
        return False, 'not validated', False
    if rec.get('development_run'):
        return False, 'development run, not a measurement', False
    if not cores_ok:
        return False, f'affinity {rec.get("affinity")} is not the reserved cores', False
    if rec.get('status') == 'OPTIMAL':
        return True, 'proved optimal (exact whatever the CPU share)', timing
    if share >= SHARE_MIN:
        return True, f'cpu_share >= {SHARE_MIN}', timing
    return False, f'cpu_share {share} < {SHARE_MIN} and not proved optimal: re-run', False


def replay_makespan(case, amch, st, ct):
    """Makespan of the schedule's machine sequences replayed through the
    reference simulator (every operation as early as the rules allow, in
    the order of its setup start)."""
    from constraint_family import cpsat_arms as A
    from constraint_family import reference_sim
    d = A.derive(case, amch, st, ct)
    order = np.lexsort((np.arange(case.N), d['block']))
    acts = [(int(case.job[o]), int(amch[o])) for o in order]
    return float(reference_sim.simulate(case.jl, case.pt, case.desc, acts)['makespan'])


def solve_task(t):
    """One solve in a fresh process: writes the record or a failure line."""
    cores = t.get('cores')
    if cores:
        os.sched_setaffinity(0, cores)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    import ortools
    from constraint_family import cpsat_arms as A
    out_dir = t['out_dir']
    tag = f"{t['regime']} {t['name']}"
    try:
        case, drec = load_case(t['set_dir'], t['regime'], t['name'])
        hint = tuple(np.asarray(x, dtype=int) for x in t['hint'])
        bad = A.validate_all(case, *hint)
        if bad:
            fail(out_dir, f'{tag}: hint fails validation: {bad[:2]}')
            return
        hint_ms = int(hint[2].max())
        load0 = [round(x, 2) for x in os.getloadavg()]
        started = time.strftime('%Y-%m-%d %H:%M:%S')
        w0, c0 = time.time(), time.process_time()
        if t['model'] == 'compact':
            b = A.build_model(case, base=hint, horizon=hint_ms)
            b.add_hint(*hint)
            model, extract, horizon = b.model, b.extract, hint_ms
        else:
            built = A.build_reference_model(case, warmstart=(hint[0], hint[2]),
                                            complete_hint=True)
            model, extract, horizon = built['model'], built['extract'], built['horizon']
        build_s, cpu_build = time.time() - w0, time.process_time() - c0
        log = []
        res = A.solve_built(model, extract, t['budget'], t['workers'],
                            seed=t['seed'], log=log)
        wall, cpu = time.time() - w0, time.process_time() - c0
        ended = time.strftime('%Y-%m-%d %H:%M:%S')
        load1 = [round(x, 2) for x in os.getloadavg()]
    except Exception as e:                      # recorded, never silently dropped
        fail(out_dir, f'{tag}: {type(e).__name__}: {e}')
        return
    if res['schedule'] is None:
        fail(out_dir, f"{tag}: no schedule returned (status {res['status']})")
        return
    amch, st, ct = res['schedule']
    viol = A.validate_all(case, amch, st, ct)
    if viol:
        fail(out_dir, f'{tag}: returned schedule fails validation: {viol[:2]}')
        return
    ms = float(np.max(ct))
    if abs(ms - res['objective']) > EPS:
        fail(out_dir, f"{tag}: schedule makespan {ms} differs from the objective "
                      f"{res['objective']}")
        return
    rep = replay_makespan(case, amch, st, ct)
    if rep > ms + EPS:
        fail(out_dir, f'{tag}: reference simulator replay {rep} is longer than {ms}')
        return
    size = A.proto_size(model)
    hint_line = [x.strip() for x in log if 'solution hint' in x.lower()][:1]
    subsolvers = [x.strip() for x in log if 'subsolver' in x][:4]
    anytime = res['anytime']
    proved = res['status'] == 'OPTIMAL'
    rec = {
        'set': t['set'], 'instance': t['name'], 'index': t['index'],
        'regime': t['regime'], 'descriptor': show(desc_path(t['set_dir'], t['regime'], t['name'])),
        'desc_seed': drec.get('desc_seed'), 'rules_version': drec.get('rules_version'),
        'model': t['model'], 'objective': ms, 'bound': res['bound'],
        'status': res['status'], 'proved_optimal': proved,
        'gap_to_bound': (ms - res['bound']) / ms if res['bound'] is not None else None,
        'time_to_last_improvement_s': anytime[-1][0] if anytime else None,
        'solve_time_s': res['solve_s'], 'build_s': round(build_s, 3),
        'wall_total_s': round(wall, 3), 'anytime': anytime,
        'n_solutions': len(anytime),
        'warm_makespan': t['meta']['warm_makespan'], 'warm_rule': t['meta']['warm_rule'],
        'warm_source': t['meta']['warm_source'],
        'warm_all_rules': t['meta']['warm_all_rules'],
        'stored_rules_match': t['meta'].get('stored_rules_match'),
        'hint': 'complete', 'hint_log': hint_line, 'subsolvers': subsolvers,
        'budget_s': t['budget'], 'workers': t['workers'], 'seed': t['seed'],
        'horizon': int(horizon),
        'model_size': {'variables': size['variables'],
                       'constraints': size['constraints'],
                       'circuit_arcs': size['circuit_arcs']},
        'cpu_time_s': round(cpu, 3), 'cpu_build_s': round(cpu_build, 3),
        'cpu_share': round(cpu / max(wall * t['workers'], 1e-9), 4),
        'cores': list(cores or []), 'affinity': sorted(os.sched_getaffinity(0)),
        'loadavg': load0, 'loadavg_end': load1,
        'peak_rss_gb': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 4),
        'started': started, 'ended': ended,
        'validated': True, 'violations': [],
        'validators': ['cpsat_arms.validate_strict', 'checks.check_schedule_validation (C6)'],
        'replay_makespan': rep,
        'assigned_mch': [int(x) for x in amch], 'op_st': [int(x) for x in st],
        'op_ct': [int(x) for x in ct],
        'ortools': ortools.__version__, 'python': platform.python_version(),
        'code_sha': t.get('code_sha'), 'development_run': bool(t.get('dev')),
    }
    rec['valid'], rec['valid_reason'], rec['timing_valid'] = record_validity(rec)
    write_json_atomic(t['out'], rec)


# ---------------------------------------------------------------- consistency
def instance_violations(name, row, recs, stored, bk):
    """Disagreements between the records of one base instance (recs:
    {regime: record}), the stored makespans of the rules and policies
    (stored: {regime: {method: array}}, row = sorted-name index) and the
    best-known table (bk: name -> (value, lower bound, marked optimal))."""
    v = []
    lb_mono = {}
    for r, rec in recs.items():
        bs = [recs[s]['bound'] for s in subsets(r) if s in recs and recs[s]['bound'] is not None]
        lb_mono[r] = max(bs) if bs else None
    if 'N' in recs and name in bk:
        val, lb, opt = bk[name]
        rec = recs['N']
        if rec['bound'] is not None and rec['bound'] > val + EPS:
            v.append(f'{name} N: bound {rec["bound"]} above the best-known makespan {val}')
        if np.isfinite(lb) and rec['objective'] < lb - EPS:
            v.append(f'{name} N: makespan {rec["objective"]} below the lower bound {lb}')
        if rec['proved_optimal'] and opt and abs(rec['objective'] - val) > EPS:
            v.append(f'{name} N: proved optimum {rec["objective"]} differs from the '
                     f'best-known optimum {val}')
    for r, rec in recs.items():
        for s in subsets(r):
            if s == r or s not in recs or recs[s]['bound'] is None:
                continue
            if rec['objective'] < recs[s]['bound'] - EPS:
                v.append(f'{name}: {r} makespan {rec["objective"]} below the {s} '
                         f'bound {recs[s]["bound"]} although {r} only adds rules')
        for method, arr in (stored.get(r) or {}).items():
            if lb_mono[r] is not None and arr[row] < lb_mono[r] - EPS:
                v.append(f'{name} {r}: stored makespan {arr[row]} of {method} below '
                         f'the CP-SAT bound {lb_mono[r]}')
    return v


def read_records(out_dir):
    """{regime: {name: record}} of every record in a run folder."""
    out = {}
    for r in R8:
        for p in glob.glob(os.path.join(out_dir, r, '*.json')):
            with open(p) as f:
                d = json.load(f)
            out.setdefault(r, {})[d['instance']] = d
    return out


def load_refs(out_dir, names=None, data_root=DATA_ROOT):
    """Per combination, arrays in the sorted-name order of the result arrays:
    {regime: {'names', 'objective', 'bound', 'bound_mono', 'proved',
    'valid'}}. Missing records give nan and False. bound_mono is the largest
    bound over the combination and its sub-combinations (a valid lower bound
    because switching a rule on can only delay). Validity is recomputed by
    record_validity."""
    recs = read_records(in_root(out_dir))
    if names is None:
        sets = {d['set'] for rr in recs.values() for d in rr.values()}
        if len(sets) != 1:
            raise ValueError(f'{out_dir}: records of {len(sets)} sets; pass names')
        names = set_names(os.path.join(in_root(data_root), sets.pop()))
    out = {}
    for r in R8:
        rr = recs.get(r, {})
        obj = np.array([rr[n]['objective'] if n in rr else np.nan for n in names], float)
        bnd = np.array([rr[n]['bound'] if n in rr and rr[n]['bound'] is not None
                        else np.nan for n in names], float)
        out[r] = {'names': list(names), 'objective': obj, 'bound': bnd,
                  'proved': np.array([n in rr and rr[n]['status'] == 'OPTIMAL'
                                      for n in names]),
                  'valid': np.array([n in rr and record_validity(rr[n])[0]
                                     for n in names])}
    for r in R8:
        bs = np.vstack([out[s]['bound'] for s in subsets(r)])
        with np.errstate(all='ignore'):
            out[r]['bound_mono'] = np.where(np.isnan(bs).all(0), np.nan, np.nanmax(
                np.where(np.isnan(bs), -np.inf, bs), axis=0))
    return out


# ---------------------------------------------------------------- phases
def resolve(a):
    a.set_dir = os.path.join(in_root(a.data_root), a.set)
    if not os.path.isdir(a.set_dir):
        raise SystemExit(f'REFUSING: no family set at {a.set_dir}')
    a.out_shown = a.out or default_out(a.set, a.budget, a.workers, a.model, a.seed)
    a.out_dir = in_root(a.out_shown)
    if a.stored == 'none':
        a.stored_dir = None
    else:
        a.stored_dir = in_root(a.stored or os.path.join(STORED_ROOT, a.set + '_xmask'))
        if not os.path.isdir(a.stored_dir):
            raise SystemExit(f'REFUSING: no stored results at {a.stored_dir} '
                             f'(pass --stored none to solve without the check)')
    if a.best_known == 'none' or (a.best_known is None and a.set not in BEST_KNOWN):
        a.bk_path = None
    else:
        a.bk_path = in_root(a.best_known or BEST_KNOWN[a.set])
    a.names = set_names(a.set_dir)
    order = a.order or list(ORDER)
    if sorted(order) != sorted(R8):
        raise SystemExit(f'REFUSING: --order must list each of {" ".join(R8)} once')
    a.regimes = [r for r in order if r in (a.regimes or order)]
    a.idx = parse_indices(a.instances, len(a.names))
    a.cores_list = parse_cores(a.cores) if a.cores else []
    a.hold = in_root(a.hold_file) if a.hold_file != 'none' else None
    return a


def build_tasks(a):
    tasks = []
    for r in a.regimes:
        for i in a.idx:
            nm = a.names[i]
            out = os.path.join(a.out_dir, r, f'{nm}.json')
            if os.path.exists(out):
                continue
            tasks.append({'set': a.set, 'set_dir': a.set_dir, 'name': nm, 'index': i,
                          'regime': r, 'budget': float(a.budget), 'workers': a.workers,
                          'seed': a.seed, 'model': a.model, 'cores': a.cores_list,
                          'dev': a.dev, 'out_dir': a.out_dir, 'out': out})
    return tasks


def phase_plan(a):
    tasks = build_tasks(a)
    probs = check_members(a.set_dir, a.names)
    per = {r: sum(t['regime'] == r for t in tasks) for r in a.regimes}
    h = len(tasks) * a.budget / 3600
    print(f'set {a.set}: {len(a.names)} base instances; selected {len(a.idx)} x '
          f'{len(a.regimes)} combinations; {len(tasks)} solves to run '
          f'(existing records skipped)')
    print('solves per combination, in run order: ' +
          ', '.join(f'{r} {per[r]}' for r in a.regimes))
    print(f'model {a.model}, budget {a.budget:g} s, {a.workers} workers, cores '
          f'{a.cores or "(none given)"}, seed {a.seed}')
    n_set = sum(t['regime'] in SETUP for t in tasks)
    print(f'worst case (every solve to the cap): {h:.2f} h; the {n_set} solves with '
          f'setups alone: {n_set * a.budget / 3600:.2f} h (combinations without '
          f'setups usually end at a proof long before the cap); plus about 3 s per '
          f'solve for the warm start, the child process and the checks '
          f'({len(tasks) * 3 / 3600:.2f} h)')
    print(f'folder: {a.out_shown}')
    print(f'stored results: {show(a.stored_dir) if a.stored_dir else "none"}; '
          f'best-known table: {show(a.bk_path) if a.bk_path else "none"}')
    print('state files: start ' + (f'"{a.start_when[1]}" in {a.start_when[0]}'
                                    if a.start_when else 'none') +
          '; stop ' + (f'"{a.stop_when[1]}" in {a.stop_when[0]}' if a.stop_when else 'none') +
          f'; hold file {show(a.hold) if a.hold else "none"}')
    print(f'descriptor members across combinations: '
          f'{"consistent" if not probs else f"{len(probs)} problems, e.g. {probs[:3]}"}')
    shown = tasks if len(tasks) <= 6 else tasks[:3] + [None] + tasks[-3:]
    for t in shown:
        print('   ...' if t is None else f"   {t['regime']:<4} {t['name']} -> {show(t['out'])}")


def phase_check(a, quiet=False):
    """Item-5 checks over a folder; returns the list of violations."""
    recs = read_records(a.out_dir)
    stored = load_stored(a.stored_dir, R8, len(a.names))
    bk = load_best_known(a.bk_path, a.names)
    viol = []
    for i, nm in enumerate(a.names):
        inst = {r: recs[r][nm] for r in R8 if nm in recs.get(r, {})}
        if inst:
            viol += instance_violations(nm, i, inst, stored, bk)
    if quiet:
        return viol
    print(f'folder {a.out_shown}')
    print('| combination | records | proved optimal | valid | mean makespan | '
          'mean gap to bound % | median end s | max end s | at cap |')
    print('|---|---|---|---|---|---|---|---|---|')
    for r in ORDER:
        rr = list(recs.get(r, {}).values())
        if not rr:
            continue
        gaps = [100 * x['gap_to_bound'] for x in rr if x['gap_to_bound'] is not None]
        ends = [x['solve_time_s'] for x in rr]
        print(f"| {r} | {len(rr)} | {sum(x['proved_optimal'] for x in rr)} | "
              f"{sum(record_validity(x)[0] for x in rr)} | "
              f"{np.mean([x['objective'] for x in rr]):.1f} | {np.mean(gaps):.2f} | "
              f"{np.median(ends):.1f} | {max(ends):.1f} | "
              f"{sum(not x['proved_optimal'] for x in rr)} |")
    n_rec = recs.get('N', {})
    if bk and n_rec:
        eq = sum(abs(d['objective'] - bk[nm][0]) < EPS for nm, d in n_rec.items() if nm in bk)
        both = sum(d['proved_optimal'] and bk[nm][2] for nm, d in n_rec.items() if nm in bk)
        print(f'N against best known: {eq}/{len(n_rec)} equal; {both} proved here '
              f'and marked optimal there')
    print(f'violations: {len(viol)}')
    for x in viol[:20]:
        print('  ', x)
    return viol


def stop_campaign(a, msg):
    fail(a.out_dir, f'STOP: {msg}')
    with open(os.path.join(a.out_dir, STOP_MARK), 'a') as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")
    print(f'STOPPED: {msg}', flush=True)


def phase_solve(a):
    if not a.cores_list:
        raise SystemExit('REFUSING: --cores is required (the exclusive cores of the solves)')
    allowed = set(range(os.cpu_count() or 1))
    if not set(a.cores_list) <= allowed:
        raise SystemExit(f'REFUSING: cores {a.cores_list} are not on this machine')
    if a.workers > len(a.cores_list) and not a.dev:
        raise SystemExit(f'REFUSING: {a.workers} workers on {len(a.cores_list)} cores '
                         f'(--dev allows it for development runs, which are not '
                         f'measurements)')
    if os.path.exists(os.path.join(a.out_dir, STOP_MARK)):
        raise SystemExit(f'REFUSING: {show(a.out_dir)}/{STOP_MARK} exists; read it and '
                         f'remove it before resuming')
    probs = check_members(a.set_dir, a.names)
    if probs:
        raise SystemExit(f'REFUSING: descriptor members differ across combinations: '
                         f'{probs[:5]}')
    tasks = build_tasks(a)
    sha0 = code_fingerprint()
    print(f'{len(tasks)} solves, model {a.model}, budget {a.budget:g} s, '
          f'{a.workers} workers on cores {a.cores_list}, seed {a.seed}, '
          f'folder {a.out_shown}, code {sha0}', flush=True)
    if not tasks:
        return
    stored = load_stored(a.stored_dir, a.regimes, len(a.names))
    bk = load_best_known(a.bk_path, a.names)
    start = tuple(a.start_when) if a.start_when else None
    stop = tuple(a.stop_when) if a.stop_when else None

    def log(msg):
        print(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)

    if not wait_for_start(start, stop, poll=a.poll, log=log):
        return
    os.makedirs(a.out_dir, exist_ok=True)
    ctx = mp.get_context('spawn')
    t_camp, n_done = time.time(), 0
    for k, t in enumerate(tasks):
        if state_is(stop):
            log(f'stop condition met ({last_state(stop[0])!r}): no new solve')
            break
        if not wait_while_hold(a.hold, stop, poll=a.poll, log=log):
            log('stop condition met during the hold: no new solve')
            break
        if os.path.exists(t['out']):
            continue
        tag = f"{t['regime']} {t['name']}"
        try:
            case, _ = load_case(a.set_dir, t['regime'], t['name'])
            hint, meta = make_hint(case, stored.get(t['regime']) if a.stored_dir else None,
                                   t['index'])
        except Exception as e:                  # a refused or failed warm start
            stop_campaign(a, f'{tag}: warm start refused: {type(e).__name__}: {e}')
            break
        t['hint'] = [x.tolist() for x in hint]
        t['meta'] = meta
        if code_fingerprint() != sha0:
            stop_campaign(a, f'{tag}: a source file of the solve changed since the '
                             f'campaign started ({", ".join(CODE_FILES)})')
            break
        t['code_sha'] = sha0
        t0 = time.time()
        p = ctx.Process(target=solve_task, args=(t,))
        p.start()
        log(f'[{k + 1}/{len(tasks)}] {tag} solving (child pid {p.pid})')
        p.join()
        fresh = os.path.exists(t['out']) and os.path.getmtime(t['out']) >= t0
        if not fresh:
            log(f'{tag}: NO RECORD (exit {p.exitcode}, see {FAIL_LOG})')
            continue
        n_done += 1
        with open(t['out']) as f:
            rec = json.load(f)
        recs = read_records(a.out_dir)
        inst = {r: recs[r][t['name']] for r in R8 if t['name'] in recs.get(r, {})}
        viol = instance_violations(t['name'], t['index'], inst, stored, bk)
        el = (time.time() - t_camp) / 3600
        log(f"{tag} {rec['status']} obj={rec['objective']:g} bound={rec['bound']:g} "
            f"warm={rec['warm_makespan']:g} end={rec['solve_time_s']:.1f}s "
            f"share={rec['cpu_share']} valid={rec['valid']} ({el:.2f} h elapsed)")
        if viol:
            stop_campaign(a, f'{tag}: ' + ' | '.join(viol))
            break
    log(f'done: {n_done} records written')


def main(argv=None):
    ap = argparse.ArgumentParser(description='CP-SAT references for a public family set')
    ap.add_argument('--phase', default='solve', choices=('solve', 'plan', 'check'))
    ap.add_argument('--set', default=DEFAULT_SET)
    ap.add_argument('--data_root', default=DATA_ROOT)
    ap.add_argument('--regimes', nargs='+', choices=R8, default=None)
    ap.add_argument('--order', nargs=8, choices=R8, default=None,
                    help='run order of the combinations (default N S LS SW LSW L W LW)')
    ap.add_argument('--instances', default='all',
                    help='index ranges in sorted-name order, e.g. 0-2 or 0,5,7-9')
    ap.add_argument('--budget', type=float, default=300.0)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--cores', default=None, help='exclusive cores of the solves, e.g. 0-1')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--model', choices=MODELS, default='compact')
    ap.add_argument('--out', default=None)
    ap.add_argument('--stored', default=None,
                    help='folder of stored Result_*.npy arrays (default '
                         'test_results/PUBFAMILY/<set>_xmask); none to skip')
    ap.add_argument('--best_known', default=None, help='best-known table; none to skip')
    ap.add_argument('--start_when', nargs=2, metavar=('FILE', 'PREFIX'), default=None)
    ap.add_argument('--stop_when', nargs=2, metavar=('FILE', 'PREFIX'), default=None)
    ap.add_argument('--hold_file', default=HOLD_FILE)
    ap.add_argument('--poll', type=float, default=10.0, help='seconds between state checks')
    ap.add_argument('--dev', action='store_true',
                    help='development run: more workers than cores allowed, records '
                         'marked development_run and never valid')
    a = resolve(ap.parse_args(argv))
    if a.phase == 'plan':
        phase_plan(a)
    elif a.phase == 'check':
        sys.exit(1 if phase_check(a) else 0)
    else:
        phase_solve(a)


if __name__ == '__main__':
    main()
