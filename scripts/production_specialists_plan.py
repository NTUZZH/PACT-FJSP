r"""Job plan, pre-flight and completeness check of scripts/run_production_specialists_eval.sh.

The specialists at the production batches, in the environment of the paper (exact non-delay
mask): each specialist xmask-solo-<R>-s<seed> (seeds 301 to 305) on its own combination R of
the 30-instance sets 50x25n30 and 80x25n30, 2 x 8 x 5 = 80 jobs. The standard result folders
of the two sets (arm_registry.res_dir('50x25n30'), ...('80x25n30')) hold the policy and the
dispatching rules only; these jobs add the specialists.

THE JOB (job())
  The evaluation plan's job of the policy on the same set (scripts/eval_newseeds_plan.py, group
  n30: xmask-v1-s<seed> on the eight combinations, exporting its schedules), for the same seed,
  with these keys exchanged and no other: the model, its checkpoint and training log (the
  specialist's), the combination list (R alone), the schedule export (none), the tag and the
  result folder (TAG below), and the bookkeeping keys id, group, campaign, files, cost and
  pre-flight. The command is built by build_cmd of scripts/run_eval_newseeds.sh, as every
  other evaluation of the plan, so the resolved commands differ in --family_model, --regimes,
  --tag and the absence of --save_schedules alone (besides the staging --out_root, which every
  runner sets): CPU, OMP and MKL threads 1, --device cpu --device_id "", --torch_threads 1,
  --first_mask exact --mask_ties exact, --methods greedy, the set's data and base folders.

RESULT FOLDER (TAG = _xmask_spec)
  <RESULTS>/<set>_xmask_spec, e.g. test_results/FAMILY/50x25n30_xmask_spec, the folder
  eval_family.py writes for --data data/FAMILY/<set> --tag _xmask_spec. Not the standard folder
  <set>_xmask: scripts/p120_lower_bound_reference.py (exact_arrays) reads every
  Result_*.npy of arm_registry.res_dir(<set>) as a feasible makespan of its cells (best known
  makespan, self-check, exclusion of stored solver bounds above the best known), so a
  specialist array there would change that analysis of the policy's gap to the bound.
  scripts/p128_production_specialists.py reads the specialists from this folder.

SEEDS AND TRAINING
  Seeds 301 to 305 (the plan's jobs exist for these only). A run is evaluated only when it is
  trained: checkpoint present and completion line 'done in' in the training log
  (scripts/further_sets_plan.py trained, the plan's own condition).

LANES (lane())
  Jobs in a fixed order, longest first (cost, then id); lane i of K takes the jobs whose index
  is i modulo K, so the K lanes are disjoint and together hold every job.

PRE-FLIGHT (preflight())
  The job of xmask-solo-S-s301 on 50x25n30 limited to the first PRE_N = 2 instances, into a
  staging folder; its makespans must equal, bit for bit, the default level of
  scripts/p103_five_classes.py for the same run, instance and combination: pre_stored(),
  p103's result_path under its DEFAULT_ROOT for level default, 50x25n30, S and
  greedy+10x25+family+xmask-solo-S-s301 (key makespan of the npz; p103's own rollout of the
  stored descriptors, scripts/p101_mask_controls.py PolicyRunner). It proves that
  eval_family.py under the job's options reproduces an independent evaluation path of the
  same run.

TIME (estimate())
  Seconds per rollout measured on the same sets by the same evaluator: the mean of the time
  column (column 1) of the stored arrays of the policy of the n30 jobs in res_dir(<set>), the
  same network size and evaluation path; JOB_OVERHEAD of the plan added per job. The plan's
  table eval_newseeds_plan.SPEED (measured 2026-11-11) is printed beside it as a second figure.

Subcommands:
  jobs [i/K]               one line per job of the lane (default 0/1), the format of
                           'eval_newseeds_plan.py jobs'
  template <id>            the plan's n30 job of the policy that job <id> is built from
  preflight                the pre-flight job line, then the stored array it must reproduce
  compare <folder>         compare the pre-flight's array in <folder> with the stored array
  trained <model>          exit 0 when the run is trained
  rows <n> <file> ...      exit 0 when every file exists and every .npy holds n rows
  check                    exit 0 when every file of every job exists, every array with one
                           row per test instance
  estimate [K ...]         jobs, rollouts and core-hours, in total and per lane for each K
"""
import contextlib
import functools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import arm_registry as REG  # noqa: E402

SETS = ('50x25n30', '80x25n30')
ENV = 'exact'
SEEDS = (301, 302, 303, 304, 305)
TAG = '_xmask_spec'
GROUP = 'n30spec'
PRE_SET, PRE_COMBO, PRE_SEED, PRE_N = '50x25n30', 'S', 301, 2
# keys of a job that differ from its template, the n30 job of the policy (module docstring)
EXCHANGED = ('id', 'group', 'campaign', 'model', 'ckpt', 'log', 'regimes', 'sched', 'tag',
             'folder', 'files', 'cost', 'pre')


class Refused(Exception):
    """A setting, a set or a job that this plan does not accept."""


def _plan():
    import eval_newseeds_plan as PLAN
    return PLAN


def require_exact():
    if REG.env() != ENV:
        raise Refused(f'the production specialists are evaluated in the {ENV} environment; the '
                      f'active environment is {REG.env()} (set PACT_ENV={ENV})')


def check_set(name):
    if name not in SETS:
        raise Refused(f'unknown set {name!r}; want one of {", ".join(SETS)}')
    return name


def results_root():
    return _plan().RESULTS


def folder(name):
    """Result folder of the specialists of a set."""
    return os.path.join(results_root(), check_set(name) + TAG)


@contextlib.contextmanager
def plan_setting(plan, groups, seed_list):
    old = plan.GROUPS, plan.EXACT_SEEDS
    plan.GROUPS, plan.EXACT_SEEDS = list(groups), list(seed_list)
    try:
        yield plan
    finally:
        plan.GROUPS, plan.EXACT_SEEDS = old


def template(name, seed):
    """The plan's n30 job of the policy xmask-v1-s<seed> on the set."""
    check_set(name)
    if seed not in SEEDS:
        raise Refused(f'the evaluation plan has n30 jobs of seeds {SEEDS}, not {seed}')
    P = _plan()
    with plan_setting(P, ['n30'], [seed]):
        js = [dict(j) for j in P.exact_jobs() if j['campaign'] == name]
    m = REG.PREFIX + REG.model('released', None, seed, ENV)
    js = [j for j in js if j['model'] == m]
    if len(js) != 1:
        raise Refused(f'the evaluation plan has {len(js)} n30 jobs of {m} on {name}, want one')
    j = js[0]
    if j['sched'] == '-' or j['regimes'].split() != list(REG.COMBOS) or j['tag'] != P.XS:
        raise Refused(f'the n30 job {j["id"]} is not the policy job with schedule export on '
                      f'the eight combinations')
    return j


def speed(name):
    """(seconds per rollout from the stored n30 arrays, seconds per rollout of the plan's
    SPEED table) of the set."""
    return _speed(check_set(name), REG.res_dir(name))


@functools.lru_cache(maxsize=None)
def _speed(name, res):
    P = _plan()
    t = []
    for s in SEEDS:
        for r in REG.COMBOS:
            p = REG.result_path('released', None, s, name, r)
            if os.path.exists(p):
                a = np.load(p)
                if a.ndim == 2 and a.shape[1] > 1:
                    t.append(a[:, 1])
    measured = float(np.mean(np.concatenate(t))) if t else None
    return measured, P.SPEED[P.size_key(P.N30[name][0])][0]


def job(name, combo, seed):
    """The job of the specialist of combo, seed, on the set (module docstring)."""
    P = _plan()
    if combo not in REG.COMBOS:
        raise Refused(f'unknown combination {combo!r}')
    t = template(name, seed)
    run = REG.model('specialist', combo, seed, ENV)
    model = REG.PREFIX + run
    fold = name + TAG
    out = os.path.join(t['out_root'], fold)
    n = len(P.instance_names(t['data']))
    measured, table = speed(name)
    per = measured if measured is not None else table
    return dict(t, id=f'exact:{name}_spec:{run}', group=GROUP, campaign=f'{name}_spec',
                model=model, ckpt=REG.checkpoint('specialist', combo, seed, ENV),
                log=REG.train_log('specialist', combo, seed, ENV), regimes=combo, sched='-',
                tag=TAG, folder=fold,
                files=[f'{out}/Result_greedy+{model}_{combo}.npy', f'{out}/summary_{model}{TAG}.md'],
                cost=per * n + P.JOB_OVERHEAD, pre='prodspec', n=n)


def jobs():
    """Every job, longest first (cost, then id)."""
    require_exact()
    out = [job(name, r, s) for name in SETS for r in REG.COMBOS for s in SEEDS]
    ids = [j['id'] for j in out]
    if len(ids) != len(set(ids)) or len(ids) != len(SETS) * len(REG.COMBOS) * len(SEEDS):
        raise Refused('the job list is not 80 distinct jobs')
    files = [f for j in out for f in j['files']]
    if len(files) != len(set(files)):
        raise Refused('two jobs share a file')
    return sorted(out, key=lambda j: (-j['cost'], j['id']))


def parse_lane(text):
    try:
        i, k = (int(x) for x in str(text).split('/'))
    except ValueError:
        raise Refused(f'LANE={text!r}: want <i>/<K> with 0 <= i < K')
    if not (k >= 1 and 0 <= i < k):
        raise Refused(f'LANE={text!r}: want <i>/<K> with 0 <= i < K')
    return i, k


def lane(text='0/1'):
    """The jobs of lane i of K: index i modulo K in the order of jobs()."""
    i, k = parse_lane(text)
    return [j for x, j in enumerate(jobs()) if x % k == i]


def job_line(j):
    P = _plan()
    return '|'.join(str(j[k]) for k in P.KEYS_JOB) + '|' + ' '.join(j['files'])


def template_of(job_id):
    """The n30 policy job that the job of this id is built from."""
    for j in jobs():
        if j['id'] == job_id:
            name = j['campaign'][:-len('_spec')]
            seed = int(j['model'].rsplit('-s', 1)[1])
            return dict(template(name, seed), pre='exact')
    raise Refused(f'no job {job_id!r}')


def pre_stored():
    """p103's default-level array of the pre-flight run (relative to the repository root)."""
    saved = sys.argv[:]
    try:
        import p103_five_classes as P103          # sets sys.argv for params.py at import
    finally:
        sys.argv = saved
    label = 'greedy+' + REG.PREFIX + REG.model('specialist', PRE_COMBO, PRE_SEED, ENV)
    return P103.result_path(P103.DEFAULT_ROOT, ENV, 'default', PRE_SET, PRE_COMBO, label)


def preflight():
    """(job, stored array): the job of the specialist of PRE_COMBO, seed PRE_SEED, on
    PRE_SET limited to PRE_N instances, and p103's stored array it must reproduce."""
    j = job(PRE_SET, PRE_COMBO, PRE_SEED)
    j = dict(j, limit=PRE_N, pre='-', id=f'preflight:{PRE_SET}_spec')
    return j, pre_stored()


def compare(staged_folder):
    """0 when the pre-flight's makespans in staged_folder equal the stored ones bit for bit."""
    j, stored = preflight()
    a_p = os.path.join(staged_folder, j['folder'], os.path.basename(j['files'][0]))
    if not os.path.exists(a_p):
        print(f'PREFLIGHT MISMATCH: {a_p} was not written')
        return 1
    if not os.path.exists(stored):
        print(f'PREFLIGHT MISMATCH: the stored array {stored} is missing')
        return 1
    a = np.load(a_p)
    a = (a[:, 0] if a.ndim > 1 else a).astype(float)
    with np.load(stored) as z:
        b = np.asarray(z['makespan'], dtype=float)[:PRE_N]
    if len(a) != PRE_N or not np.array_equal(a, b):
        print(f'PREFLIGHT MISMATCH: {a.tolist()} against {b.tolist()} ({stored})')
        return 1
    print(f'preflight identical: {len(a)} makespans of {j["model"]} on {PRE_SET}/{PRE_COMBO} '
          f'({a.tolist()}) equal {stored}')
    return 0


def trained(model):
    import further_sets_plan as F
    return F.trained(model)


def rows(n, files):
    import further_sets_plan as F
    return F.rows(n, files)


def check():
    """(complete files, missing or wrong files) of every job."""
    have, bad = 0, []
    for j in jobs():
        b = rows(j['n'], j['files'])
        bad += b
        have += len(j['files']) - len(b)
    return have, bad


def estimate(lanes=(8, 16)):
    js = jobs()
    out = dict(jobs=len(js), rollouts=sum(j['n'] for j in js),
               core_hours=sum(j['cost'] for j in js) / 3600.0, lanes={})
    for k in lanes:
        per = [sum(j['cost'] for j in lane(f'{i}/{k}')) / 3600.0 for i in range(k)]
        out['lanes'][k] = dict(longest_hours=max(per), shortest_hours=min(per),
                               jobs=[len(lane(f'{i}/{k}')) for i in range(k)])
    out['seconds_per_rollout'] = {s: speed(s) for s in SETS}
    return out


def main(argv):
    os.environ.setdefault('PACT_ENV', ENV)
    os.environ.setdefault('PACT_SEEDS', f'{SEEDS[0]}-{SEEDS[-1]}')
    if not argv:
        print(__doc__)
        return 2
    cmd, args = argv[0], argv[1:]
    try:
        require_exact()
        if cmd == 'jobs' and len(args) <= 1:
            for j in lane(args[0] if args else '0/1'):
                print(job_line(j))
            return 0
        if cmd == 'template' and len(args) == 1:
            print(job_line(template_of(args[0])))
            return 0
        if cmd == 'preflight' and not args:
            j, stored = preflight()
            print(job_line(j))
            print(stored)
            return 0
        if cmd == 'compare' and len(args) == 1:
            return compare(args[0])
        if cmd == 'trained' and len(args) == 1:
            why = trained(args[0])
            for x in why:
                print(f'{args[0]}: {x}')
            return 0 if not why else 1
        if cmd == 'rows' and len(args) >= 2:
            bad = rows(int(args[0]), args[1:])
            for x in bad:
                print(x)
            return 0 if not bad else 1
        if cmd == 'check' and not args:
            have, bad = check()
            print(f'production specialists: {have} files complete, {len(bad)} missing or of the '
                  f'wrong length (seeds {",".join(map(str, SEEDS))}; {", ".join(folder(s) for s in SETS)})')
            for x in bad[:10]:
                print(f'   {x}')
            if len(bad) > 10:
                print(f'   ... and {len(bad) - 10} more')
            return 0 if not bad else 1
        if cmd == 'estimate':
            ks = tuple(int(x) for x in args) or (8, 16)
            e = estimate(ks)
            sp = '; '.join(f'{s}: {m:.2f} s measured, {t:.2f} s in the plan\'s table'
                           for s, (m, t) in e['seconds_per_rollout'].items() if m is not None)
            print(f'production specialists: {e["jobs"]} jobs, {e["rollouts"]} rollouts, '
                  f'{e["core_hours"]:.2f} core-hours (seconds per rollout {sp})')
            for k, v in e['lanes'].items():
                print(f'  {k} lanes: {min(v["jobs"])} to {max(v["jobs"])} jobs per lane, '
                      f'{v["shortest_hours"]:.2f} to {v["longest_hours"]:.2f} core-hours per lane')
            return 0
    except Refused as e:
        print(f'REFUSING: {e}')
        return 2
    print(__doc__)
    return 2


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
