r"""The second five training seeds of the principal comparison and the extension rule.

THE EXTENSION RULE
  If, with five seeds, the interval of any single combination of the principal comparison
  (policy against specialist at 10x25) is inconclusive at the 1% margin, seeds 306 to 310
  of the policy and the eight specialists are trained and every verdict is recomputed with
  ten seeds. The verdict rule does not change.

  "Inconclusive at the 1% margin" is the verdict word 'inconclusive' in the key v1.0 of a
  component of section A of scripts/p88_family_inference.py: stats_family.verdict applied to
  the hierarchical 95% interval of the combination at delta = 1% (lo <= 0 < 1 <= hi, or an
  interval that straddles both margins). The other three words of that rule are conclusive
  and do not trigger the rule: 'equivalent' (the interval lies inside the band), 'non-inferior
  only' (the interval lies below +1%, so the policy is shown to be no worse than the margin)
  and 'inferior' (the interval lies above zero and reaches the margin). The Bonferroni and
  seed-level t verdicts of the same component (v1.0_sim, v1.0_t) are reported by p88 beside
  it and are not read here.

  triggered() reads the five-seed analysis of record, arm_registry.out(
  'results/family_inference.json') under PACT_ENV=exact PACT_SEEDS=301-305 (PACT_OUT, default
  out/exact on the command line), refuses a file that p88 did not write for the exact
  environment with seeds 301 to 305, and returns the combinations that trigger the rule.

THE RUNS OF THE SECOND FIVE SEEDS
  45 runs in the exact environment: the released recipe xmask-v1-s306 to -s310 and the
  specialist xmask-solo-<R>-s306 to -s310 of each of the eight combinations R. The rule names
  the policy and the specialists, so the single-rule control gets no run of a second seed.
  Names and paths come from the registry: the stem and the seed suffix of
  arm_registry.model() and arm_registry.suffix(), and each path helper of the registry
  (checkpoint, train_log, config_path, vali_log, result_path) applied to the run of the same
  arm at seed 301 with the run name exchanged, so no pattern is written twice. For a first
  seed every function here returns what the registry returns.
  Training: scripts/run_exactmask_queue.sh with SEED_TIERS="306 307 308 309 310".
  Evaluation: scripts/run_seed_extension_eval.sh, each job the job of the evaluation plan
  (scripts/eval_newseeds_plan.py, group principal, 10x25) for the same arm at seed 305 with
  the run name exchanged, into arm_registry.res_dir('10x25') of the exact environment.

  The second five seeds serve the principal comparison only (scripts/p110_seed_extension.py).
  arm_registry.PLANNED_SEEDS is unchanged, so every loader of the registry refuses a second
  seed, every other analysis keeps the first five seeds, and arm_registry.select_seed never
  sees a second seed. load() and registry_load() below are the readers of their arrays.

A RUN IS COMPLETE when it is trained and evaluated:
  trained    its checkpoint exists; its training log holds the completion line ('done in')
             and 32786 parameters on the line '[model] params total='; its configuration
             records first_mask exact and mask_ties exact and equals the configuration of
             the same arm at seed 301 except model_suffix and seed_train, which hold its own
             name and seed (no other field of the stored configurations derives from them);
             its validation record exists;
  evaluated  every array the principal comparison reads (the policy on the eight
             combinations, a specialist on its own) exists with one row per test instance of
             10x25.

Command line (environment exact; PACT_ENV, PACT_SEEDS and PACT_OUT default to exact,
301-305 and out/exact):
  python scripts/seed_extension.py list                the 45 runs and their files
  python scripts/seed_extension.py state [--root D]    trained and evaluated, per run
  python scripts/seed_extension.py check [--root D]    exit 0 only when all 45 are complete
  python scripts/seed_extension.py triggered [--record F]
                                     exit 0 when the rule applies (prints the combinations),
                                     1 when it does not, 2 when the record is missing or refused
  python scripts/seed_extension.py trained <name> [--root D]
                                     exit 0 when the run is trained (the queue's check)
  python scripts/seed_extension.py evaljobs            the evaluation jobs, one line each, in
                                     the format of 'eval_newseeds_plan.py jobs'
  python scripts/seed_extension.py rows <n> <file> ... exit 0 when every file exists and every
                                     .npy file holds n rows
--root resolves checkpoints, logs, configurations and validation records (default the
repository); the arrays are read from arm_registry.res_dir('10x25') (PACT_RESULTS).
"""
import argparse
import contextlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import arm_registry as REG  # noqa: E402

ENV = 'exact'
FIRST_SEEDS = tuple(REG.PLANNED_SEEDS[ENV])
SECOND_SEEDS = (306, 307, 308, 309, 310)
ALL_SEEDS = FIRST_SEEDS + SECOND_SEEDS
SEED_WORDS = {**REG.SEED_WORDS, 306: 'Six', 307: 'Seven', 308: 'Eight', 309: 'Nine', 310: 'Ten'}
ARMS = ('released', 'specialist')       # the rule names the policy and the specialists
FOLDER = '10x25'                        # the training size, the folder of the comparison
TEMPLATE_SEED = FIRST_SEEDS[0]          # names and paths: the run of seed 301, name exchanged
EVAL_TEMPLATE_SEED = FIRST_SEEDS[-1]    # evaluation jobs: the plan's job of seed 305
PARAMS = 32786
COMPLETION = 'done in'
MARGIN = 1.0
VERDICT_KEY = 'v1.0'                    # p88: verdict of the hierarchical interval at 1%
TRIGGER_WORDS = ('inconclusive',)
RECORD = 'results/family_inference.json'
GENERATOR = 'scripts/p88_family_inference.py'
CONFIG_OWN = ('model_suffix', 'seed_train')   # the configuration fields of name and seed
assert not set(FIRST_SEEDS) & set(SECOND_SEEDS) and len(SECOND_SEEDS) == 5
assert set(SEED_WORDS) >= set(ALL_SEEDS)


class Refused(Exception):
    """A record, a run or a setting that this module does not accept."""


# ----------------------------------------------------------------- setting
def require_exact():
    """The active registry environment must be exact (folders and names depend on it)."""
    if REG.env() != ENV:
        raise Refused(f'the second five seeds exist in the {ENV} environment only; the active '
                      f'environment is {REG.env()} (set PACT_ENV={ENV})')


def check_seed(seed):
    seed = int(seed)
    if seed not in ALL_SEEDS:
        raise Refused(f'seed {seed} is neither a first seed {FIRST_SEEDS} nor a second seed '
                      f'{SECOND_SEEDS}')
    return seed


def is_second(seed):
    return check_seed(seed) in SECOND_SEEDS


def _check_arm(arm, combo, seed):
    if arm not in ARMS and is_second(seed):
        raise Refused(f'arm {arm!r} has no run of a second seed; the extension rule names '
                      f'{ARMS}')


# ----------------------------------------------------------------- names and paths
def runs():
    """The 45 runs (arm, combo, seed) in the order of the training queue: the policy runs of
    the five seeds, then the eight specialists of each seed."""
    return ([('released', None, s) for s in SECOND_SEEDS]
            + [('specialist', r, s) for s in SECOND_SEEDS for r in REG.COMBOS])


def name(arm, combo, seed):
    """Run name, e.g. ('specialist', 'W', 306) -> 'xmask-solo-W-s306'."""
    _check_arm(arm, combo, seed)
    seed = check_seed(seed)
    if seed in FIRST_SEEDS:
        return REG.model(arm, combo, seed, ENV)
    t = REG.model(arm, combo, TEMPLATE_SEED, ENV)
    sfx = REG.suffix(arm, TEMPLATE_SEED, ENV)
    if not sfx or not t.endswith(sfx):
        raise Refused(f'cannot derive the stem of {t!r} from the registry suffix {sfx!r}')
    return t[:-len(sfx)] + REG.suffix(arm, seed, ENV)


def _exchanged(path_of, arm, combo, seed, *args):
    """path_of(arm, combo, seed, ...) of the registry; for a second seed the same helper on
    the run of seed 301 with the run name exchanged."""
    if not is_second(seed):
        return path_of(arm, combo, int(seed), *args)
    t = REG.model(arm, combo, TEMPLATE_SEED, ENV)
    p = path_of(arm, combo, TEMPLATE_SEED, *args)
    if p.count(t) != 1:
        raise Refused(f'cannot derive a path layout from {p!r}')
    return p.replace(t, name(arm, combo, seed))


def checkpoint(arm, combo, seed):
    return _exchanged(REG.checkpoint, arm, combo, seed, ENV)


def train_log(arm, combo, seed):
    return _exchanged(REG.train_log, arm, combo, seed, ENV)


def config_path(arm, combo, seed):
    return _exchanged(REG.config_path, arm, combo, seed, ENV)


def vali_log(arm, combo, seed):
    return _exchanged(REG.vali_log, arm, combo, seed, ENV)


def result_path(arm, combo, seed, regime):
    """Array of one run on one combination at 10x25 (arm_registry.result_path)."""
    require_exact()
    if not is_second(seed):
        return REG.result_path(arm, combo, int(seed), FOLDER, regime)
    t = REG.model(arm, combo, TEMPLATE_SEED, ENV)
    p = REG.result_path(arm, combo, TEMPLATE_SEED, FOLDER, regime)
    if p.count(t) != 1:
        raise Refused(f'cannot derive a path layout from {p!r}')
    return p.replace(t, name(arm, combo, seed))


def regimes_read(arm, combo):
    """The combinations the principal comparison reads: the policy all eight, a specialist
    its own."""
    return tuple(REG.COMBOS) if arm == 'released' else (combo,)


def paths(arm, combo, seed):
    return dict(checkpoint=checkpoint(arm, combo, seed), log=train_log(arm, combo, seed),
                config=config_path(arm, combo, seed), vali=vali_log(arm, combo, seed),
                arrays={r: result_path(arm, combo, seed, r) for r in regimes_read(arm, combo)})


def find(run_name):
    """(arm, combo, seed) of a run of the second five seeds, by name."""
    for a, c, s in runs():
        if name(a, c, s) == run_name:
            return a, c, s
    raise Refused(f'{run_name!r} is not a run of the second five seeds')


# ----------------------------------------------------------------- arrays
def load(arm, combo, seed, regime, n=None):
    """Makespans of one run on one combination at 10x25. For a first seed exactly
    arm_registry.load(arm, combo, seed, '10x25', regime, n); for a second seed the same
    reader (arm_registry.read_path: MissingResult when absent, makespan column, float)."""
    require_exact()
    _check_arm(arm, combo, seed)
    if not is_second(seed):
        return REG.load(arm, combo, int(seed), FOLDER, regime, n)
    return REG.read_path(result_path(arm, combo, seed, regime), n)


def registry_load(arm, combo, seed, folder, regime, n=None):
    """load() with the signature of arm_registry.load (and of the loader of p88). A second
    seed exists in the folder 10x25 only."""
    if not is_second(seed):
        return REG.load(arm, combo, int(seed), folder, regime, n)
    if folder != FOLDER:
        raise Refused(f'the second five seeds are evaluated in {FOLDER} only, not {folder}')
    return load(arm, combo, seed, regime, n)


def n_test():
    """Test instances of 10x25: the instances of the evaluation plan's 10x25 jobs."""
    import eval_newseeds_plan as PLAN
    return len(PLAN.instance_names(PLAN.FACTORY[FOLDER][0]))


# ----------------------------------------------------------------- state
_PARAMS = re.compile(r'\[model\] params total=([0-9]+)')


def _at(root, p):
    """p resolved against root ('.' leaves the repository-relative path)."""
    return p if root in ('.', '') else os.path.join(root, p)


def config_drift(arm, combo, seed, root='.'):
    """Differences of the run's configuration from the configuration of the same arm at seed
    301, other than its own name and seed (model_suffix, seed_train); [] when none."""
    mine, ref = (_at(root, config_path(arm, combo, s)) for s in (seed, TEMPLATE_SEED))
    for p in (mine, ref):
        if not os.path.exists(p):
            return [f'missing configuration {p}']
    a, b = json.load(open(mine)), json.load(open(ref))
    out = []
    want = {'model_suffix': name(arm, combo, seed), 'seed_train': int(seed)}
    for k, v in want.items():
        if a.get(k) != v:
            out.append(f'{k} is {a.get(k)!r}, want {v!r}')
    for k in sorted(set(a) | set(b)):
        if k not in CONFIG_OWN and a.get(k, '<absent>') != b.get(k, '<absent>'):
            out.append(f'{k}: {a.get(k, "<absent>")!r} against {b.get(k, "<absent>")!r} at seed '
                       f'{TEMPLATE_SEED}')
    return out


def training_problems(arm, combo, seed, root='.'):
    """[] when the run is trained (module docstring), otherwise the reasons."""
    p = {k: (_at(root, v) if k != 'arrays' else v) for k, v in paths(arm, combo, seed).items()}
    why = []
    if not os.path.exists(p['checkpoint']):
        why.append(f'no checkpoint {p["checkpoint"]}')
    if not os.path.exists(p['log']):
        why.append(f'no training log {p["log"]}')
    else:
        txt = open(p['log'], errors='replace').read()
        if COMPLETION not in txt:
            why.append(f'{p["log"]} has no completion line')
        m = _PARAMS.search(txt)
        if not m or int(m.group(1)) != PARAMS:
            why.append(f'{p["log"]} reports {m.group(1) if m else "no"} parameters, want {PARAMS}')
    if not os.path.exists(p['config']):
        why.append(f'no configuration {p["config"]}')
    else:
        cfg = json.load(open(p['config']))
        for k in ('first_mask', 'mask_ties'):
            if cfg.get(k) != 'exact':
                why.append(f'{p["config"]} records {k}={cfg.get(k)!r}, want exact')
        why += [f'configuration drift against seed {TEMPLATE_SEED}: {x}'
                for x in config_drift(arm, combo, seed, root)]
    if not os.path.exists(p['vali']):
        why.append(f'no validation record {p["vali"]}')
    return why


def evaluation_problems(arm, combo, seed, n=None):
    """[] when every array the comparison reads exists with n rows (n test instances)."""
    n = n_test() if n is None else n
    why = []
    for r, f in paths(arm, combo, seed)['arrays'].items():
        if not os.path.exists(f):
            why.append(f'missing {f}')
            continue
        rows = np.load(f).shape[0]
        if rows != n:
            why.append(f'{f}: {rows} rows, the test set has {n}')
    return why


def state(root='.', n=None):
    """Per run of the second five seeds: name, trained, evaluated and the reasons."""
    require_exact()
    n = n_test() if n is None else n
    out = []
    for a, c, s in runs():
        tp, ep = training_problems(a, c, s, root), evaluation_problems(a, c, s, n)
        out.append(dict(name=name(a, c, s), arm=a, combo=c, seed=s, trained=not tp,
                        evaluated=not ep, complete=not tp and not ep, training_problems=tp,
                        evaluation_problems=ep))
    return out


def complete(root='.', n=None):
    return all(x['complete'] for x in state(root, n))


# ----------------------------------------------------------------- the extension rule
def record_path():
    """The five-seed analysis of record: arm_registry.out('results/family_inference.json')
    under the exact environment with seeds 301 to 305."""
    require_exact()
    if REG.seeds() != FIRST_SEEDS:
        raise Refused(f'the record of the extension rule is the analysis with seeds '
                      f'{FIRST_SEEDS}; the active seeds are {REG.seeds()} (set PACT_SEEDS='
                      f'{FIRST_SEEDS[0]}-{FIRST_SEEDS[-1]})')
    try:
        return REG.out(RECORD)
    except SystemExit as e:
        raise Refused(str(e))


def read_record(path=None):
    """The five-seed record of p88, checked: written by p88 for the exact environment with
    the seeds 301 to 305, with a section A component and its 1% verdict for every
    combination. Refused otherwise (a missing file too)."""
    path = path or record_path()
    if not os.path.exists(path):
        raise Refused(f'the five-seed record {path} does not exist')
    try:
        rec = json.load(open(path))
    except (OSError, ValueError) as e:
        raise Refused(f'{path} is not readable JSON: {e}')
    if rec.get('generator') != GENERATOR:
        raise Refused(f'{path} was written by {rec.get("generator")!r}, not {GENERATOR}')
    setting = str(rec.get('setting', ''))
    if not setting.startswith(f'environment {ENV},'):
        raise Refused(f'{path} was written for {setting!r}, not the {ENV} environment')
    if [int(x) for x in rec.get('seeds') or []] != list(FIRST_SEEDS):
        raise Refused(f'{path} was written with seeds {rec.get("seeds")}, not {list(FIRST_SEEDS)}')
    comps = (rec.get('parity') or {}).get('components') or {}
    import stats_family as SF
    for r in REG.COMBOS:
        v = (comps.get(r) or {}).get(VERDICT_KEY)
        if v not in SF.VERDICTS:
            raise Refused(f'{path} holds no verdict {VERDICT_KEY} of section A for {r} '
                          f'(got {v!r})')
    return rec


def inconclusive(rec):
    """Combinations whose section A verdict at the 1% margin is inconclusive."""
    comps = rec['parity']['components']
    return [r for r in REG.COMBOS if comps[r][VERDICT_KEY] in TRIGGER_WORDS]


def triggered(path=None):
    """The combinations that trigger the extension rule in the five-seed record (empty when
    the rule does not apply). Refused (exception Refused) for a missing or foreign record."""
    return inconclusive(read_record(path))


# ----------------------------------------------------------------- evaluation jobs
@contextlib.contextmanager
def _plan_groups(plan, groups, seeds):
    """The evaluation plan with only these groups and exact seeds, restored afterwards."""
    old = plan.GROUPS, plan.EXACT_SEEDS
    plan.GROUPS, plan.EXACT_SEEDS = list(groups), list(seeds)
    try:
        yield plan
    finally:
        plan.GROUPS, plan.EXACT_SEEDS = old


def plan_jobs(seed=EVAL_TEMPLATE_SEED):
    """{(arm, combo): job} of the evaluation plan's principal group at 10x25 for a first seed,
    the policy and the eight specialists (the control is left out)."""
    import eval_newseeds_plan as PLAN
    seed = int(seed)
    if seed not in FIRST_SEEDS:
        raise Refused(f'the evaluation plan has jobs of the first seeds only, not {seed}')
    with _plan_groups(PLAN, ['principal'], [seed]):
        js = [j for j in PLAN.exact_jobs() if j['group'] == 'principal' and j['campaign'] == FOLDER]
    by_model = {j['model']: j for j in js}
    out = {}
    for arm in ARMS:
        for c in (REG.COMBOS if arm == 'specialist' else (None,)):
            m = REG.PREFIX + name(arm, c, seed)
            if m not in by_model:
                raise Refused(f'the evaluation plan has no 10x25 job of {m}')
            j = dict(by_model[m])
            j['pre'] = PLAN.preflight_of(j)
            out[(arm, c)] = j
    return out


def renamed_job(job, arm, combo, seed):
    """The job with the run name exchanged and nothing else: id, model, checkpoint, training
    log and file names."""
    old = job['model'][len(REG.PREFIX):]
    new = name(arm, combo, seed)
    om, nm = REG.PREFIX + old, REG.PREFIX + new
    if job['model'] != om or not job['id'].endswith(':' + old):
        raise Refused(f'unexpected layout of the job {job["id"]}')
    files = [f.replace(om, nm) for f in job['files']]
    if not all(nm in f for f in files) or len(set(files)) != len(files):
        raise Refused(f'cannot rename the files of the job {job["id"]}')
    ck, lg = checkpoint(arm, combo, seed), train_log(arm, combo, seed)
    if job['ckpt'].replace(old, new) != ck or job['log'].replace(old, new) != lg:
        raise Refused(f'the checkpoint or log of the job {job["id"]} does not follow the registry')
    return dict(job, id=job['id'][:-len(old)] + new, model=nm, ckpt=ck, log=lg, files=files)


def eval_jobs():
    """One job per run of the second five seeds, in the order of runs(): the plan's job of the
    same arm at seed 305 with the run name exchanged. The arrays the comparison reads must be
    among the job's files at the registry's paths."""
    require_exact()
    tmpl = plan_jobs(EVAL_TEMPLATE_SEED)
    out = []
    for a, c, s in runs():
        j = renamed_job(tmpl[(a, c)], a, c, s)
        for r in regimes_read(a, c):
            p = os.path.normpath(result_path(a, c, s, r))
            if p not in {os.path.normpath(f) for f in j['files']}:
                raise Refused(f'the job {j["id"]} does not write {p} (evaluation plan RESULTS '
                              f'and arm_registry PACT_RESULTS must name one results root)')
        out.append(j)
    return out


def job_line(job):
    """A job as one line of 'eval_newseeds_plan.py jobs'."""
    import eval_newseeds_plan as PLAN
    return '|'.join(str(job[k]) for k in PLAN.KEYS_JOB) + '|' + ' '.join(job['files'])


# ----------------------------------------------------------------- command line
def _rows(n, files):
    bad = []
    for f in files:
        if not os.path.exists(f):
            bad.append(f'missing {f}')
        elif f.endswith('.npy') and np.load(f).shape[0] != n:
            bad.append(f'{f}: {np.load(f).shape[0]} rows, want {n}')
    return bad


def main(argv=None):
    for k, v in (('PACT_ENV', ENV), ('PACT_SEEDS', f'{FIRST_SEEDS[0]}-{FIRST_SEEDS[-1]}'),
                 ('PACT_OUT', 'out/exact')):
        os.environ.setdefault(k, v)
    ap = argparse.ArgumentParser(description='the second five training seeds')
    ap.add_argument('cmd', choices=('list', 'state', 'check', 'triggered', 'trained',
                                    'evaljobs', 'rows'))
    ap.add_argument('args', nargs='*')
    ap.add_argument('--root', default='.')
    ap.add_argument('--record', default=None)
    a = ap.parse_args(argv)
    try:
        require_exact()
        if a.cmd == 'triggered':
            try:
                hit = triggered(a.record)
            except Refused as e:
                print(f'extension rule: record refused: {e}')
                return 2
            src = a.record or record_path()
            if hit:
                print(f'the extension rule applies: inconclusive at the {MARGIN:g}% margin with '
                      f'seeds {FIRST_SEEDS[0]} to {FIRST_SEEDS[-1]}: {", ".join(hit)} ({src})')
                return 0
            print(f'the extension rule does not apply: no combination is inconclusive at the '
                  f'{MARGIN:g}% margin with seeds {FIRST_SEEDS[0]} to {FIRST_SEEDS[-1]} ({src})')
            return 1
        if a.cmd == 'list':
            for x, c, s in runs():
                p = paths(x, c, s)
                print(f'{name(x, c, s)}  {p["checkpoint"]}  {p["log"]}  {p["config"]}  '
                      f'{p["vali"]}  ' + ' '.join(p['arrays'].values()))
            return 0
        if a.cmd in ('state', 'check'):
            st = state(a.root)
            tr, ev, co = (sum(x[k] for x in st) for k in ('trained', 'evaluated', 'complete'))
            print(f'second five seeds: {len(st)} runs, {tr} trained, {ev} evaluated, {co} complete')
            shown = st if a.cmd == 'state' else [x for x in st if not x['complete']][:6]
            for x in shown:
                why = x['training_problems'] + x['evaluation_problems']
                print(f'  {x["name"]:22s} trained {"yes" if x["trained"] else "no "}  evaluated '
                      f'{"yes" if x["evaluated"] else "no "}' + (f'  ({why[0]}'
                      + (f'; and {len(why) - 1} more' if len(why) > 1 else '') + ')' if why else ''))
            if a.cmd == 'check' and co < len(st) - len(shown):
                print(f'  ... and {len(st) - co - len(shown)} more incomplete runs')
            return 0 if co == len(st) else 1
        if a.cmd == 'trained':
            if len(a.args) != 1:
                ap.error('trained takes one run name')
            why = training_problems(*find(a.args[0]), root=a.root)
            for x in why:
                print(f'{a.args[0]}: {x}')
            if not why:
                print(f'{a.args[0]}: trained')
            return 0 if not why else 1
        if a.cmd == 'evaljobs':
            for j in eval_jobs():
                print(job_line(j))
            return 0
        if a.cmd == 'rows':
            if len(a.args) < 2:
                ap.error('rows takes a row count and at least one file')
            bad = _rows(int(a.args[0]), a.args[1:])
            for x in bad:
                print(x)
            return 0 if not bad else 1
    except Refused as e:
        print(f'REFUSING: {e}')
        return 2
    return 2


if __name__ == '__main__':
    sys.exit(main())
