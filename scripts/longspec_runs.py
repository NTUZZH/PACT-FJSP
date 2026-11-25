r"""Names, paths and checks of the specialists trained for the policy's number of updates.

THE RUNS
  xlong-solo-<R>-s<seed>: the recipe of the specialist xmask-solo-<R>-s<seed> (one combination R,
  exact mask, 32786 parameters, scripts/run_exactmask_queue.sh) with --max_updates 2000 in place
  of 1000, the number of updates of the policy. Eight combinations R (N, L, S, W, LS, LW, SW,
  LSW), training seeds 301 to 305, the seeds of the 1000-update specialists.
    training    scripts/run_longspec_queue.sh
    evaluation  scripts/run_longspec_eval.sh: one greedy pass on the test instances of R at
                10x25 with the call of the evaluation plan's principal job of
                xmask-solo-<R>-s<seed> (scripts/eval_newseeds_plan.py, read through
                seed_extension.plan_jobs), the run name exchanged and the combinations
                restricted to R; the arrays go to <results>/10x25_xlong, never to 10x25_xmask
    analysis    scripts/p117_long_specialists.py
  Checkpoint, training log, configuration and validation record follow the registry's helpers
  for xmask-solo-<R>-s<seed> with the run name exchanged (arm_registry.checkpoint, train_log,
  config_path, vali_log), so no path pattern is written twice.

A RUN IS TRAINED when
  its checkpoint exists; its training log holds the header line of train_family.py with its own
  model name, the combination R and 2000 updates, the line of the exact mask, 32786 parameters
  on the line '[model] params total=' and the completion line; its configuration records
  first_mask exact and mask_ties exact and equals the configuration of xmask-solo-<R>-s<seed>
  except model_suffix (its own name) and max_updates (2000 against 1000); its validation record
  exists.
A RUN IS EVALUATED when its array <results>/10x25_xlong/Result_greedy+10x25+family+<name>_<R>.npy
  holds one row per test instance of R at 10x25 (the descriptors of data/FAMILY/10x25/desc/<R>).

THE PROBE (scripts/run_longspec_queue.sh, before any training)
  One update on the CPU with seed 301 per combination R. Checked here: the probe's
  configuration records the exact mask, trains R, and equals the stored configuration of
  xmask-solo-<R>-s301 except model_suffix, seed_train, device and max_updates; its log holds
  32786 parameters, the mask line, and the MWKR norms line of the log of xmask-solo-<R>-s301
  (same validation instances and scale).

Command line (the environment is exact; PACT_ENV defaults to exact here):
  python scripts/longspec_runs.py list [--seeds S]       the runs and their files
  python scripts/longspec_runs.py state [--seeds S] [--root D]
                                       trained and evaluated, per run; exit 0 when all are
  python scripts/longspec_runs.py probe <R> <suffix> <log> [--root D]
                                       exit 0 when the probe of R passes (the queue's check)
  python scripts/longspec_runs.py trained <name> [--root D]
                                       exit 0 when the run is trained (the queue's and the
                                       evaluation's check)
  python scripts/longspec_runs.py evaljob <name>          the evaluation job of the run, one
                                       line in the format of 'eval_newseeds_plan.py jobs'
  python scripts/longspec_runs.py verifyjob               the plan's job of xmask-solo-N-s301
                                       restricted to N, the same format
  python scripts/longspec_runs.py rows <n> <file> ...     exit 0 when every file exists and
                                       every .npy file holds n rows
  python scripts/longspec_runs.py compare <a.npy> <b.npy> largest difference of the makespan
                                       columns; exit 0 when the row counts agree and it is at
                                       most 1e-6
--seeds takes '301', '301,302' or '301-305' (default 301). --root resolves checkpoints, logs,
configurations and validation records (default the repository); arrays are read from
arm_registry.results_root() (PACT_RESULTS, default test_results/FAMILY).

THE VARIANT OF 250 UPDATES (scripts/budgetspec_runs.py) loads this file as a module of its own
and exchanges TAG, STEM, UPDATES, RESULT_FOLDER, SEEDS and DEFAULT_SEEDS; every name, path, check
and log line below reads these constants when it is called. The variant takes the seeds 301 to
310. For a seed that the registry does not plan (306 to 310, the second five seeds of the
principal comparison) every name and path of the 1000-update specialist and of the policy comes
from scripts/seed_extension.py (name, checkpoint, train_log, config_path, vali_log,
result_path), which gives the registry's answer for a first seed, and the evaluation job is the
plan's job of xmask-solo-<R>-s305 (seed_extension.EVAL_TEMPLATE_SEED) with the run name
exchanged (seed_extension.renamed_job), then exchanged again to the run of this file. The
module is imported only when such a seed is named, so the runs of seeds 301 to 305 read the
registry alone.
"""
import argparse
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import numpy as np  # noqa: E402

import arm_registry as REG  # noqa: E402

ENV = 'exact'
COMBOS = tuple(REG.COMBOS)
SEEDS = tuple(REG.PLANNED_SEEDS[ENV])          # 301 to 305, the seeds of the specialists
DEFAULT_SEEDS = (301,)
TAG = 'xlong'                                   # the word of the log lines and the job ids
STEM = 'xlong-solo-{R}'
UPDATES = 2000                                  # the policy's number of updates
STANDARD_UPDATES = 1000                         # the specialist's
PARAMS = 32786
FOLDER = '10x25'                                # training size, folder of the comparison
RESULT_FOLDER = '10x25_xlong'                   # the arrays of the runs, under the results root
PROBE_SEED = 301
PROBE_ALLOWED = ('model_suffix', 'seed_train', 'device', 'max_updates')
RUN_ALLOWED = ('model_suffix', 'max_updates')
VERIFY = ('N', 301)                             # the run whose stored array the runner reproduces
TOL = 1e-6
COMPLETION = 'done in'
MASK_LINE = '[train_family] MASK: first_mask=exact mask_ties=exact'

_PARAMS = re.compile(r'\[model\] params total=([0-9]+)')
_NORMS = re.compile(r'^\[vali\] MWKR norms: ([^\n]*)$', re.M)
_HEAD = re.compile(r'^\[train_family\] model=(\S+) regimes=\[([^\]]*)\] updates=(\d+)\s*$', re.M)


class Refused(Exception):
    """A run, a setting or a file that this module does not accept."""


# ----------------------------------------------------------------- setting and names
def require_exact():
    if REG.env() != ENV:
        raise Refused(f'the runs exist in the {ENV} environment only; the active environment is '
                      f'{REG.env()} (set PACT_ENV={ENV})')


def check_seed(seed):
    seed = int(seed)
    if seed not in SEEDS:
        raise Refused(f'seed {seed}: the runs take the seeds of the specialists, {SEEDS[0]} to '
                      f'{SEEDS[-1]}')
    return seed


def check_combo(combo):
    if combo not in COMBOS:
        raise Refused(f'combination {combo!r}: want one of {COMBOS}')
    return combo


def parse_seeds(text):
    """'301', '301,302', '301 302' or '301-305' -> (301, ...); each seed once, in order."""
    out = []
    for part in str(text).replace(',', ' ').split():
        if re.fullmatch(r'\d+-\d+', part):
            a, b = (int(x) for x in part.split('-'))
            out.extend(range(a, b + 1))
        elif re.fullmatch(r'\d+', part):
            out.append(int(part))
        else:
            raise Refused(f'seed list {text!r}: {part!r} is not a seed or a range')
    if not out or len(set(out)) != len(out):
        raise Refused(f'seed list {text!r} must be non-empty and name each seed once')
    return tuple(check_seed(s) for s in out)


def name(combo, seed):
    """Run name, e.g. ('W', 301) -> 'xlong-solo-W-s301'."""
    return f'{STEM.format(R=check_combo(combo))}-s{check_seed(seed)}'


def _second(seed):
    """A seed that the registry does not plan: one of the second five seeds (306 to 310)."""
    return int(seed) not in REG.PLANNED_SEEDS[ENV]


def _extension():
    import seed_extension as SE
    return SE


def _run_of(helper, arm, combo, seed):
    """The registry's helper ('model', 'checkpoint', 'train_log', 'config_path', 'vali_log')
    for the run (arm, combo, seed); for a second seed the helper of scripts/seed_extension.py
    of the same name ('name' for 'model')."""
    if not _second(seed):
        return getattr(REG, helper)(arm, combo, seed, ENV)
    SE = _extension()
    try:
        return getattr(SE, 'name' if helper == 'model' else helper)(arm, combo, seed)
    except SE.Refused as e:
        raise Refused(str(e))


def _array_of(arm, combo, seed, regime):
    """arm_registry.result_path at 10x25; for a second seed seed_extension.result_path."""
    if not _second(seed):
        return REG.result_path(arm, combo, seed, FOLDER, regime)
    SE = _extension()
    try:
        return SE.result_path(arm, combo, seed, regime)
    except SE.Refused as e:
        raise Refused(str(e))


def standard_name(combo, seed):
    """The 1000-update specialist of the same combination and seed (arm_registry)."""
    return _run_of('model', 'specialist', check_combo(combo), check_seed(seed))


def _name_rx():
    """The run names of STEM: STEM with R one of the combinations, then -s<seed>."""
    head, tail = STEM.split('{R}')
    return re.compile('^' + re.escape(head) + '(' + '|'.join(COMBOS) + ')' + re.escape(tail)
                      + r'-s(\d+)$')


def find(run_name):
    """(combination, seed) of a run, by name."""
    m = _name_rx().match(str(run_name))
    if not m:
        raise Refused(f'{run_name!r} is not a run {STEM.format(R="<R>")}-s<seed>')
    return m.group(1), check_seed(m.group(2))


def runs(seeds=None):
    """(combination, seed) of every run of the seeds (default DEFAULT_SEEDS), per seed the
    eight combinations."""
    return [(r, check_seed(s)) for s in (DEFAULT_SEEDS if seeds is None else seeds) for r in COMBOS]


def _exchanged(helper, combo, seed):
    """The path helper of the 1000-update specialist (arm_registry, or seed_extension for a
    second seed) with the run name exchanged."""
    t = standard_name(combo, seed)
    p = _run_of(helper, 'specialist', combo, seed)
    if p.count(t) != 1:
        raise Refused(f'cannot derive a path layout from {p!r}')
    return p.replace(t, name(combo, seed))


def checkpoint(combo, seed):
    return _exchanged('checkpoint', combo, seed)


def train_log(combo, seed):
    return _exchanged('train_log', combo, seed)


def config_path(combo, seed):
    return _exchanged('config_path', combo, seed)


def vali_log(combo, seed):
    return _exchanged('vali_log', combo, seed)


def result_dir():
    return f'{REG.results_root()}/{RESULT_FOLDER}'


def result_path(combo, seed):
    """The array of the run on its own combination at 10x25."""
    require_exact()
    p = os.path.basename(_array_of('specialist', combo, seed, combo))
    t = standard_name(combo, seed)
    if p.count(t) != 1:
        raise Refused(f'cannot derive an array name from {p!r}')
    return f'{result_dir()}/{p.replace(t, name(combo, seed))}'


def standard_result(combo, seed):
    """The stored array of xmask-solo-<R>-s<seed> on R at 10x25."""
    require_exact()
    return _array_of('specialist', combo, check_seed(seed), combo)


def policy_result(combo, seed):
    """The stored array of the policy xmask-v1-s<seed> on R at 10x25."""
    require_exact()
    return _array_of('released', None, check_seed(seed), check_combo(combo))


def paths(combo, seed):
    return dict(checkpoint=checkpoint(combo, seed), log=train_log(combo, seed),
                config=config_path(combo, seed), vali=vali_log(combo, seed),
                array=result_path(combo, seed))


def standard_paths(combo, seed):
    return {k: _run_of(h, 'specialist', combo, seed) for k, h in
            (('checkpoint', 'checkpoint'), ('log', 'train_log'), ('config', 'config_path'),
             ('vali', 'vali_log'))}


def n_test(combo):
    """Test instances of the combination at 10x25: the descriptors eval_family.py reads."""
    n = len(glob.glob(os.path.join(ROOT, 'data', 'FAMILY', FOLDER, 'desc', check_combo(combo),
                                   '*.desc.json')))
    if not n:
        raise Refused(f'no test descriptor in data/FAMILY/{FOLDER}/desc/{combo}')
    return n


def _at(root, p):
    return p if root in ('.', '') else os.path.join(root, p)


# ----------------------------------------------------------------- checks
def config_differences(mine, ref):
    """Keys whose values differ between two configuration dicts (an absent key counts)."""
    return sorted(k for k in set(mine) | set(ref) if mine.get(k, '<absent>') != ref.get(k, '<absent>'))


def norms_line(path):
    """The text after 'MWKR norms: ' on the first norms line of a training log, or None."""
    if not os.path.exists(path):
        return None
    m = _NORMS.search(open(path, errors='replace').read())
    return None if m is None else m.group(1).strip()


def probe_problems(combo, suffix, probe_log, root='.'):
    """([] or the reasons, the differences found, the norms line) of the probe of a combination."""
    check_combo(combo)
    ref_cfg = _at(root, REG.config_path('specialist', combo, PROBE_SEED, ENV))
    ref_log = _at(root, REG.train_log('specialist', combo, PROBE_SEED, ENV))
    cfg_p = _at(root, f'train_log/FAMILY/config_{REG.PREFIX}{suffix}.json')
    why, diffs = [], []
    for p in (ref_cfg, ref_log, cfg_p, probe_log):
        if not os.path.exists(p):
            why.append(f'missing {p}')
    if why:
        return why, diffs, None
    txt = open(probe_log, errors='replace').read()
    m = _PARAMS.search(txt)
    if not m or int(m.group(1)) != PARAMS:
        why.append(f'{probe_log} reports {m.group(1) if m else "no"} parameters, want {PARAMS}')
    if MASK_LINE not in txt.splitlines():
        why.append(f'{probe_log} has no line "{MASK_LINE}"')
    probe, ref = json.load(open(cfg_p)), json.load(open(ref_cfg))
    for k in ('first_mask', 'mask_ties'):
        if probe.get(k) != 'exact':
            why.append(f'the probe configuration records {k}={probe.get(k)!r}, want exact')
    if probe.get('family_regimes') != combo:
        why.append(f'the probe trains {probe.get("family_regimes")!r}, want {combo!r}')
    if probe.get('model_suffix') != suffix:
        why.append(f'the probe configuration names {probe.get("model_suffix")!r}, want {suffix!r}')
    if int(ref.get('max_updates', -1)) != STANDARD_UPDATES or int(ref.get('seed_train', -1)) != PROBE_SEED:
        why.append(f'{ref_cfg} records max_updates={ref.get("max_updates")} and '
                   f'seed_train={ref.get("seed_train")}, want {STANDARD_UPDATES} and {PROBE_SEED}')
    diffs = config_differences(probe, ref)
    bad = [k for k in diffs if k not in PROBE_ALLOWED]
    if bad:
        why.append('configuration drift against ' + standard_name(combo, PROBE_SEED) + ': '
                   + '; '.join(f'{k}: {ref.get(k, "<absent>")!r} stored, {probe.get(k, "<absent>")!r} '
                               'in the probe' for k in bad))
    want, got = norms_line(ref_log), norms_line(probe_log)
    if want is None or got is None or want != got:
        why.append(f'MWKR norms line: {want!r} in {ref_log}, {got!r} in the probe')
    return why, diffs, got


def log_problems(log_path, run_name, combo, updates):
    """[] or the reasons why a training log is not the finished log of the run."""
    if not os.path.exists(log_path):
        return [f'no training log {log_path}']
    txt = open(log_path, errors='replace').read()
    why = []
    head = _HEAD.findall(txt)
    if len(head) != 1:
        why.append(f'{log_path} holds {len(head)} header lines of train_family.py, want one')
    else:
        model, regimes, upd = head[0]
        regs = tuple(x.strip().strip("'\"") for x in regimes.split(',') if x.strip())
        if model != REG.PREFIX + run_name:
            why.append(f'{log_path} is the log of {model}')
        if regs != (combo,):
            why.append(f'{log_path} trains {regs}, want ({combo!r},)')
        if int(upd) != updates:
            why.append(f'{log_path} declares {upd} updates, want {updates}')
    if MASK_LINE not in txt.splitlines():
        why.append(f'{log_path} has no line "{MASK_LINE}"')
    m = _PARAMS.search(txt)
    if not m or int(m.group(1)) != PARAMS:
        why.append(f'{log_path} reports {m.group(1) if m else "no"} parameters, want {PARAMS}')
    if COMPLETION not in txt:
        why.append(f'{log_path} has no completion line')
    return why


def training_problems(combo, seed, root='.'):
    """[] when the run is trained (module docstring), otherwise the reasons."""
    nm = name(combo, seed)
    p = {k: _at(root, v) for k, v in paths(combo, seed).items() if k != 'array'}
    ref_cfg = _at(root, _run_of('config_path', 'specialist', combo, seed))
    why = []
    if not os.path.exists(p['checkpoint']):
        why.append(f'no checkpoint {p["checkpoint"]}')
    why += log_problems(p['log'], nm, combo, UPDATES)
    if not os.path.exists(p['config']):
        why.append(f'no configuration {p["config"]}')
    elif not os.path.exists(ref_cfg):
        why.append(f'no configuration {ref_cfg} of {standard_name(combo, seed)}')
    else:
        a, b = json.load(open(p['config'])), json.load(open(ref_cfg))
        for k in ('first_mask', 'mask_ties'):
            if a.get(k) != 'exact':
                why.append(f'{p["config"]} records {k}={a.get(k)!r}, want exact')
        want = {'model_suffix': (nm, standard_name(combo, seed)),
                'max_updates': (UPDATES, STANDARD_UPDATES)}
        for k, (mine, ref) in want.items():
            if a.get(k) != mine or b.get(k) != ref:
                why.append(f'{k}: {a.get(k)!r} against {b.get(k)!r} in '
                           f'{standard_name(combo, seed)}, want {mine!r} against {ref!r}')
        for k in config_differences(a, b):
            if k not in RUN_ALLOWED:
                why.append(f'configuration drift against {standard_name(combo, seed)}: {k}: '
                           f'{a.get(k, "<absent>")!r} against {b.get(k, "<absent>")!r}')
    if not os.path.exists(p['vali']):
        why.append(f'no validation record {p["vali"]}')
    return why


def rows_problems(n, files):
    bad = []
    for f in files:
        if not os.path.exists(f):
            bad.append(f'missing {f}')
        elif f.endswith('.npy'):
            k = np.load(f).shape[0]
            if k != n:
                bad.append(f'{f}: {k} rows, want {n}')
    return bad


def evaluation_problems(combo, seed):
    """[] when the run's array exists with one row per test instance of its combination."""
    return rows_problems(n_test(combo), [result_path(combo, seed)])


def state(seeds=None, root='.'):
    """Per run of the seeds (default DEFAULT_SEEDS): name, trained, evaluated and the reasons."""
    require_exact()
    out = []
    for r, s in runs(seeds):
        tp, ep = training_problems(r, s, root), evaluation_problems(r, s)
        out.append(dict(name=name(r, s), combo=r, seed=s, trained=not tp, evaluated=not ep,
                        complete=not tp and not ep, training_problems=tp,
                        evaluation_problems=ep))
    return out


def complete_seeds(root='.'):
    """The seeds whose eight runs are trained and evaluated."""
    st = state(SEEDS, root)
    return tuple(s for s in SEEDS if all(x['complete'] for x in st if x['seed'] == s))


# ----------------------------------------------------------------- arrays
def makespans(path):
    """Makespan column of an array (arm_registry.read_path)."""
    return REG.read_path(path)


def compare(a_path, b_path):
    """(largest absolute difference of the makespan columns, rows of a, rows of b)."""
    a, b = makespans(a_path), makespans(b_path)
    if len(a) != len(b):
        return float('inf'), len(a), len(b)
    return (float(np.max(np.abs(a - b))) if len(a) else 0.0), len(a), len(b)


# ----------------------------------------------------------------- evaluation jobs
def plan_job(combo, seed):
    """The evaluation plan's principal 10x25 job of xmask-solo-<R>-s<seed>; for a second seed
    the plan's job of xmask-solo-<R>-s305 with the run name exchanged (seed_extension.renamed_job,
    as seed_extension.eval_jobs builds the jobs of the second five seeds)."""
    require_exact()
    seed, combo = check_seed(seed), check_combo(combo)
    SE = _extension()
    try:
        if not _second(seed):
            return SE.plan_jobs(seed)[('specialist', combo)]
        tmpl = SE.plan_jobs(SE.EVAL_TEMPLATE_SEED)[('specialist', combo)]
        return SE.renamed_job(tmpl, 'specialist', combo, seed)
    except SE.Refused as e:
        raise Refused(str(e))


def _restricted(job, combo):
    """The job on the one combination R; refuses a job that does not evaluate R greedily on the
    CPU under the exact mask without schedule export."""
    if combo not in job['regimes'].split():
        raise Refused(f'the job {job["id"]} does not evaluate {combo}')
    want = dict(env=ENV, device='cpu', methods='greedy', sched='-', campaign=FOLDER)
    bad = {k: job.get(k) for k, v in want.items() if job.get(k) != v}
    if bad:
        raise Refused(f'the job {job["id"]} is not a greedy CPU job of the exact environment at '
                      f'{FOLDER} without schedule export: {bad}')
    return dict(job, regimes=combo)


def eval_job(combo, seed):
    """The plan's job of xmask-solo-<R>-s<seed> with the run name exchanged and the combinations
    restricted to R; files: the run's array and summary in result_dir()."""
    j = _restricted(plan_job(combo, seed), combo)
    old, new = j['model'], REG.PREFIX + name(combo, seed)
    if old != REG.PREFIX + standard_name(combo, seed) or not j['id'].endswith(':' + old[len(REG.PREFIX):]):
        raise Refused(f'unexpected layout of the job {j["id"]}')
    summary = f'summary_{new}{j["tag"]}.md'
    files = [result_path(combo, seed), f'{result_dir()}/{summary}']
    if os.path.basename(files[0]) != f'Result_greedy+{new}_{combo}.npy':
        raise Refused(f'the array name {files[0]} does not follow eval_family.py')
    return dict(j, id=f'{TAG}:{FOLDER}:{name(combo, seed)}', model=new, ckpt=checkpoint(combo, seed),
                log=train_log(combo, seed), files=files, cost=None)


def verify_job():
    """The plan's job of xmask-solo-N-s301 restricted to N; file: its stored array."""
    combo, seed = VERIFY
    j = _restricted(plan_job(combo, seed), combo)
    return dict(j, files=[standard_result(combo, seed)], cost=None)


def job_line(job):
    import eval_newseeds_plan as PLAN
    return '|'.join(str(job[k]) for k in PLAN.KEYS_JOB) + '|' + ' '.join(job['files'])


# ----------------------------------------------------------------- command line
def main(argv=None):
    os.environ.setdefault('PACT_ENV', ENV)
    ap = argparse.ArgumentParser(description=f'the specialists trained for {UPDATES} updates')
    ap.add_argument('cmd', choices=('list', 'state', 'probe', 'trained', 'evaljob', 'verifyjob',
                                    'rows', 'compare'))
    ap.add_argument('args', nargs='*')
    ap.add_argument('--seeds', default=None)
    ap.add_argument('--root', default='.')
    a = ap.parse_args(argv)
    try:
        require_exact()
        seeds = parse_seeds(a.seeds) if a.seeds else DEFAULT_SEEDS
        if a.cmd == 'list':
            for r, s in runs(seeds):
                print(f'{name(r, s)}  ' + '  '.join(paths(r, s).values()))
            return 0
        if a.cmd == 'state':
            st = state(seeds, a.root)
            tr, ev = sum(x['trained'] for x in st), sum(x['evaluated'] for x in st)
            print(f'specialists trained for {UPDATES} updates, seeds {",".join(map(str, seeds))}: '
                  f'{len(st)} runs, {tr} trained, {ev} evaluated')
            for x in st:
                why = x['training_problems'] + x['evaluation_problems']
                print(f'  {x["name"]:20s} trained {"yes" if x["trained"] else "no "}  evaluated '
                      f'{"yes" if x["evaluated"] else "no "}'
                      + (f'  ({why[0]}' + (f'; and {len(why) - 1} more' if len(why) > 1 else '')
                         + ')' if why else ''))
            return 0 if all(x['complete'] for x in st) else 1
        if a.cmd == 'probe':
            if len(a.args) != 3:
                ap.error('probe <combination> <suffix> <probe log>')
            why, diffs, norms = probe_problems(*a.args, root=a.root)
            ref = standard_name(a.args[0], PROBE_SEED)
            if why:
                print(f'ABORT: the probe of {a.args[0]}: ' + '; '.join(why))
                return 1
            print(f'[{TAG}] {a.args[0]}: configuration against {ref} OK, differs only in '
                  f'{diffs}; MWKR norms equal ({norms}); {PARAMS} parameters, exact mask')
            return 0
        if a.cmd == 'trained':
            if len(a.args) != 1:
                ap.error('trained <name>')
            r, s = find(a.args[0])
            why = training_problems(r, s, a.root)
            if why:
                print(f'{a.args[0]} is not trained as required: ' + '; '.join(why))
                return 1
            print(f'{a.args[0]}: trained ({UPDATES} updates, {PARAMS} parameters, exact mask, '
                  f'configuration of {standard_name(r, s)} except model_suffix and max_updates)')
            return 0
        if a.cmd == 'evaljob':
            if len(a.args) != 1:
                ap.error('evaljob <name>')
            print(job_line(eval_job(*find(a.args[0]))))
            return 0
        if a.cmd == 'verifyjob':
            print(job_line(verify_job()))
            return 0
        if a.cmd == 'rows':
            if len(a.args) < 2:
                ap.error('rows <n> <file> ...')
            bad = rows_problems(int(a.args[0]), a.args[1:])
            print('; '.join(bad) if bad else f'{len(a.args) - 1} files, {a.args[0]} rows each')
            return 1 if bad else 0
        if a.cmd == 'compare':
            if len(a.args) != 2:
                ap.error('compare <a.npy> <b.npy>')
            for p in a.args:
                if not os.path.exists(p):
                    print(f'missing {p}')
                    return 1
            d, na, nb = compare(*a.args)
            ok = na == nb and d <= TOL
            print(f'makespan column: {na} and {nb} rows, largest absolute difference {d:.3g} '
                  f'({"equal" if ok else "NOT equal"} within {TOL:g}); the second column, the '
                  'wall-clock time of each instance, is not compared')
            return 0 if ok else 1
    except Refused as e:
        print(f'REFUSING: {e}')
        return 2
    return 2


if __name__ == '__main__':
    sys.exit(main())
