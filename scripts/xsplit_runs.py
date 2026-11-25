r"""Names, paths and checks of the policies trained with other held-out pairs of combinations.

THE RUNS
  The policy xmask-v1-s<seed> trains on N, L, S, W, LS, LW and is tested on the held-out
  combinations SW and LSW. Two further policies, the recipe of xmask-v1 in every option except
  the training combinations:
    xsplit-b-s<seed>  trains on N, L, S, W, LW, SW   holds out LS and LSW
    xsplit-c-s<seed>  trains on N, L, S, W, LS, SW   holds out LW and LSW
  Training seeds 301, 302, 303; 2000 updates; exact mask; 32786 parameters. --family_regimes
  lists the combinations in the order of arm_registry.COMBOS, as the stored policy, the
  single-rule control and xmask-all8 pass theirs (N,L,S,W,LS,LW; N,L,S,W;
  N,L,S,W,LS,LW,SW,LSW).
    training    scripts/run_heldout_splits_queue.sh
    evaluation  scripts/run_heldout_splits_eval.sh: one greedy pass on the test instances of
                all eight combinations at 10x25 with the call of the evaluation plan's
                principal job of xmask-v1-s<seed> (scripts/eval_newseeds_plan.py, read
                through seed_extension.plan_jobs), the run name exchanged; the arrays go to
                <results>/10x25_xsplit
    analysis    scripts/p122_heldout_splits.py
  Checkpoint, training log, configuration and validation record follow the registry's helpers
  for xmask-v1-s<seed> with the run name exchanged.

A RUN IS TRAINED when
  its checkpoint exists; its training log holds the header line of train_family.py with its own
  model name, its six training combinations in this order and 2000 updates, the line of the
  exact mask, 32786 parameters and the completion line; its configuration records the exact
  mask and equals the configuration of xmask-v1-s<seed> except model_suffix (its own name) and
  family_regimes (its own six combinations, against N,L,S,W,LS,LW); its validation record
  exists.
A RUN IS EVALUATED when its eight arrays <results>/10x25_xsplit/Result_greedy+10x25+family+
  <name>_<R>.npy hold one row per test instance of R at 10x25.

THE PROBE (scripts/run_heldout_splits_queue.sh, before any training)
  One update on the CPU with seed 301 per split. The probe's configuration records the exact
  mask and the split's combinations, and equals the stored configuration of xmask-v1-s301
  except model_suffix, seed_train, device, family_regimes and max_updates (1 in the probe). Its
  log holds 32786 parameters and the mask line, and its MWKR norms line names exactly the
  split's six combinations, each with the norm of that combination in a stored log that
  validates on it: the log of xmask-v1-s301 for N, L, S, W, LS and LW, the log of
  xmask-solo-<R>-s301 for a combination the policy does not validate on (SW). The norm of a
  combination depends only on its validation instances (base instances and descriptor seeds
  per instance and combination, train_family.py), not on the other combinations of the run.

Command line (the environment is exact; PACT_ENV defaults to exact here):
  python scripts/xsplit_runs.py list [--seeds S]       the runs and their files
  python scripts/xsplit_runs.py state [--seeds S] [--root D]
  python scripts/xsplit_runs.py probe <split> <suffix> <log> [--root D]
  python scripts/xsplit_runs.py trained <name> [--root D]
  python scripts/xsplit_runs.py evaljob <name>          one line in the format of
                                                        'eval_newseeds_plan.py jobs'
  python scripts/xsplit_runs.py verifyjob               the plan's job of xmask-v1-s301 on LSW,
                                                        first 10 instances
  python scripts/xsplit_runs.py rows <n> <file> ...
  python scripts/xsplit_runs.py compare <a.npy> <b.npy> [<n>]
                                       largest difference of the makespan columns over the
                                       first n rows (default all); exit 0 when a has n rows,
                                       b at least n, and it is at most 1e-6
--seeds takes '301', '301,302' or '301-303' (default 301-303).
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
import longspec_runs as LR  # noqa: E402

ENV = 'exact'
COMBOS = tuple(REG.COMBOS)
SPLITS = {'b': ('N', 'L', 'S', 'W', 'LW', 'SW'), 'c': ('N', 'L', 'S', 'W', 'LS', 'SW')}
HELD = {k: tuple(r for r in COMBOS if r not in v) for k, v in SPLITS.items()}
SEEDS = (301, 302, 303)
UPDATES = 2000
PARAMS = LR.PARAMS
FOLDER = '10x25'
RESULT_FOLDER = '10x25_xsplit'
PROBE_SEED = 301
PROBE_ALLOWED = ('model_suffix', 'seed_train', 'device', 'family_regimes', 'max_updates')
RUN_ALLOWED = ('model_suffix', 'family_regimes')
VERIFY = ('LSW', 10)                 # the combination and the first instances of the pre-flight
TOL = 1e-6
MASK_LINE = LR.MASK_LINE
for _k, _v in SPLITS.items():        # the trainer's order: arm_registry.COMBOS
    assert list(_v) == [r for r in COMBOS if r in _v] and len(_v) == 6 and len(HELD[_k]) == 2

_NAME = re.compile(r'^xsplit-(' + '|'.join(SPLITS) + r')-s(\d+)$')
_PAIRS = re.compile(r'\b([A-Z]+)=([\d.]+)')
Refused = LR.Refused


# ----------------------------------------------------------------- setting and names
def require_exact():
    if REG.env() != ENV:
        raise Refused(f'the runs exist in the {ENV} environment only; the active environment is '
                      f'{REG.env()} (set PACT_ENV={ENV})')


def check_seed(seed):
    seed = int(seed)
    if seed not in SEEDS:
        raise Refused(f'seed {seed}: the split policies take the seeds {SEEDS[0]} to {SEEDS[-1]}')
    return seed


def check_split(split):
    if split not in SPLITS:
        raise Refused(f'split {split!r}: want one of {tuple(SPLITS)}')
    return split


def parse_seeds(text):
    """'301', '301,302' or '301-303' -> (301, ...); seeds 301 to 303, each once."""
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


def regimes_arg(split):
    """The value of --family_regimes of a split."""
    return ','.join(SPLITS[check_split(split)])


def name(split, seed):
    return f'xsplit-{check_split(split)}-s{check_seed(seed)}'


def main_name(seed):
    """The policy xmask-v1-s<seed> (arm_registry)."""
    return REG.model('released', None, check_seed(seed), ENV)


def find(run_name):
    m = _NAME.match(str(run_name))
    if not m:
        raise Refused(f'{run_name!r} is not a run xsplit-<split>-s<seed>')
    return m.group(1), check_seed(m.group(2))


def runs(seeds=SEEDS):
    """(split, seed) of every run, per seed the splits b and c."""
    return [(x, check_seed(s)) for s in seeds for x in SPLITS]


def _exchanged(path_of, split, seed):
    t = main_name(seed)
    p = path_of('released', None, seed, ENV)
    if p.count(t) != 1:
        raise Refused(f'cannot derive a path layout from {p!r}')
    return p.replace(t, name(split, seed))


def checkpoint(split, seed):
    return _exchanged(REG.checkpoint, split, seed)


def train_log(split, seed):
    return _exchanged(REG.train_log, split, seed)


def config_path(split, seed):
    return _exchanged(REG.config_path, split, seed)


def vali_log(split, seed):
    return _exchanged(REG.vali_log, split, seed)


def result_dir():
    return f'{REG.results_root()}/{RESULT_FOLDER}'


def result_path(split, seed, combo):
    """The array of the run on one combination at 10x25."""
    require_exact()
    if combo not in COMBOS:
        raise Refused(f'combination {combo!r}: want one of {COMBOS}')
    p = os.path.basename(REG.result_path('released', None, check_seed(seed), FOLDER, combo))
    t = main_name(seed)
    if p.count(t) != 1:
        raise Refused(f'cannot derive an array name from {p!r}')
    return f'{result_dir()}/{p.replace(t, name(split, seed))}'


def paths(split, seed):
    return dict(checkpoint=checkpoint(split, seed), log=train_log(split, seed),
                config=config_path(split, seed), vali=vali_log(split, seed),
                arrays={r: result_path(split, seed, r) for r in COMBOS})


def n_test(combo):
    return LR.n_test(combo)


def _at(root, p):
    return p if root in ('.', '') else os.path.join(root, p)


# ----------------------------------------------------------------- the norms
def norm_pairs(line):
    """[(combination, value text)] of a norms line, in its order."""
    return _PAIRS.findall(line or '')


def reference_norms(combos, root='.'):
    """{combination: (norm text, log)}: from the log of xmask-v1-s301 when it validates on the
    combination, else from the log of xmask-solo-<R>-s301."""
    main_log = _at(root, REG.train_log('released', None, PROBE_SEED, ENV))
    main = dict(norm_pairs(LR.norms_line(main_log)))
    out = {}
    for r in combos:
        if r in main:
            out[r] = (main[r], main_log)
            continue
        log = _at(root, REG.train_log('specialist', r, PROBE_SEED, ENV))
        got = dict(norm_pairs(LR.norms_line(log)))
        out[r] = (got.get(r), log)
    return out


# ----------------------------------------------------------------- checks
def probe_problems(split, suffix, probe_log, root='.'):
    """([] or the reasons, the configuration differences, the norms line) of a split's probe."""
    train = SPLITS[check_split(split)]
    ref_cfg = _at(root, REG.config_path('released', None, PROBE_SEED, ENV))
    cfg_p = _at(root, f'train_log/FAMILY/config_{REG.PREFIX}{suffix}.json')
    why = [f'missing {p}' for p in (ref_cfg, cfg_p, probe_log) if not os.path.exists(p)]
    if why:
        return why, [], None
    txt = open(probe_log, errors='replace').read()
    m = LR._PARAMS.search(txt)
    if not m or int(m.group(1)) != PARAMS:
        why.append(f'{probe_log} reports {m.group(1) if m else "no"} parameters, want {PARAMS}')
    if MASK_LINE not in txt.splitlines():
        why.append(f'{probe_log} has no line "{MASK_LINE}"')
    probe, ref = json.load(open(cfg_p)), json.load(open(ref_cfg))
    for k in ('first_mask', 'mask_ties'):
        if probe.get(k) != 'exact':
            why.append(f'the probe configuration records {k}={probe.get(k)!r}, want exact')
    if probe.get('family_regimes') != regimes_arg(split):
        why.append(f'the probe trains {probe.get("family_regimes")!r}, want {regimes_arg(split)!r}')
    if probe.get('model_suffix') != suffix:
        why.append(f'the probe configuration names {probe.get("model_suffix")!r}, want {suffix!r}')
    if (ref.get('family_regimes') != ','.join(REG.TRAIN_COMBOS)
            or int(ref.get('max_updates', -1)) != UPDATES or int(ref.get('seed_train', -1)) != PROBE_SEED):
        why.append(f'{ref_cfg} records family_regimes={ref.get("family_regimes")!r}, max_updates='
                   f'{ref.get("max_updates")}, seed_train={ref.get("seed_train")}')
    diffs = LR.config_differences(probe, ref)
    bad = [k for k in diffs if k not in PROBE_ALLOWED]
    if bad:
        why.append(f'configuration drift against {main_name(PROBE_SEED)}: '
                   + '; '.join(f'{k}: {ref.get(k, "<absent>")!r} stored, {probe.get(k, "<absent>")!r} '
                               'in the probe' for k in bad))
    line = LR.norms_line(probe_log)
    got = norm_pairs(line)
    if [r for r, _ in got] != list(train):
        why.append(f'MWKR norms line {line!r} does not name {",".join(train)} in this order')
    else:
        want = reference_norms(train, root)
        for r, v in got:
            if want[r][0] != v:
                why.append(f'MWKR norm of {r}: {v} in the probe, {want[r][0]} in {want[r][1]}')
    return why, diffs, line


def training_problems(split, seed, root='.'):
    """[] when the run is trained (module docstring), otherwise the reasons."""
    nm, train = name(split, seed), SPLITS[split]
    p = {k: _at(root, v) for k, v in paths(split, seed).items() if k != 'arrays'}
    ref_cfg = _at(root, REG.config_path('released', None, seed, ENV))
    why = []
    if not os.path.exists(p['checkpoint']):
        why.append(f'no checkpoint {p["checkpoint"]}')
    if not os.path.exists(p['log']):
        why.append(f'no training log {p["log"]}')
    else:
        txt = open(p['log'], errors='replace').read()
        head = LR._HEAD.findall(txt)
        if len(head) != 1:
            why.append(f'{p["log"]} holds {len(head)} header lines of train_family.py, want one')
        else:
            model, regimes, upd = head[0]
            regs = tuple(x.strip().strip("'\"") for x in regimes.split(',') if x.strip())
            if model != REG.PREFIX + nm:
                why.append(f'{p["log"]} is the log of {model}')
            if regs != train:
                why.append(f'{p["log"]} trains {regs}, want {train}')
            if int(upd) != UPDATES:
                why.append(f'{p["log"]} declares {upd} updates, want {UPDATES}')
        if MASK_LINE not in txt.splitlines():
            why.append(f'{p["log"]} has no line "{MASK_LINE}"')
        m = LR._PARAMS.search(txt)
        if not m or int(m.group(1)) != PARAMS:
            why.append(f'{p["log"]} reports {m.group(1) if m else "no"} parameters, want {PARAMS}')
        if 'done in' not in txt:
            why.append(f'{p["log"]} has no completion line')
    if not os.path.exists(p['config']):
        why.append(f'no configuration {p["config"]}')
    elif not os.path.exists(ref_cfg):
        why.append(f'no configuration {ref_cfg} of {main_name(seed)}')
    else:
        a, b = json.load(open(p['config'])), json.load(open(ref_cfg))
        for k in ('first_mask', 'mask_ties'):
            if a.get(k) != 'exact':
                why.append(f'{p["config"]} records {k}={a.get(k)!r}, want exact')
        want = {'model_suffix': (nm, main_name(seed)),
                'family_regimes': (regimes_arg(split), ','.join(REG.TRAIN_COMBOS))}
        for k, (mine, ref) in want.items():
            if a.get(k) != mine or b.get(k) != ref:
                why.append(f'{k}: {a.get(k)!r} against {b.get(k)!r} in {main_name(seed)}, want '
                           f'{mine!r} against {ref!r}')
        for k in LR.config_differences(a, b):
            if k not in RUN_ALLOWED:
                why.append(f'configuration drift against {main_name(seed)}: {k}: '
                           f'{a.get(k, "<absent>")!r} against {b.get(k, "<absent>")!r}')
    if not os.path.exists(p['vali']):
        why.append(f'no validation record {p["vali"]}')
    return why


def evaluation_problems(split, seed):
    """[] when the run's eight arrays exist with one row per test instance."""
    bad = []
    for r in COMBOS:
        bad += LR.rows_problems(n_test(r), [result_path(split, seed, r)])
    return bad


def state(seeds=SEEDS, root='.'):
    require_exact()
    out = []
    for x, s in runs(seeds):
        tp, ep = training_problems(x, s, root), evaluation_problems(x, s)
        out.append(dict(name=name(x, s), split=x, seed=s, trained=not tp, evaluated=not ep,
                        complete=not tp and not ep, training_problems=tp,
                        evaluation_problems=ep))
    return out


def complete_seeds(root='.'):
    """The seeds whose two split runs are trained and evaluated."""
    st = state(SEEDS, root)
    return tuple(s for s in SEEDS if all(x['complete'] for x in st if x['seed'] == s))


# ----------------------------------------------------------------- arrays
def compare(a_path, b_path, n=None):
    """(largest absolute difference of the makespan columns over the first n rows, rows of a,
    rows of b); a must hold n rows (all of b when n is None) and b at least n."""
    a, b = LR.makespans(a_path), LR.makespans(b_path)
    n = len(b) if n is None else int(n)
    if len(a) != n or len(b) < n:
        return float('inf'), len(a), len(b)
    return (float(np.max(np.abs(a - b[:n]))) if n else 0.0), len(a), len(b)


# ----------------------------------------------------------------- evaluation jobs
def plan_job(seed):
    """The evaluation plan's principal 10x25 job of xmask-v1-s<seed>."""
    require_exact()
    import seed_extension as SE
    try:
        j = SE.plan_jobs(check_seed(seed))[('released', None)]
    except SE.Refused as e:
        raise Refused(str(e))
    want = dict(env=ENV, device='cpu', methods='greedy', sched='-', campaign=FOLDER,
                regimes=' '.join(COMBOS), limit=0)
    bad = {k: j.get(k) for k, v in want.items() if j.get(k) != v}
    if bad:
        raise Refused(f'the job {j["id"]} is not a greedy CPU job of the exact environment on the '
                      f'eight combinations at {FOLDER} without schedule export: {bad}')
    return j


def eval_job(split, seed):
    """The plan's job of xmask-v1-s<seed> with the run name exchanged; files: the run's eight
    arrays and summary in result_dir()."""
    j = plan_job(seed)
    old, new = j['model'], REG.PREFIX + name(split, seed)
    if old != REG.PREFIX + main_name(seed) or not j['id'].endswith(':' + main_name(seed)):
        raise Refused(f'unexpected layout of the job {j["id"]}')
    files = [result_path(split, seed, r) for r in COMBOS] + [f'{result_dir()}/summary_{new}{j["tag"]}.md']
    for r, f in zip(COMBOS, files):
        if os.path.basename(f) != f'Result_greedy+{new}_{r}.npy':
            raise Refused(f'the array name {f} does not follow eval_family.py')
        if n_test(r) != int(j['n']):
            raise Refused(f'{r} has {n_test(r)} test instances at {FOLDER}, the job {j["id"]} '
                          f'expects {j["n"]}')
    return dict(j, id=f'xsplit:{FOLDER}:{name(split, seed)}', model=new,
                ckpt=checkpoint(split, seed), log=train_log(split, seed), files=files, cost=None)


def verify_job():
    """The plan's job of xmask-v1-s301 on VERIFY's combination and first instances; file: the
    stored array."""
    combo, n = VERIFY
    j = plan_job(PROBE_SEED)
    return dict(j, regimes=combo, limit=n, n=n,
                files=[REG.result_path('released', None, PROBE_SEED, FOLDER, combo)], cost=None)


def job_line(job):
    import eval_newseeds_plan as PLAN
    return '|'.join(str(job[k]) for k in PLAN.KEYS_JOB) + '|' + ' '.join(job['files'])


# ----------------------------------------------------------------- command line
def main(argv=None):
    os.environ.setdefault('PACT_ENV', ENV)
    ap = argparse.ArgumentParser(description='the policies trained with other held-out pairs')
    ap.add_argument('cmd', choices=('list', 'state', 'probe', 'trained', 'evaljob', 'verifyjob',
                                    'rows', 'compare'))
    ap.add_argument('args', nargs='*')
    ap.add_argument('--seeds', default=None)
    ap.add_argument('--root', default='.')
    a = ap.parse_args(argv)
    try:
        require_exact()
        seeds = parse_seeds(a.seeds) if a.seeds else SEEDS
        if a.cmd == 'list':
            for x, s in runs(seeds):
                p = paths(x, s)
                print(f'{name(x, s)}  {regimes_arg(x)}  {p["checkpoint"]}  {p["log"]}  {p["config"]}  '
                      f'{p["vali"]}  ' + ' '.join(p['arrays'].values()))
            return 0
        if a.cmd == 'state':
            st = state(seeds, a.root)
            print(f'policies with other held-out pairs, seeds {",".join(map(str, seeds))}: {len(st)} '
                  f'runs, {sum(x["trained"] for x in st)} trained, '
                  f'{sum(x["evaluated"] for x in st)} evaluated')
            for x in st:
                why = x['training_problems'] + x['evaluation_problems']
                print(f'  {x["name"]:16s} trained {"yes" if x["trained"] else "no "}  evaluated '
                      f'{"yes" if x["evaluated"] else "no "}'
                      + (f'  ({why[0]}' + (f'; and {len(why) - 1} more' if len(why) > 1 else '')
                         + ')' if why else ''))
            return 0 if all(x['complete'] for x in st) else 1
        if a.cmd == 'probe':
            if len(a.args) != 3:
                ap.error('probe <split> <suffix> <probe log>')
            why, diffs, line = probe_problems(*a.args, root=a.root)
            if why:
                print(f'ABORT: the probe of split {a.args[0]}: ' + '; '.join(why))
                return 1
            srcs = sorted({os.path.basename(v[1]) for v in
                           reference_norms(SPLITS[a.args[0]], a.root).values()})
            print(f'[xsplit] {a.args[0]} ({regimes_arg(a.args[0])}): configuration against '
                  f'{main_name(PROBE_SEED)} OK, differs only in {diffs}; MWKR norms {line} equal '
                  f'to the stored ({", ".join(srcs)}); {PARAMS} parameters, exact mask')
            return 0
        if a.cmd == 'trained':
            if len(a.args) != 1:
                ap.error('trained <name>')
            x, s = find(a.args[0])
            why = training_problems(x, s, a.root)
            if why:
                print(f'{a.args[0]} is not trained as required: ' + '; '.join(why))
                return 1
            print(f'{a.args[0]}: trained ({regimes_arg(x)}, {UPDATES} updates, {PARAMS} parameters, '
                  f'exact mask, configuration of {main_name(s)} except model_suffix and '
                  'family_regimes)')
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
            bad = LR.rows_problems(int(a.args[0]), a.args[1:])
            print('; '.join(bad) if bad else f'{len(a.args) - 1} files, {a.args[0]} rows each')
            return 1 if bad else 0
        if a.cmd == 'compare':
            if len(a.args) not in (2, 3):
                ap.error('compare <a.npy> <b.npy> [<n>]')
            for p in a.args[:2]:
                if not os.path.exists(p):
                    print(f'missing {p}')
                    return 1
            n = int(a.args[2]) if len(a.args) == 3 else None
            d, na, nb = compare(a.args[0], a.args[1], n)
            ok = d <= TOL
            print(f'makespan column, first {na if n is None else n} rows: {na} and {nb} rows, '
                  f'largest absolute difference {d:.3g} ({"equal" if ok else "NOT equal"} within '
                  f'{TOL:g}); the second column, the wall-clock time of each instance, is not '
                  'compared')
            return 0 if ok else 1
    except Refused as e:
        print(f'REFUSING: {e}')
        return 2
    return 2


if __name__ == '__main__':
    sys.exit(main())
