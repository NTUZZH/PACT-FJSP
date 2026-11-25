r"""The second five training seeds of the benchmark from public instances under the active mask.

STATUS: AN ADDITION AFTER THE FIVE-SEED RESULT WAS KNOWN
  The comparison of the policy with the specialists on the benchmark from public instances
  (SD2 test set at 10x5, active mask) was read with the first five training seeds, 301 to 305,
  by scripts/p89_public_family.py --family active. After that result was known, seeds 306 to
  310 of the policy and of the eight specialists were added, for that comparison only. The
  five-seed record of p89 stays the reading of record; the ten-seed reading
  (scripts/p115_public_seed_extension.py) is reported beside it as an added analysis. This is
  not the extension rule of the factory benchmark (scripts/seed_extension.py), which was fixed
  before any result.

THE RUNS OF THE SECOND FIVE SEEDS
  45 runs under the active mask: the policy pubx-act-s306 to -s310 and the specialist
  pubx-soloact-<R>-s306 to -s310 of each of the eight combinations R, each the recipe of the
  first five seeds with nothing changed but the seed and the name. No single-rule control
  and no longer-trained policy has a run of a second seed.
  Names and paths come from the registry: arm_registry.pub_rule_model and pub_rule_paths of
  the run of the same recipe at seed 301, with the run name exchanged, so no pattern is
  written twice. For a first seed every function here returns what the registry returns.
  arm_registry.PUB_ACTFAM_SEEDS is unchanged, so the registry still refuses a second seed and
  every other analysis of the public benchmark keeps seeds 301 to 305.
  Training: scripts/run_pubfamily_queue.sh ARMS=actfam SEED_TIERS="306 307 308 309 310".
  Evaluation: scripts/run_public_seed_extension_eval.sh, each job the job of the evaluation
  plan (scripts/eval_newseeds_plan.py, group pubactfam, pub_SD2_test_10x5+mix) of the same
  recipe at seed 305 with the run name exchanged, into the folder of the first seeds.

A RUN IS COMPLETE when it is trained and evaluated:
  trained    its checkpoint exists; its training log holds the completion line ('done in'),
             29602 parameters on the line '[model] params total=', the trainer's lines of the
             exact mask and of the active mask; its configuration records first_mask exact,
             mask_ties exact and mask_mode active and equals the configuration of the same
             recipe at seed 301 except model_suffix and seed_train, which hold its own name
             and seed (a key that the seed-301 configuration lacks counts as equal while it
             holds the trainer's inert default); its validation record exists;
  evaluated  every array the comparison reads (the policy on the eight combinations, a
             specialist on its own) exists on pub_SD2_test_10x5+mix with one row of
             (makespan, seconds) per test instance, and the summary of the run exists.

Command line (exit 0 yes, 1 no, 2 refused):
  python scripts/pub_seed_extension.py list                the 45 runs and their files
  python scripts/pub_seed_extension.py status [--root D]   trained and evaluated, per run;
                                                           exit 0 only when all 45 are complete
  python scripts/pub_seed_extension.py check [--root D]    the same, the incomplete runs only
  python scripts/pub_seed_extension.py trained <name> ... [--root D]   the queue's check;
                                                           exit 0 when every named run is
  python scripts/pub_seed_extension.py evaluated <name> ... [--root D]
  python scripts/pub_seed_extension.py probecheck <probe configuration> <stem> <combinations>
                                     <updates> [--root D]   the queue's check of a probe
  python scripts/pub_seed_extension.py evaljobs            the evaluation jobs, one line each,
                                                           in the format of
                                                           'eval_newseeds_plan.py jobs'
  python scripts/pub_seed_extension.py rows <n> <file> ... every file exists and every .npy
                                                           file holds n rows
--root resolves every file (default the current folder); PUB_RESULTS (as the evaluation
plan reads it) names the folder of the arrays.
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import arm_registry as REG  # noqa: E402

STATUS = ('added after the five-seed result was known: the five-seed record of '
          'scripts/p89_public_family.py --family active stays the reading of record')
FIRST_SEEDS = tuple(REG.PUB_ACTFAM_SEEDS)
SECOND_SEEDS = (306, 307, 308, 309, 310)
ALL_SEEDS = FIRST_SEEDS + SECOND_SEEDS
SEED_WORDS = {**REG.SEED_WORDS, 306: 'Six', 307: 'Seven', 308: 'Eight', 309: 'Nine', 310: 'Ten'}
ARMS = ('released', 'specialist')       # the policy and the specialists; no control
RULE = REG.PUB_ACT_RULE                 # active
WORD = REG.PUB_RULE_WORD[RULE]          # act
HEADLINE = 'pub_SD2_test_10x5+mix'      # the one test set of the comparison
XS = REG.FOLDER_SUFFIX['exact']         # _xmask
TEMPLATE_SEED = FIRST_SEEDS[0]          # names, paths, configurations: the run of seed 301
EVAL_TEMPLATE_SEED = FIRST_SEEDS[-1]    # evaluation jobs: the plan's job of seed 305
PARAMS = 29602
COMPLETION = 'done in'
MASK_LINE = '[train_family] MASK: first_mask=exact mask_ties=exact'
BANNER = f'[train_family] OPTIONS: mask_mode={RULE} pair_extra=none'
CONFIG_OWN = ('model_suffix', 'seed_train')   # the configuration fields of name and seed
PROBE_OWN = CONFIG_OWN + ('max_updates',)     # and the probe's one update
# keys the trainer added after a configuration was written, at their inert defaults (the
# queue's table without mask_mode, which every run here records as active)
INERT_NEW = {'dr_shift': ('',), 'delta_mode': (None,), 'pair_extra': (None, 'none'),
             'reward_mode': ('dense',), 'pub_type_source': ('none',)}
RECIPE = {'released': ('N,L,S,W,LS,LW', 2000), 'specialist': (None, 1000)}
assert not set(FIRST_SEEDS) & set(SECOND_SEEDS) and len(SECOND_SEEDS) == 5
assert set(SEED_WORDS) >= set(ALL_SEEDS) and HEADLINE in REG.PUB_SETS


class Refused(Exception):
    """A setting, a name or a record that this module does not accept."""


# ----------------------------------------------------------------- seeds and names
def check_seed(seed):
    seed = int(seed)
    if seed not in ALL_SEEDS:
        raise Refused(f'seed {seed} is neither a first seed {FIRST_SEEDS} nor a second seed '
                      f'{SECOND_SEEDS}')
    return seed


def is_second(seed):
    return check_seed(seed) in SECOND_SEEDS


def _check(arm, combo, seed):
    if is_second(seed) and arm not in ARMS:
        raise Refused(f'{arm!r} has no run of a second seed; the second five seeds are the '
                      f'policy and the specialists')
    if arm == 'specialist' and combo not in REG.COMBOS:
        raise Refused(f'a specialist needs a combination of {REG.COMBOS}, got {combo!r}')


def runs():
    """The 45 runs (recipe, combo, seed) in the order of the training queue: the policy runs
    of the five seeds, then the eight specialists of each seed."""
    return ([('released', None, s) for s in SECOND_SEEDS]
            + [('specialist', r, s) for s in SECOND_SEEDS for r in REG.COMBOS])


def name(arm, combo, seed):
    """Run name, e.g. ('specialist', 'W', 306) -> 'pubx-soloact-W-s306'."""
    _check(arm, combo, seed)
    seed = check_seed(seed)
    if seed in FIRST_SEEDS:
        return REG.pub_rule_model(arm, combo, seed)
    t = REG.pub_rule_model(arm, combo, TEMPLATE_SEED)
    sfx = f'-s{TEMPLATE_SEED}'
    if not t.endswith(sfx):
        raise Refused(f'cannot derive the stem of {t!r}')
    return t[:-len(sfx)] + f'-s{seed}'


def stem(arm, combo):
    """The name without its seed suffix, e.g. 'pubx-soloact-W'."""
    return name(arm, combo, TEMPLATE_SEED)[:-len(f'-s{TEMPLATE_SEED}')]


def training_paths(arm, combo, seed):
    """(checkpoint, training log, validation record, configuration): arm_registry.
    pub_rule_paths, for a second seed those of the run of seed 301 with the name exchanged."""
    if not is_second(seed):
        return tuple(REG.pub_rule_paths(arm, combo, int(seed)))
    t = REG.pub_rule_model(arm, combo, TEMPLATE_SEED)
    out = []
    for p in REG.pub_rule_paths(arm, combo, TEMPLATE_SEED):
        if p.count(t) != 1:
            raise Refused(f'cannot derive a path layout from {p!r}')
        out.append(p.replace(t, name(arm, combo, seed)))
    return tuple(out)


def checkpoint(arm, combo, seed):
    return training_paths(arm, combo, seed)[0]


def train_log(arm, combo, seed):
    return training_paths(arm, combo, seed)[1]


def vali_log(arm, combo, seed):
    return training_paths(arm, combo, seed)[2]


def config_path(arm, combo, seed):
    return training_paths(arm, combo, seed)[3]


def results_root():
    """The root of the arrays, as the evaluation plan reads it (PUB_RESULTS)."""
    return os.environ.get('PUB_RESULTS', REG.PUB_RESULTS)


def folder(st=HEADLINE):
    return f'{results_root()}/{st}{XS}'


def result_path(arm, combo, seed, regime):
    """Array of one run on one combination of pub_SD2_test_10x5+mix (the layout of the
    plan's public jobs)."""
    return f'{folder()}/Result_greedy+{REG.PUB_PREFIX}{name(arm, combo, seed)}_{regime}.npy'


def summary_path(arm, combo, seed):
    return f'{folder()}/summary_{REG.PUB_PREFIX}{name(arm, combo, seed)}{XS}.md'


def regimes_read(arm, combo):
    """The combinations the comparison reads: the policy all eight, a specialist its own."""
    return tuple(REG.COMBOS) if arm == 'released' else (combo,)


def paths(arm, combo, seed):
    ck, lg, vl, cf = training_paths(arm, combo, seed)
    return dict(checkpoint=ck, log=lg, vali=vl, config=cf,
                arrays={r: result_path(arm, combo, seed, r) for r in regimes_read(arm, combo)},
                summary=summary_path(arm, combo, seed))


def files(arm, combo, seed):
    """Every file of one run, in a fixed order."""
    p = paths(arm, combo, seed)
    return [p['checkpoint'], p['log'], p['vali'], p['config']] + list(p['arrays'].values()) + \
        [p['summary']]


def find(run_name):
    """(recipe, combo, seed) of a run of the second five seeds, by name."""
    for a, c, s in runs():
        if name(a, c, s) == run_name:
            return a, c, s
    raise Refused(f'{run_name!r} is not a run of the second five seeds')


def find_stem(run_stem):
    """(recipe, combo) whose run name at seed 301 is <run_stem>-s301."""
    for a in ARMS:
        for c in (REG.COMBOS if a == 'specialist' else (None,)):
            if stem(a, c) == run_stem:
                return a, c
    raise Refused(f'{run_stem!r} is the stem of neither the policy nor a specialist under the '
                  f'{RULE} mask')


def n_test(root='.'):
    """Test instances of pub_SD2_test_10x5+mix: the instances of the evaluation plan's jobs."""
    import eval_newseeds_plan as PLAN
    return len(PLAN.instance_names(_at(root, f'data/PUBLIC/family/{HEADLINE}')))


# ----------------------------------------------------------------- state
_PARAMS = re.compile(r'\[model\] params total=([0-9]+)')


def _at(root, p):
    """p resolved against root ('.' leaves the relative path)."""
    return p if root in ('.', '') else os.path.join(root, p)


def drift(ref, new, free=()):
    """Keys whose values differ between two configurations, the keys of free left out; a key
    the reference lacks counts as equal while the other holds its inert default."""
    out = []
    for k in sorted(set(ref) | set(new)):
        if k in free:
            continue
        rv, nv = ref.get(k, '<absent>'), new.get(k, '<absent>')
        if rv == nv or (k in INERT_NEW and k not in ref and nv in INERT_NEW[k]):
            continue
        out.append(f'{k}: {nv!r} against {rv!r}')
    return out


def _json(p):
    with open(p) as f:
        return json.load(f)


def config_drift(arm, combo, seed, root='.'):
    """Differences of the run's configuration from that of the same recipe at seed 301, other
    than its own name and seed; [] when none."""
    mine, ref = (_at(root, config_path(arm, combo, s)) for s in (seed, TEMPLATE_SEED))
    for p in (mine, ref):
        if not os.path.exists(p):
            return [f'missing configuration {p}']
    a, b = _json(mine), _json(ref)
    out = []
    for k, v in (('model_suffix', name(arm, combo, seed)), ('seed_train', int(seed))):
        if a.get(k) != v:
            out.append(f'{k} is {a.get(k)!r}, want {v!r}')
    return out + [f'{x} at seed {TEMPLATE_SEED}' for x in drift(b, a, CONFIG_OWN)]


def training_problems(arm, combo, seed, root='.'):
    """[] when the run is trained (module docstring), otherwise the reasons."""
    p = {k: _at(root, v) for k, v in paths(arm, combo, seed).items() if k != 'arrays'}
    why = []
    if not os.path.exists(p['checkpoint']):
        why.append(f'no checkpoint {p["checkpoint"]}')
    if not os.path.exists(p['log']):
        why.append(f'no training log {p["log"]}')
    else:
        txt = open(p['log'], errors='replace').read()
        lines = txt.splitlines()
        if COMPLETION not in txt:
            why.append(f'{p["log"]} has no completion line')
        m = _PARAMS.search(txt)
        if not m or int(m.group(1)) != PARAMS:
            why.append(f'{p["log"]} reports {m.group(1) if m else "no"} parameters, want {PARAMS}')
        for want, what in ((MASK_LINE, 'the exact mask'), (BANNER, f'the {RULE} mask')):
            if want not in lines:
                why.append(f'{p["log"]} does not state {what} ({want!r})')
    if not os.path.exists(p['config']):
        why.append(f'no configuration {p["config"]}')
    else:
        cfg = _json(p['config'])
        for k, v in (('first_mask', 'exact'), ('mask_ties', 'exact'), ('mask_mode', RULE)):
            if cfg.get(k) != v:
                why.append(f'{p["config"]} records {k}={cfg.get(k)!r}, want {v!r}')
        why += [f'configuration drift against seed {TEMPLATE_SEED}: {x}'
                for x in config_drift(arm, combo, seed, root)]
    if not os.path.exists(p['vali']):
        why.append(f'no validation record {p["vali"]}')
    return why


def evaluation_problems(arm, combo, seed, root='.', n=None):
    """[] when every array the comparison reads holds n rows of two columns and the summary
    exists (n test instances)."""
    n = n_test(root) if n is None else n
    why = []
    for r, f in paths(arm, combo, seed)['arrays'].items():
        f = _at(root, f)
        if not os.path.exists(f):
            why.append(f'missing {f}')
            continue
        shape = np.load(f).shape
        if shape != (n, 2):
            why.append(f'{f}: shape {shape}, the test set has {n} instances')
    s = _at(root, summary_path(arm, combo, seed))
    if not os.path.exists(s):
        why.append(f'missing {s}')
    return why


def state(root='.', n=None):
    """Per run of the second five seeds: name, trained, evaluated and the reasons."""
    n = n_test(root) if n is None else n
    out = []
    for a, c, s in runs():
        tp, ep = training_problems(a, c, s, root), evaluation_problems(a, c, s, root, n)
        out.append(dict(name=name(a, c, s), recipe=a, combo=c, seed=s, trained=not tp,
                        evaluated=not ep, complete=not tp and not ep, training_problems=tp,
                        evaluation_problems=ep))
    return out


def complete(root='.', n=None):
    return all(x['complete'] for x in state(root, n))


# ----------------------------------------------------------------- the queue's probe
def probe_problems(probe_cfg, run_stem, regimes, updates, root='.'):
    """The queue's check of one probe update of a recipe of the second five seeds:
    ([problems], text). The stored configuration of <stem>-s301 must record the recipe (the
    combinations, the updates, seed 301, the exact first-decision and tie options, the active
    mask, the CPU); the probe's configuration must equal it except the name, the seed and the
    one update; the stored configurations of <stem>-s301 to -s305 must be one recipe (they
    differ only in the name and the seed), so the probe stands for every second seed."""
    arm, combo = find_stem(run_stem)
    ref_name = name(arm, combo, TEMPLATE_SEED)
    why = []
    ref_p = _at(root, config_path(arm, combo, TEMPLATE_SEED))
    if not os.path.exists(ref_p):
        return [f'missing configuration {ref_p}'], ''
    if not os.path.exists(probe_cfg):
        return [f'missing probe configuration {probe_cfg}'], ''
    ref, new = _json(ref_p), _json(probe_cfg)
    regs = RECIPE[arm][0] or combo
    for k, v in (('model_suffix', ref_name), ('family_regimes', regs), ('max_updates', RECIPE[arm][1]),
                 ('seed_train', TEMPLATE_SEED), ('first_mask', 'exact'), ('mask_ties', 'exact'),
                 ('mask_mode', RULE), ('family_base', 'sd2'), ('use_type_embedding', False),
                 ('device', 'cpu')):
        if ref.get(k) != v:
            why.append(f'{ref_name} records {k}={ref.get(k)!r}, want {v!r}')
    if regimes != regs or int(updates) != RECIPE[arm][1]:
        why.append(f'the queue names the recipe {regimes} with {updates} updates, the stored run '
                   f'{ref_name} is {regs} with {RECIPE[arm][1]}')
    for k, v in (('mask_mode', RULE), ('max_updates', 1), ('family_regimes', regs)):
        if new.get(k) != v:
            why.append(f'{probe_cfg} records {k}={new.get(k)!r}, want {v!r}')
    why += [f'configuration drift of the probe against {ref_name}: {x}'
            for x in drift(ref, new, PROBE_OWN)]
    for s in FIRST_SEEDS[1:]:
        p = _at(root, config_path(arm, combo, s))
        if not os.path.exists(p):
            why.append(f'missing configuration {p}')
            continue
        other = _json(p)
        extra = drift(ref, other, CONFIG_OWN)
        if extra:
            why.append(f'{name(arm, combo, s)} differs from {ref_name}: {extra}')
        if other.get('seed_train') != s or other.get('model_suffix') != name(arm, combo, s):
            why.append(f'{p} records the name {other.get("model_suffix")!r} and the seed '
                       f'{other.get("seed_train")!r}')
    free = sorted(k for k in set(ref) | set(new) if ref.get(k, '<absent>') != new.get(k, '<absent>'))
    text = (f'[pubfamily] second five seeds {run_stem}: configuration of the probe against '
            f'{ref_name} differs only in {free}; {run_stem}-s{FIRST_SEEDS[0]}..{FIRST_SEEDS[-1]} '
            f'are one recipe under the {RULE} mask ({STATUS})')
    return why, text


# ----------------------------------------------------------------- evaluation jobs
def plan_jobs(seed=EVAL_TEMPLATE_SEED):
    """{(recipe, combo): job} of the evaluation plan's group pubactfam on
    pub_SD2_test_10x5+mix for a first seed: the policy and the eight specialists."""
    import eval_newseeds_plan as PLAN
    seed = int(seed)
    if seed not in FIRST_SEEDS:
        raise Refused(f'the evaluation plan has jobs of the first seeds only, not {seed}')
    js = PLAN.public_actfam_jobs('pubactfam', HEADLINE, REG.PUB_SETS[HEADLINE])
    by_model = {j['model']: j for j in js}
    out = {}
    for arm in ARMS:
        for c in (REG.COMBOS if arm == 'specialist' else (None,)):
            m = REG.PUB_PREFIX + name(arm, c, seed)
            if m not in by_model:
                raise Refused(f'the evaluation plan has no job of {m} on {HEADLINE} (seed {seed} of '
                              f'the policy runs 301 to 303 belongs to group pubact)')
            j = dict(by_model[m])
            j['pre'] = PLAN.preflight_of(j)
            out[(arm, c)] = j
    return out


def renamed_job(job, arm, combo, seed):
    """The job with the run name exchanged and nothing else (the plan's renamed_job): id,
    model, checkpoint, training log and file names."""
    import eval_newseeds_plan as PLAN
    new = name(arm, combo, seed)
    ck, lg = checkpoint(arm, combo, seed), train_log(arm, combo, seed)
    try:
        j = PLAN.renamed_job(job, job['group'], new, ck, lg)
    except AssertionError:
        raise Refused(f'cannot rename the files of the job {job["id"]}')
    old = job['model'][len(REG.PUB_PREFIX):]
    if job['ckpt'].replace(old, new) != ck or job['log'].replace(old, new) != lg:
        raise Refused(f'the checkpoint or log of the job {job["id"]} does not follow the registry')
    return j


def eval_jobs():
    """One job per run of the second five seeds, in the order of runs(): the plan's job of the
    same recipe at seed 305 with the run name exchanged. The arrays and the summary the
    comparison reads must be among the job's files."""
    tmpl = plan_jobs(EVAL_TEMPLATE_SEED)
    out = []
    for a, c, s in runs():
        j = renamed_job(tmpl[(a, c)], a, c, s)
        have = {os.path.normpath(f) for f in j['files']}
        for p in list(paths(a, c, s)['arrays'].values()) + [summary_path(a, c, s)]:
            if os.path.normpath(p) not in have:
                raise Refused(f'the job {j["id"]} does not write {p} (PUB_RESULTS of the plan and '
                              f'of this module must name one folder)')
        out.append(j)
    return out


def job_line(job):
    """A job as one line of 'eval_newseeds_plan.py jobs'."""
    import eval_newseeds_plan as PLAN
    return '|'.join(str(job[k]) for k in PLAN.KEYS_JOB) + '|' + ' '.join(job['files'])


# ----------------------------------------------------------------- command line
def _rows(n, fs):
    bad = []
    for f in fs:
        if not os.path.exists(f):
            bad.append(f'missing {f}')
        elif f.endswith('.npy') and np.load(f).shape[0] != n:
            bad.append(f'{f}: {np.load(f).shape[0]} rows, want {n}')
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser(description='the second five training seeds of the benchmark '
                                             'from public instances (' + STATUS + ')')
    ap.add_argument('cmd', choices=('list', 'status', 'check', 'trained', 'evaluated',
                                    'probecheck', 'evaljobs', 'rows'))
    ap.add_argument('args', nargs='*')
    ap.add_argument('--root', default='.')
    a = ap.parse_args(argv)
    try:
        if a.cmd == 'list':
            for x, c, s in runs():
                print(f'{name(x, c, s)}  ' + '  '.join(files(x, c, s)))
            return 0
        if a.cmd in ('status', 'check'):
            st = state(a.root)
            tr, ev, co = (sum(x[k] for x in st) for k in ('trained', 'evaluated', 'complete'))
            print(f'second five seeds of the benchmark from public instances, active mask '
                  f'({STATUS}): {len(st)} runs, {tr} trained, {ev} evaluated, {co} complete')
            shown = st if a.cmd == 'status' else [x for x in st if not x['complete']][:6]
            for x in shown:
                why = x['training_problems'] + x['evaluation_problems']
                print(f'  {x["name"]:22s} trained {"yes" if x["trained"] else "no "}  evaluated '
                      f'{"yes" if x["evaluated"] else "no "}' + (f'  ({why[0]}'
                      + (f'; and {len(why) - 1} more' if len(why) > 1 else '') + ')' if why else ''))
            if a.cmd == 'check' and co < len(st) - len(shown):
                print(f'  ... and {len(st) - co - len(shown)} more incomplete runs')
            return 0 if co == len(st) else 1
        if a.cmd in ('trained', 'evaluated'):
            if not a.args:
                ap.error(f'{a.cmd} takes one or more run names')
            runs_ = [find(x) for x in a.args]
            n = n_test(a.root) if a.cmd == 'evaluated' else None
            bad = 0
            for nm, run in zip(a.args, runs_):
                why = (training_problems(*run, root=a.root) if a.cmd == 'trained'
                       else evaluation_problems(*run, root=a.root, n=n))
                for x in why:
                    print(f'{nm}: {x}')
                if not why:
                    print(f'{nm}: {a.cmd}')
                bad += bool(why)
            return 0 if not bad else 1
        if a.cmd == 'probecheck':
            if len(a.args) != 4:
                ap.error('probecheck takes the probe configuration, the stem, the combinations '
                         'and the updates')
            why, text = probe_problems(*a.args, root=a.root)
            if why:
                for x in why:
                    print(f'ABORT: {x}', file=sys.stderr)
                return 1
            print(text)
            return 0
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
