r"""Job plan, pre-flight and completeness check of scripts/run_further_sets_eval.sh.

Two evaluation-only test sets in the environment of the paper (exact non-delay mask), each
evaluated with the calls the evaluation plan (scripts/eval_newseeds_plan.py) makes on its model
set, the set exchanged and nothing else:

  set               model set        data                          base instances
  27x25             20x25            data/FAMILY/27x25             data/PPVC/27x25+ppvc-mixed
  10x25_asymsetup   10x25_oodsetup   data/FAMILY/10x25_asymsetup   data/PPVC/10x25+ppvc-mixed

27x25 is the per-floor batch (scripts/p57_build_scale_data.py, seeds 140000 to 140049);
10x25_asymsetup holds asymmetric setup matrices at the trained magnitude
(scripts/p29_ood.py).

JOBS OF A SET (the plan's groups principal and rules on the model set, eval_newseeds_plan
principal_runs and rules_job)
  the policy xmask-v1-s<seed> on the eight combinations, each specialist xmask-solo-<R>-s<seed>
  on its own combination (the plan evaluates the single-rule control on 10x25 only, so no
  control job), and one job of the four dispatching rules FIFO, MOR, SPT and MWKR on the eight
  combinations under the exact mask (--rule_set masked). Every job: CPU, OMP and MKL threads 1,
  --device cpu --device_id "", --torch_threads 1, --first_mask exact --mask_ties exact,
  --tag _xmask. A job is the plan's job of the same run on the model set with data, base,
  result folder, file names, campaign, id, instance count and cost exchanged
  (exchange_set); the command is built by build_cmd of scripts/run_eval_newseeds.sh.

SEEDS (SEEDS, default '301 302 303 304 305'; 306 to 310 accepted)
  A first seed takes the plan's job of that seed. A second seed (306 to 310, the policy and the
  specialists only) takes the job of seed 305 with the run name exchanged
  (scripts/seed_extension.py renamed_job). A run is evaluated only when it is trained: a first
  seed by the plan's own condition (checkpoint present, completion line 'done in' in the
  training log), a second seed by scripts/seed_extension.py training_problems (completion line,
  32786 parameters, exact mask, the configuration of seed 301 but for name and seed).

RESULTS
  arm_registry.res_dir(<set>) of the exact environment, test_results/FAMILY/<set>_xmask: the
  folder eval_family.py writes for --data data/FAMILY/<set> --tag _xmask. Every file of every job
  lies in it (checked).

PRE-FLIGHT (one per set)
  The plan's job of the seed-301 policy on the model set, on the first PRE_N instances of
  PRE_REG, into a staging folder; its arrays must equal the stored arrays of the model set bit
  for bit (eval_newseeds_plan.py compare). It proves the command builder and the evaluation path
  against arrays the plan wrote.

TIME (estimate)
  Seconds per rollout from the plan's table eval_newseeds_plan.SPEED (one core, one thread,
  measured 2026-11-11): the model size for 10x25_asymsetup (10x25); for 27x25 the log-log
  interpolation in the module count between 20x25 and 30x25, t20 (27/20)^k with
  k = ln(t30/t20) / ln(30/20).

Subcommands:
  jobs <set>                  one line per job, the format of 'eval_newseeds_plan.py jobs'
  preflight <set>             the pre-flight job line, then the stored folder of the model set
  trained <model>             exit 0 when the run is trained (the condition above)
  rows <n> <file> ...         exit 0 when every file exists and every .npy holds n rows
  check <set>                 exit 0 when every file of every job of the set exists, every array
                              with one row per test instance
  estimate [<set>]            jobs, rollouts and core-hours per set
"""
import contextlib
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import arm_registry as REG  # noqa: E402

SETS = {
    '27x25': dict(model='20x25', data='data/FAMILY/27x25', base='data/PPVC/27x25+ppvc-mixed',
                  what='per-floor batch, 27 modules on the default factory of 25 stations'),
    '10x25_asymsetup': dict(model='10x25_oodsetup', data='data/FAMILY/10x25_asymsetup',
                            base='data/PPVC/10x25+ppvc-mixed',
                            what='asymmetric setup matrices at the trained magnitude'),
}
ENV = 'exact'
FIRST = (301, 302, 303, 304, 305)
SECOND = (306, 307, 308, 309, 310)
TEMPLATE_SEED = 305                     # the job of a second seed: the plan's job of seed 305
PRE_SEED, PRE_N, PRE_REG = 301, 3, ('S', 'LSW')
GROUPS = ('principal', 'rules')


class Refused(Exception):
    """A setting, a set or a job that this plan does not accept."""


def _plan():
    import eval_newseeds_plan as PLAN
    return PLAN


def require_exact():
    if REG.env() != ENV:
        raise Refused(f'the further sets are evaluated in the {ENV} environment; the active '
                      f'environment is {REG.env()} (set PACT_ENV={ENV})')


def check_set(name):
    if name not in SETS:
        raise Refused(f'unknown set {name!r}; want one of {", ".join(SETS)}')
    return SETS[name]


def seeds():
    """SEEDS (default the first five), each a first or a second seed, increasing."""
    text = os.environ.get('SEEDS', ' '.join(map(str, FIRST)))
    out = [int(x) for x in text.replace(',', ' ').split()]
    bad = [s for s in out if s not in FIRST + SECOND]
    if not out or bad or out != sorted(set(out)):
        raise Refused(f'SEEDS={text!r}: want increasing seeds from {FIRST + SECOND}')
    return out


@contextlib.contextmanager
def plan_setting(plan, groups, seed_list):
    """The evaluation plan with only these groups and exact seeds, restored afterwards."""
    old = plan.GROUPS, plan.EXACT_SEEDS
    plan.GROUPS, plan.EXACT_SEEDS = list(groups), list(seed_list)
    try:
        yield plan
    finally:
        plan.GROUPS, plan.EXACT_SEEDS = old


def model_jobs(name, seed):
    """The plan's jobs of the groups principal and rules on the model set of a set, for one
    first seed: {model: job}."""
    s = check_set(name)
    if seed not in FIRST:
        raise Refused(f'the evaluation plan has jobs of the first seeds only, not {seed}')
    P = _plan()
    with plan_setting(P, GROUPS, [seed]):
        js = [j for j in P.exact_jobs() if j['campaign'] == s['model']]
    if not js:
        raise Refused(f'the evaluation plan has no job on {s["model"]}')
    return {j['model']: dict(j) for j in js}


def set_n(name):
    """Test instances of a set (the plan's instance_names of its data folder)."""
    s = check_set(name)
    if not os.path.isdir(os.path.join(s['data'], 'desc', 'N')):
        raise Refused(f'{s["data"]} holds no descriptors; build the set first')
    return len(_plan().instance_names(s['data']))


def check_data(name):
    """The set's descriptors name its base folder and cover the eight combinations equally."""
    import json
    s = check_set(name)
    meta = os.path.join(s['data'], 'dataset_meta.json')
    if not os.path.exists(meta):
        raise Refused(f'{meta} is missing; build the set first')
    with open(meta) as f:
        m = json.load(f)
    if m.get('base_dir') != s['base']:
        raise Refused(f'{meta} names the base folder {m.get("base_dir")!r}, want {s["base"]}')
    n = set_n(name)
    for r in REG.COMBOS:
        d = os.path.join(s['data'], 'desc', r)
        k = len([x for x in os.listdir(d) if x.endswith('.desc.json')]) if os.path.isdir(d) else 0
        if k != n:
            raise Refused(f'{s["data"]}/desc/{r} holds {k} descriptors, N holds {n}')
    if m.get('n_instances') != n:
        raise Refused(f'{meta} records {m.get("n_instances")} instances, the folder holds {n}')
    return n


def speed(name):
    """(seconds per policy rollout, seconds per rule rollout) of the set (module docstring)."""
    s = check_set(name)
    P = _plan()
    key = os.path.basename(s['data'])
    if key in P.SPEED:
        return P.SPEED[key]
    size = P.size_key(s['data'])
    if size in P.SPEED:
        return P.SPEED[size]
    j = int(size.split('x')[0])
    lo, hi = P.SPEED['20x25'], P.SPEED['30x25']
    if not 20 < j < 30 or size.split('x')[1] != '25':
        raise Refused(f'no rollout time for {size} and no neighbours to interpolate from')
    return tuple(a * (j / 20.0) ** (math.log(b / a) / math.log(30.0 / 20.0))
                 for a, b in zip(lo, hi))


def exchange_set(job, name):
    """The plan's job on the model set with the set exchanged: data, base, folder, the file
    names, campaign, id, instance count, cost and pre-flight. Nothing else changes."""
    s = check_set(name)
    P = _plan()
    model_data, model_base = P.FACTORY[s['model']]
    if job['data'] != model_data or job['base'] != model_base or job['campaign'] != s['model']:
        raise Refused(f'the job {job["id"]} is not a job of the model set {s["model"]}')
    old_folder, new_folder = job['folder'], os.path.basename(s['data']) + P.XS
    if old_folder != os.path.basename(model_data) + P.XS:
        raise Refused(f'unexpected result folder {old_folder} of the job {job["id"]}')
    old_pre, new_pre = f'{job["out_root"]}/{old_folder}/', f'{job["out_root"]}/{new_folder}/'
    if job['sched'] != '-' or not all(f.startswith(old_pre) for f in job['files']):
        raise Refused(f'the job {job["id"]} writes outside {old_pre}')
    head = f'exact:{s["model"]}:'
    if not job['id'].startswith(head):
        raise Refused(f'unexpected id {job["id"]}')
    n = set_n(name)
    t_pol, t_rule = speed(name)
    per = t_rule if job['model'] == 'rules' else t_pol
    nroll = n * len(job['regimes'].split()) * len(job['methods'].split())
    out = dict(job, id=f'exact:{name}:' + job['id'][len(head):], campaign=name, data=s['data'],
               base=s['base'], folder=new_folder, n=n,
               files=[new_pre + f[len(old_pre):] for f in job['files']],
               cost=per * nroll + P.JOB_OVERHEAD, pre=f'further_{name}')
    res = os.path.normpath(REG.res_dir(name))
    if os.path.normpath(os.path.join(out['out_root'], new_folder)) != res:
        raise Refused(f'the job writes {out["out_root"]}/{new_folder}, arm_registry.res_dir gives '
                      f'{res} (the evaluation plan RESULTS and PACT_RESULTS must name one root)')
    return out


def jobs(name, seed_list=None):
    """Every job of a set for the seeds (module docstring), longest first."""
    require_exact()
    import seed_extension as SE
    seed_list = seeds() if seed_list is None else list(seed_list)
    out, rules = [], None
    for s in seed_list:
        if s in FIRST:
            mj = model_jobs(name, s)
            for m, j in mj.items():
                if m == 'rules':
                    rules = j
                    continue
                out.append(exchange_set(j, name))
        else:
            mj = model_jobs(name, TEMPLATE_SEED)
            for m, j in mj.items():
                if m == 'rules':
                    rules = j
                    continue
                run = m[len(REG.PREFIX):]
                arm = combo = None
                for a in SE.ARMS:
                    for c in (REG.COMBOS if a == 'specialist' else (None,)):
                        if REG.model(a, c, TEMPLATE_SEED, ENV) == run:
                            arm, combo = a, c
                if arm is None:
                    continue            # a run of an arm without second seeds
                out.append(exchange_set(SE.renamed_job(j, arm, combo, s), name))
    if rules is None:
        rules = model_jobs(name, FIRST[0])['rules']
    out.append(exchange_set(rules, name))
    ids = [j['id'] for j in out]
    if len(ids) != len(set(ids)):
        raise Refused('two jobs share an id')
    return sorted(out, key=lambda j: (-j['cost'], j['id']))


def job_line(job):
    P = _plan()
    return '|'.join(str(job[k]) for k in P.KEYS_JOB) + '|' + ' '.join(job['files'])


def preflight(name):
    """(job, stored folder): the plan's job of the seed-301 policy on the model set, limited to
    PRE_N instances of PRE_REG, and the folder of the stored arrays it must reproduce."""
    s = check_set(name)
    m = REG.PREFIX + REG.model('released', None, PRE_SEED, ENV)
    j = model_jobs(name, PRE_SEED)[m]
    stored = os.path.join(j['out_root'], j['folder'])
    j = dict(j, limit=PRE_N, regimes=' '.join(PRE_REG), pre='-',
             id=f'preflight:{name}:{s["model"]}')
    return j, stored


def trained(model):
    """[] when the run is trained (module docstring), otherwise the reasons."""
    import seed_extension as SE
    run = model[len(REG.PREFIX):] if model.startswith(REG.PREFIX) else model
    for a, c, s in SE.runs():
        if SE.name(a, c, s) == run:
            return SE.training_problems(a, c, s)
    for arm in ('released', 'control', 'specialist'):
        for c in (REG.COMBOS if arm == 'specialist' else (None,)):
            for s in FIRST:
                if REG.model(arm, c, s, ENV) == run:
                    ck, lg = REG.checkpoint(arm, c, s, ENV), REG.train_log(arm, c, s, ENV)
                    why = []
                    if not os.path.isfile(ck):
                        why.append(f'no checkpoint {ck}')
                    if not os.path.exists(lg) or 'done in' not in open(lg, errors='replace').read():
                        why.append(f'{lg} has no completion line')
                    return why
    return [f'{run} is not a run of the policy, the single-rule control or a specialist']


def rows(n, files):
    bad = []
    for f in files:
        if not os.path.exists(f):
            bad.append(f'missing {f}')
        elif f.endswith('.npy') and np.load(f).shape[0] != n:
            bad.append(f'{f}: {np.load(f).shape[0]} rows, want {n}')
    return bad


def check(name, seed_list=None):
    """(complete files, missing or wrong files) of every job of the set."""
    n = set_n(name)
    have, bad = 0, []
    for j in jobs(name, seed_list):
        b = rows(n, j['files'])
        bad += b
        have += len(j['files']) - len(b)
    return have, bad


def estimate(name, seed_list=None):
    js = jobs(name, seed_list)
    nroll = sum(j['n'] * len(j['regimes'].split()) * len(j['methods'].split()) for j in js)
    return dict(jobs=len(js), rollouts=nroll, core_hours=sum(j['cost'] for j in js) / 3600.0,
                longest_hours=max(j['cost'] for j in js) / 3600.0,
                seconds_per_rollout=dict(zip(('policy', 'rule'), speed(name))))


def main(argv):
    os.environ.setdefault('PACT_ENV', ENV)
    os.environ.setdefault('PACT_SEEDS', f'{FIRST[0]}-{FIRST[-1]}')
    if not argv:
        print(__doc__)
        return 2
    cmd, args = argv[0], argv[1:]
    try:
        require_exact()
        if cmd == 'jobs' and len(args) == 1:
            check_data(args[0])
            for j in jobs(args[0]):
                print(job_line(j))
            return 0
        if cmd == 'preflight' and len(args) == 1:
            j, stored = preflight(args[0])
            print(job_line(j))
            print(stored)
            return 0
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
        if cmd == 'check' and len(args) == 1:
            check_data(args[0])
            have, bad = check(args[0])
            print(f'{args[0]}: {have} files complete, {len(bad)} missing or of the wrong length '
                  f'(seeds {",".join(map(str, seeds()))})')
            for x in bad[:10]:
                print(f'   {x}')
            if len(bad) > 10:
                print(f'   ... and {len(bad) - 10} more')
            return 0 if not bad else 1
        if cmd == 'estimate' and len(args) <= 1:
            for nm in (args or list(SETS)):
                e = estimate(nm)
                print(f'{nm}: {e["jobs"]} jobs, {e["rollouts"]} rollouts, {e["core_hours"]:.2f} '
                      f'core-hours (longest job {e["longest_hours"]:.2f} h); seconds per rollout '
                      f'policy {e["seconds_per_rollout"]["policy"]:.3f}, rule '
                      f'{e["seconds_per_rollout"]["rule"]:.3f}; seeds '
                      f'{",".join(map(str, seeds()))}')
            return 0
    except Refused as e:
        print(f'REFUSING: {e}')
        return 2
    print(__doc__)
    return 2


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
