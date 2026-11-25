r"""Job plan, pre-flights and completeness check of scripts/run_eval_newseeds.sh.

The evaluation campaign of the exact non-delay mask, the environment of the paper.
Every job runs on the CPU with one torch thread (OMP and MKL threads 1,
--device cpu --device_id "", --torch_threads 1, --first_mask exact --mask_ties
exact, --tag _xmask), one command per run and folder, so no new array is ever
mixed with an array from another device or thread count.

Groups (EVAL_GROUPS, default 'principal secondary n30 rules'; PUBLIC=1 adds 'public',
LEGACY45=1 adds 'legacy45'):
  principal   the three principal arms (released, control, specialists), seeds
              301-305, mirroring every stored legacy campaign of the seed-303 runs:
              10x25 (the stored 10x25, 10x25_blindref and 10x25_cpuref sets in one
              folder, arm_registry.FOLDER_ALIAS), 10x25_oodsetup, 10x25_oodwindow,
              20x25, 30x25 and the four 20-instance scale cells.
  secondary   the further exact-mask arms of arm_registry.SECONDARY_ARMS, seeds
              301-303, each on the folders and combinations of its stored legacy
              counterpart (SECONDARY_EVAL), including the token interventions the
              stored joint-v1 seed-301 run has in 10x25. The arm with the
              active-schedule mask (xmask-active) is evaluated under that mask: its
              training configuration records mask_mode active, and eval_family.py
              applies the recorded mask_mode to the policy's rollouts.
  n30         the released policy, seeds 301-305, on the 30-instance production
              sets 50x25n30 and 80x25n30, exporting every schedule
              (or_solution/FAMILY/pactwarm_cache_xmask_s<seed>_<set>/<R>/<inst>.npz)
              for the policy-seeded solver arms.
  rules       FIFO, MOR, SPT and MWKR under the exact mask in every exact folder,
              restricted to the pairs the mask admits (--rule_set masked; the
              public group's rules too). Legacy rule jobs keep the classical rules.
  public      the public-instance family (arm_registry.PUB_*), seeds 301-305, and
              the rules, on the six public test sets (--base record, best-known
              table per set), into test_results/PUBFAMILY/<set>_xmask.
  pubsens     the same jobs on the sensitivity sets of the public family
              (arm_registry.PUB_SENS_SETS, e.g. lag scale 0.5 on SD2 10x5), the
              stored policies only; off by default (EVAL_GROUPS or PUBSENS=1).
  pubeq       the equal-experience arm of the public family (arm_registry.PUB_EQ_*,
              pubx-eq-s<seed>, seeds 301-305) on the six public test sets and the
              sensitivity sets, all eight combinations: each job is the pubx-v1 job
              of the same seed and set with the run name changed, so it shares the
              options, the folder and the pre-flight of the public group; no rule job
              (the stored rule arrays serve). Off by default (EVAL_GROUPS or PUBEQ=1).
  pubact      the active-rule arm of the public family (arm_registry.PUB_ACT_*,
              pubx-act-s<seed>, seeds 301-303) on the sets of PUB_ACT_SETS (SD2 10x5,
              the headline set, and SD2 20x10), all eight combinations: each job is
              the pubx-v1 job of the same seed and set with the run name changed, as in
              group pubeq. eval_family.py restores mask_mode from the run's stored
              configuration (ARCH_KEYS, load_model) and applies it to the policy's
              rollouts only, so the arm is evaluated under the rule it was trained
              with. No rule job. Off by default (EVAL_GROUPS or PUBACT=1).
  pubactfam  the public family under the active mask (arm_registry.pub_rule_model: the
              policy pubx-act-s<seed>, the single-rule control pubx-ctlact-s<seed>, the
              specialists pubx-soloact-<R>-s<seed>, seeds PUB_ACTFAM_SEEDS) on the six public
              test sets: one job per job of its non-delay counterpart in group public, the
              run name changed and nothing else; the jobs of pubx-act-s301..303 on
              PUB_ACT_SETS are left out, they are the jobs of group pubact. No rule job: the
              rules under the active mask are rolled out by
              scripts/p109_public_active_rules.py. eval_family.py restores mask_mode from each
              run's stored configuration. Off by default (EVAL_GROUPS or PUBACTFAM=1).
  pubactsens the same runs on the sensitivity sets (group pubsens renamed), pubx-act-s301..303
              included. Off by default (EVAL_GROUPS or PUBACTSENS=1).
  pubacteq   the equal-experience recipe under the active mask (pubx-eqact-s<seed>) on the
              sets of group pubeq: each job is the pubx-eq job of the same seed and set with
              the run name changed. Off by default (EVAL_GROUPS or PUBACTEQ=1).
  legacy45   legacy seeds 304-305 in every stored legacy campaign, each with the
              device and threads of its stored seed-303 files, and the legacy
              30-instance sets for seeds 301-305 (GPU like the scale cells).
  hgnn        the HGNN port of Song et al. trained under the exact mask by
              scripts/run_hgnn_exact_queue.sh: the joint runs with the PACT reward
              (hgnn_reward certified), seeds 301-305, and with the published reward,
              seeds 301-303, on the eight combinations, and the eight specialists
              (seed 301, each on its own combination) once they are trained, all on the
              100 test instances at 10x25, by backbones/hgnn/eval_family_hgnn.py on the
              CPU with one thread and --tag _xmask, into
              10x25_xmask/Result_<run>_<R>.npy (the layout of the stored earlier-mask
              evaluations in 10x25). The learning rate in the run names is HGNN_LR,
              else the selection of scripts/hgnn_select_lr.py. Off by default
              (EVAL_GROUPS or HGNN=1).
  cross       every exact-mask specialist (seeds EXACT_SEEDS) under each of the seven
              combinations it was not trained for, on the 100 test instances at 10x25,
              one job per specialist run, into the folder of its own
              arm_registry.CROSS_FOLDER (10x25_cross_xmask, eval_family.py --tag
              _cross_xmask). The specialist on its own combination and the policy come
              from the principal group. The check requires the 19 off-combination
              arrays per seed that the principal 10x25 campaign also writes to be equal
              to them. Off by default (EVAL_GROUPS or CROSS=1).

Subcommands:
  jobs        one line per job, '|'-separated (the queue's input)
  preflight   the pre-flight evaluations, one line each
  compare     compare <folder a> <folder b> <tags,> "<regimes>" <n> (makespans)
  check       completeness, derived from the stored legacy files (seed 303 of the
              principal arms; seed 303 of each secondary arm's counterpart, seed 301
              for the token interventions), not from the job list
  counts      jobs, files, rollouts and core-hours per group and campaign

Environment variables: EVAL_GROUPS, PUBLIC, PUBSENS, PUBEQ, PUBACT, PUBACTFAM, PUBACTSENS, PUBACTEQ,
LEGACY45, CROSS, EXACT_SEEDS (default 301-305),
LEGACY_SEEDS (304 305), RESULTS (test_results/FAMILY), PUB_RESULTS
(test_results/PUBFAMILY), STORED (test_results/FAMILY, read only), CACHE_ROOT
('' = repository root; tests only); for the hgnn group HGNN (1 adds it), HGNN_LR
(1e-4, 2e-4 or 4e-4; default the selection) and HGNN_SOLO (auto: the specialists once
a checkpoint or training log of one exists; 1 always; 0 never).
"""
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

import arm_registry as REG  # noqa: E402
import hgnn_select_lr as HLR  # noqa: E402

R8 = list(REG.COMBOS)
RULES = ('FIFO', 'MOR', 'SPT', 'MWKR')
PPVC = 'data/PPVC/{}+ppvc-mixed'
GROUPS = os.environ.get('EVAL_GROUPS', 'principal secondary n30 rules').split()
if os.environ.get('PUBLIC') == '1' and 'public' not in GROUPS:
    GROUPS.append('public')
if os.environ.get('LEGACY45') == '1' and 'legacy45' not in GROUPS:
    GROUPS.append('legacy45')
if os.environ.get('PUBSENS') == '1' and 'pubsens' not in GROUPS:
    GROUPS.append('pubsens')
if os.environ.get('PUBEQ') == '1' and 'pubeq' not in GROUPS:
    GROUPS.append('pubeq')
if os.environ.get('PUBACT') == '1' and 'pubact' not in GROUPS:
    GROUPS.append('pubact')
for _g, _v in (('pubactfam', 'PUBACTFAM'), ('pubactsens', 'PUBACTSENS'), ('pubacteq', 'PUBACTEQ')):
    if os.environ.get(_v) == '1' and _g not in GROUPS:
        GROUPS.append(_g)
if os.environ.get('HGNN') == '1' and 'hgnn' not in GROUPS:
    GROUPS.append('hgnn')
if os.environ.get('CROSS') == '1' and 'cross' not in GROUPS:
    GROUPS.append('cross')
EXACT_SEEDS = [int(s) for s in os.environ.get('EXACT_SEEDS', '301 302 303 304 305').split()]
LEGACY_SEEDS = [int(s) for s in os.environ.get('LEGACY_SEEDS', '304 305').split()]
N30_SEEDS = (301, 302, 303, 304, 305)
RESULTS = os.environ.get('RESULTS', 'test_results/FAMILY')
PUB_RESULTS = os.environ.get('PUB_RESULTS', REG.PUB_RESULTS)
STORED = os.environ.get('STORED', 'test_results/FAMILY')
CACHE_ROOT = os.environ.get('CACHE_ROOT', '')
REF_SEED = 303
XS = REG.FOLDER_SUFFIX['exact']

# Seconds per rollout on one core with one thread (OMP, MKL and torch), exact mask,
# measured on core 10 (otherwise idle) on 2026-11-11: the first three instances of
# N and of LSW per size, the stored seed-303 released policy (public sets:
# pubx-v1-s301) under the exact options, read from the time column of the arrays,
# which agrees with the wall-clock of the job within the model load of about 5 s.
SPEED = {  # size: (policy rollout, one rule rollout)
    '10x25': (0.741, 0.206), '20x25': (3.206, 1.113), '30x25': (7.192, 1.884),
    '50x15': (15.209, 3.225), '50x25': (16.285, 4.191),
    '80x15': (30.995, 6.345), '80x25': (33.262, 8.917),
    'pub_SD2_test_10x5+mix': (0.275, 0.072), 'pub_SD2_test_20x10+mix': (1.311, 0.337),
    'pub_Brandimarte': (0.518, 0.136), 'pub_Hurink_edata': (1.172, 0.318),
    'pub_Hurink_rdata': (1.159, 0.314), 'pub_Hurink_vdata': (1.173, 0.326),
}
# a sensitivity set of the public family is not measured: its test set's speed
for _st, (_base, _bk, _change) in REG.PUB_SENS_SETS.items():
    SPEED.setdefault(_st, SPEED[_base])
FILM_FACTOR = 0.779 / 0.741   # descriptor-input variant (joint-v1-s303) against the released one, 10x25
JOB_OVERHEAD = 5.0            # seconds per job: imports and model load


def size_key(data):
    b = os.path.basename(data.rstrip('/'))
    if b in SPEED:
        return b
    m = re.match(r'(\d+x\d+)', b)
    return m.group(1)


def instance_names(data, limit=0):
    names = sorted(os.path.basename(p)[:-len('.desc.json')]
                   for p in glob.glob(f'{data}/desc/N/*.desc.json'))
    return names[:limit] if limit else names


def cache_dir(seed, size, environment):
    d = REG.schedule_cache(seed, size, environment)
    return os.path.join(CACHE_ROOT, d) if CACHE_ROOT else d


def mode_tag(token_mode, blind):
    """eval_family.py's mode tag of a token intervention."""
    t = '' if token_mode == 'true' else '+' + token_mode.replace(':', '-')
    ch = sorted(b for b in blind.split(',') if b) if blind not in ('', '-') else []
    return t + ('+chblind-' + '-'.join(ch) if ch else '')


def parse_mode(tag):
    """'+withhold+chblind-lag-setup' -> ('withhold', 'lag,setup')."""
    token, blind = 'true', ''
    for part in tag.strip('+').split('+'):
        if part.startswith('chblind-'):
            blind = ','.join(part[len('chblind-'):].split('-'))
        elif part.startswith('counterfactual-'):
            token = 'counterfactual:' + part[len('counterfactual-'):]
        else:
            token = part
    if mode_tag(token, blind) != tag:
        raise SystemExit(f'cannot rebuild the eval_family.py options of mode {tag!r}')
    return token, blind


# ----------------------------------------------------------------- exact campaigns
# folder (stored legacy name, without the exact suffix) -> data, base
FACTORY = {f: (f'data/FAMILY/{f}', PPVC.format(f[:5])) for f in
           ('10x25', '10x25_oodsetup', '10x25_oodwindow', '20x25', '30x25',
            '50x15', '50x25', '80x15', '80x25')}
N30 = {'50x25n30': ('data/FAMILY/50x25n30', PPVC.format('50x25') + '-n30'),
       '80x25n30': ('data/FAMILY/80x25n30', PPVC.format('80x25') + '-n30')}
SCALE20 = ('50x15', '50x25', '80x15', '80x25')


def principal_runs(folder):
    """(arm, combo, regimes) of the principal arms in one exact folder: the union of
    the stored legacy seed-303 sets of every legacy folder that maps to it."""
    out = [('released', None, R8)]
    if folder == '10x25':
        out.append(('control', None, R8))
        # 10x25: own and one-more-rule combinations; 10x25_blindref adds LSW on all 8
        out += [('specialist', r, R8 if r == 'LSW' else list(REG.SPEC_EVAL_10x25[r]))
                for r in R8]
    elif folder in ('10x25_oodsetup', '10x25_oodwindow', '20x25', '30x25'):
        out += [('specialist', r, [r]) for r in R8]
    return out


# Further arms: (folder, regimes) of the stored counterpart's seed-303 files, with the
# stored legacy folders 10x25 and 10x25_cpuref both mapped to 10x25. xmask-sparse has
# no stored counterpart and takes the set of the other recipe variants (10x25, all 8).
SECONDARY_EVAL = {
    'xmask-eq': [('10x25', R8), ('20x25', R8), ('30x25', R8)],
    'xmask-wthird': [('10x25', ['W'])],
    'xmask-sparse': [('10x25', R8)],
    'xmask-dzero': [('10x25', R8)],
    'xmask-dmax': [('10x25', R8)],
    'xmask-film': [(f, R8) for f in ('10x25', '10x25_oodsetup', '10x25_oodwindow', '20x25',
                                     '30x25', '50x15', '50x25', '80x15', '80x25')],
    'xmask-dr': [('10x25', R8), ('10x25_oodsetup', R8), ('10x25_oodwindow', R8)],
    'xmask-all8': [('10x25', R8)],
    'xmask-pair': [('10x25', R8)],
    # the active-schedule mask: eval_family.py restores mask_mode from the run's
    # training configuration (ARCH_KEYS) for the policy, after the command line is
    # parsed, so the job is the job of every secondary arm (--first_mask exact
    # --mask_ties exact) and the rollouts use the mask the run recorded, active
    'xmask-active': [('10x25', R8)],
}
E3_ARM, E3_SEED, E3_COUNTERPART = 'xmask-film', 301, 'joint-v1'   # stored seed 301 only


def e3_modes():
    """Token interventions of the stored joint-v1 seed-301 run in 10x25:
    {mode tag: regimes}."""
    rx = re.compile(r'^Result_greedy\+' + re.escape(REG.PREFIX + E3_COUNTERPART)
                    + r'(\+[^_]+)_([A-Z]+)\.npy$')
    out = {}
    for f in sorted(os.listdir(f'{STORED}/10x25')):
        m = rx.match(f)
        if m:
            out.setdefault(m.group(1), []).append(m.group(2))
    return {k: [r for r in R8 if r in v] for k, v in out.items()}


def job(**k):
    k.setdefault('token_mode', 'true')
    k.setdefault('blind', '-')
    k.setdefault('best_known', '-')
    k.setdefault('sched', '-')
    k.setdefault('extra', '-')
    k.setdefault('limit', 0)
    k.setdefault('rule_set', 'classical')
    return k


def exact_policy_job(group, campaign, model, ckpt, log, folder_key, data, base, regs,
                     seed=None, token_mode='true', blind='-', sched_set=None, film=False):
    folder = folder_key + XS
    names = instance_names(data)
    mt = mode_tag(token_mode, blind)
    files = [f'{RESULTS}/{folder}/Result_greedy+{model}{mt}_{r}.npy' for r in regs]
    files.append(f'{RESULTS}/{folder}/summary_{model}{mt}{XS}.md')
    sched = '-'
    if sched_set:
        sched = cache_dir(seed, sched_set, 'exact')
        files += [f'{sched}/{r}/{n}.npz' for r in regs for n in names]
    sp = SPEED[size_key(data)][0] * (FILM_FACTOR if film else 1.0)
    return job(id=f'exact:{campaign}:{model.split("+")[-1]}{mt}', group=group,
               campaign=campaign, env='exact', device='cpu', omp=1, tt=1, model=model,
               methods='greedy', data=data, base=base, tag=XS, regimes=' '.join(regs),
               token_mode=token_mode, blind=blind, sched=sched, out_root=RESULTS,
               folder=folder, n=len(names), ckpt=ckpt, log=log,
               cost=sp * len(names) * len(regs) + JOB_OVERHEAD, files=files)


def rules_job(group, campaign, folder_key, data, base, out_root, best_known='-'):
    folder = os.path.basename(data) + XS
    names = instance_names(data)
    files = [f'{out_root}/{folder}/Result_{q}_{r}.npy' for q in RULES for r in R8]
    files.append(f'{out_root}/{folder}/summary_rules{XS}.md')
    sp = SPEED[size_key(data)][1]
    # exact environment: the four rules act on the pairs the non-delay mask admits
    # (eval_family.py --rule_set masked, common_utils.masked_rule_action)
    return job(id=f'exact:{campaign}:rules', group=group, campaign=campaign, env='exact',
               device='cpu', omp=1, tt=1, model='rules', methods=' '.join(RULES),
               rule_set='masked',
               data=data, base=base, tag=XS, regimes=' '.join(R8), best_known=best_known,
               out_root=out_root, folder=folder, n=len(names), ckpt='-', log='-',
               cost=sp * len(names) * 8 * len(RULES) + JOB_OVERHEAD, files=files)


def exact_jobs():
    out = []
    if 'principal' in GROUPS:
        for f, (data, base) in FACTORY.items():
            camp = ('scale20_' if f in SCALE20 else '') + f
            for arm, combo, regs in principal_runs(f):
                for s in EXACT_SEEDS:
                    m = REG.PREFIX + REG.model(arm, combo, s, 'exact')
                    out.append(exact_policy_job(
                        'principal', camp, m, REG.checkpoint(arm, combo, s, 'exact'),
                        REG.train_log(arm, combo, s, 'exact'), f, data, base, regs))
    if 'secondary' in GROUPS:
        for stem, sets in SECONDARY_EVAL.items():
            for f, regs in sets:
                data, base = FACTORY[f]
                for s in REG.SECONDARY_SEEDS:
                    ck, lg = REG.secondary_paths(stem, s)
                    out.append(exact_policy_job(
                        'secondary', f'{stem}_{f}', REG.PREFIX + REG.secondary_model(stem, s),
                        ck, lg, f, data, base, regs, film=(stem == 'xmask-film')))
        ck, lg = REG.secondary_paths(E3_ARM, E3_SEED)
        data, base = FACTORY['10x25']
        for tagm, regs in e3_modes().items():
            token, blind = parse_mode(tagm)
            out.append(exact_policy_job(
                'secondary', f'{E3_ARM}_tokens', REG.PREFIX + REG.secondary_model(E3_ARM, E3_SEED),
                ck, lg, '10x25', data, base, regs, token_mode=token, blind=blind or '-',
                film=True))
    if 'n30' in GROUPS:
        for f, (data, base) in N30.items():
            for s in N30_SEEDS:
                if s not in EXACT_SEEDS:
                    continue
                m = REG.PREFIX + REG.model('released', None, s, 'exact')
                out.append(exact_policy_job(
                    'n30', f, m, REG.checkpoint('released', None, s, 'exact'),
                    REG.train_log('released', None, s, 'exact'), f, data, base, R8,
                    seed=s, sched_set=f))
    if 'rules' in GROUPS:
        for f, (data, base) in list(FACTORY.items()) + list(N30.items()):
            out.append(rules_job('rules', f, f, data, base, RESULTS))
    if 'public' in GROUPS:
        for st, bk in REG.PUB_SETS.items():
            out += public_set_jobs('public', st, bk)
    if 'pubsens' in GROUPS:
        for st, (_, bk, _) in REG.PUB_SENS_SETS.items():
            out += public_set_jobs('pubsens', st, bk)
    if 'pubeq' in GROUPS:
        for st, bk in pub_eq_sets():
            out += public_eq_jobs(st, bk)
    if 'pubact' in GROUPS:
        for st, bk in pub_act_sets():
            out += public_act_jobs(st, bk)
    if 'pubactfam' in GROUPS:
        for st, bk in REG.PUB_SETS.items():
            out += public_actfam_jobs('pubactfam', st, bk)
    if 'pubactsens' in GROUPS:
        for st, (_, bk, _) in REG.PUB_SENS_SETS.items():
            out += public_actfam_jobs('pubactsens', st, bk)
    if 'pubacteq' in GROUPS:
        for st, bk in pub_eq_sets():
            out += public_acteq_jobs(st, bk)
    if 'hgnn' in GROUPS:
        out += hgnn_jobs()
    if 'cross' in GROUPS:
        out += cross_jobs()
    return out


# ----------------------------------------------------------------- off-combination (group cross)
# One job per specialist run with its seven other combinations: eval_family.py writes one
# summary per run and folder, so one job per combination would overwrite that summary.
CROSS_DATA = '10x25'
CROSS_TAG = REG.EXACT_TAG[REG.CROSS_FOLDER]


def cross_combos(r):
    """The combinations specialist r is evaluated on in the group cross: all but its own."""
    return [c for c in R8 if c != r]


def cross_jobs():
    data, base = FACTORY[CROSS_DATA]
    folder = os.path.basename(data) + CROSS_TAG
    if folder != REG.CROSS_FOLDER + XS:
        raise SystemExit(f'the cross folder {folder} differs from the registry '
                         f'({REG.CROSS_FOLDER + XS})')
    names = instance_names(data)
    out = []
    for r in R8:
        regs = cross_combos(r)
        for s in EXACT_SEEDS:
            name = REG.model('specialist', r, s, 'exact')
            m = REG.PREFIX + name
            files = [f'{RESULTS}/{folder}/Result_greedy+{m}_{c}.npy' for c in regs]
            files.append(f'{RESULTS}/{folder}/summary_{m}{CROSS_TAG}.md')
            out.append(job(id=f'exact:{REG.CROSS_FOLDER}:{name}', group='cross',
                           campaign=REG.CROSS_FOLDER, env='exact', device='cpu', omp=1, tt=1,
                           model=m, methods='greedy', data=data, base=base, tag=CROSS_TAG,
                           regimes=' '.join(regs), out_root=RESULTS, folder=folder,
                           n=len(names), ckpt=REG.checkpoint('specialist', r, s, 'exact'),
                           log=REG.train_log('specialist', r, s, 'exact'),
                           cost=SPEED[size_key(data)][0] * len(names) * len(regs) + JOB_OVERHEAD,
                           files=files))
    return out


PUB_RUNS = [('released', None, R8), ('control', None, R8)] + [('specialist', r, [r]) for r in R8]


def pub_eq_sets():
    """(set, best-known table) of the equal-experience arm: the six test sets, then
    the sensitivity sets, the sets on which the stored pubx-v1 runs were evaluated."""
    return list(REG.PUB_SETS.items()) + [(st, bk) for st, (_, bk, _) in REG.PUB_SENS_SETS.items()]


def public_eq_jobs(st, bk):
    """One job per seed: the released run's job of the same seed on the same set
    (public or pubsens group), with the run name and the group changed."""
    group = 'public' if st in REG.PUB_SETS else 'pubsens'
    released = {j['model']: j for j in public_set_jobs(group, st, bk) if j['model'] != 'rules'}
    out = []
    for s in REG.PUB_SEEDS:
        rel = released[REG.PUB_PREFIX + REG.pub_model('released', None, s)]
        name = REG.pub_eq_model(s)
        m = REG.PUB_PREFIX + name
        ck, lg = REG.pub_eq_paths(s)[:2]
        files = [f.replace(rel['model'], m) for f in rel['files']]
        assert all(m in f for f in files) and len(set(files)) == len(files)
        out.append(dict(rel, id=f'exact:public_{st}:{name}', group='pubeq', model=m,
                        ckpt=ck, log=lg, files=files))
    return out


# Test sets of the active-rule arm: the headline set, which
# scripts/p108_public_active_arm.py reads, and the size-transfer set SD2 20x10.
PUB_ACT_SETS = ('pub_SD2_test_10x5+mix', 'pub_SD2_test_20x10+mix')


def pub_act_sets():
    """(set, best-known table) of the active-rule arm, PUB_ACT_SETS in order."""
    return [(st, REG.PUB_SETS[st]) for st in PUB_ACT_SETS]


def public_act_jobs(st, bk):
    """One job per seed of the active-rule arm: the released run's job of the same seed
    on the same test set (public group), with the run name and the group changed. The
    options are the released run's; the rule comes from the run's stored configuration,
    which eval_family.py restores (mask_mode in ARCH_KEYS)."""
    released = {j['model']: j for j in public_set_jobs('public', st, bk) if j['model'] != 'rules'}
    out = []
    for s in REG.PUB_ACT_SEEDS:
        rel = released[REG.PUB_PREFIX + REG.pub_model('released', None, s)]
        name = REG.pub_act_model(s)
        m = REG.PUB_PREFIX + name
        ck, lg = REG.pub_act_paths(s)[:2]
        files = [f.replace(rel['model'], m) for f in rel['files']]
        assert all(m in f for f in files) and len(set(files)) == len(files)
        out.append(dict(rel, id=f'exact:public_{st}:{name}', group='pubact', model=m,
                        ckpt=ck, log=lg, files=files))
    return out


# The public family under the active mask (groups pubactfam, pubactsens, pubacteq): the
# counterpart group of each, and the runs of group pubact that pubactfam leaves out.
PUB_ACTFAM_COUNTERPART = {'pubactfam': 'public', 'pubactsens': 'pubsens', 'pubacteq': 'pubeq'}


def pub_actfam_runs(group, st):
    """(arm, combo, seed, regimes) of the runs under the active mask that group evaluates on
    set st: every run of PUB_RUNS on the seeds PUB_ACTFAM_SEEDS, without the runs of group
    pubact (pubx-act-s<seed>, seeds PUB_ACT_SEEDS, on the sets PUB_ACT_SETS) in pubactfam."""
    out = []
    for arm, combo, regs in PUB_RUNS:
        for s in REG.PUB_ACTFAM_SEEDS:
            if group == 'pubactfam' and arm == 'released' and s in REG.PUB_ACT_SEEDS and \
                    st in PUB_ACT_SETS:
                continue
            out.append((arm, combo, s, regs))
    return out


def renamed_job(job_, group, name, ckpt, log):
    """The job of a non-delay run with the run name, the checkpoint, the log and the group
    changed and nothing else."""
    m = REG.PUB_PREFIX + name
    files = [f.replace(job_['model'], m) for f in job_['files']]
    assert all(m in f for f in files) and len(set(files)) == len(files)
    assert job_['id'].endswith(':' + job_['model'][len(REG.PUB_PREFIX):])
    return dict(job_, id=job_['id'][:-len(job_['model']) + len(REG.PUB_PREFIX)] + name,
                group=group, model=m, ckpt=ckpt, log=log, files=files)


def public_actfam_jobs(group, st, bk):
    """One job per run of pub_actfam_runs: the job of its non-delay counterpart
    (arm_registry.pub_rule_counterpart) in the counterpart group on the same set, renamed."""
    base = {j['model']: j for j in public_set_jobs(PUB_ACTFAM_COUNTERPART[group], st, bk)
            if j['model'] != 'rules'}
    out = []
    for arm, combo, s, _ in pub_actfam_runs(group, st):
        k = base[REG.PUB_PREFIX + REG.pub_rule_counterpart(arm, combo, s)]
        ck, lg = REG.pub_rule_paths(arm, combo, s)[:2]
        out.append(renamed_job(k, group, REG.pub_rule_model(arm, combo, s), ck, lg))
    return out


def public_acteq_jobs(st, bk):
    """One job per seed of the equal-experience recipe under the active mask: the pubx-eq
    job of the same seed and set (group pubeq), renamed."""
    base = {j['model']: j for j in public_eq_jobs(st, bk)}
    out = []
    for s in REG.PUB_SEEDS:
        k = base[REG.PUB_PREFIX + REG.pub_rule_counterpart('eq', None, s)]
        ck, lg = REG.pub_rule_paths('eq', None, s)[:2]
        out.append(renamed_job(k, 'pubacteq', REG.pub_rule_model('eq', None, s), ck, lg))
    return out


def public_set_jobs(group, st, bk):
    """Every public-family run and the rules on one public set (test or sensitivity)."""
    out = []
    data = f'data/PUBLIC/family/{st}'
    names = instance_names(data)
    for arm, combo, regs in PUB_RUNS:
        for s in REG.PUB_SEEDS:
            name = REG.pub_model(arm, combo, s)
            m = REG.PUB_PREFIX + name
            folder = st + XS
            files = [f'{PUB_RESULTS}/{folder}/Result_greedy+{m}_{r}.npy' for r in regs]
            files.append(f'{PUB_RESULTS}/{folder}/summary_{m}{XS}.md')
            ck, lg = REG.pub_paths(arm, combo, s)
            out.append(job(id=f'exact:public_{st}:{name}', group=group,
                           campaign=f'public_{st}', env='exact', device='cpu',
                           omp=1, tt=1, model=m, methods='greedy', data=data,
                           base='record', tag=XS, regimes=' '.join(regs),
                           best_known=bk, out_root=PUB_RESULTS, folder=folder,
                           n=len(names), ckpt=ck, log=lg,
                           cost=SPEED[st][0] * len(names) * len(regs) + JOB_OVERHEAD,
                           files=files))
    out.append(rules_job(group, f'public_{st}', st, data, 'record', PUB_RESULTS, bk))
    return out


# ----------------------------------------------------------------- HGNN port (group hgnn)
# The runs of scripts/run_hgnn_exact_queue.sh at the training size, evaluated by
# backbones/hgnn/eval_family_hgnn.py (greedy, batch of one, the reward and mask that the
# run's configuration records): each joint run on the eight combinations, each
# specialist on its own, as the stored earlier-mask evaluations in test_results/FAMILY/10x25.
HGNN_FOLDER = '10x25'
HGNN_JOINT_SEEDS = {'certified': (301, 302, 303, 304, 305),   # PHASE=tune (301), panel, panel2
                    'published': (301, 302, 303)}             # PHASE=panel
HGNN_SOLO_SEEDS = (301,)                                      # PHASE=solo
HGNN_CROSS = {'certified': 'hgnn_cross', 'published': 'hgnn_cross_pub'}   # pre-flight per reward
HGNN_CROSS_SEED, HGNN_CROSS_N, HGNN_CROSS_REG = 301, 2, 'S LSW'
# Seconds per rollout at 10x25 on one core with one thread (OMP, MKL and torch): 1.34 to
# 1.47 s over the 10 rollouts of the two-instance reproductions of 2026-11-12 (stored
# runs hgnn-certified-joint-s301, hgnn-published-joint-s301, hgnn-certified-LSW-s301; S
# and LSW), read from the time column, on a core shared with other work (CPU share
# 0.97). A planning estimate.
HGNN_SPEED = 1.40


def _hgnn_layout(path_of):
    """Path template of the registry (checkpoint or training log) with the run name as {m}."""
    m = REG.model('released', None, REF_SEED, 'exact')
    p = path_of('released', None, REF_SEED, 'exact')
    if p.count(m) != 1:
        raise SystemExit(f'cannot derive a path layout from {p!r}')
    return p.replace(m, '{m}')


HGNN_CKPT = _hgnn_layout(REG.checkpoint)
HGNN_LOG = _hgnn_layout(REG.train_log)


def hgnn_name(reward, part, seed, rate):
    """Run name of an exact-mask HGNN run, hgnn-<reward>-<joint|R>-s<seed>-xmask[-lr<rate>]
    (no rate tag for 2e-4), as scripts/run_hgnn_exact_queue.sh names it."""
    return f'hgnn-{reward}-{part}-s{int(seed)}-xmask' + HLR.run_name(rate)[len(HLR.STEM):]


def hgnn_rate():
    """Learning rate in the run names: HGNN_LR when set, otherwise the selection of
    scripts/hgnn_select_lr.py; exits when neither is available."""
    lr = os.environ.get('HGNN_LR', '')
    if lr:
        if lr not in HLR.RATES:
            raise SystemExit(f'HGNN_LR={lr} is not one of {", ".join(HLR.RATES)}')
        return lr
    try:
        return HLR.select()[0]
    except SystemExit as e:
        raise SystemExit(f'the hgnn group needs the learning rate of the HGNN runs: {e} '
                         f'(or HGNN_LR=<rate>)')


def hgnn_solo_on(rate, root=''):
    """Whether the group evaluates the specialists: HGNN_SOLO=1 always, 0 never, auto (the
    default) once the checkpoint or the training log of one of them exists."""
    v = os.environ.get('HGNN_SOLO', 'auto') or 'auto'
    if v not in ('auto', '0', '1'):
        raise SystemExit(f'HGNN_SOLO must be auto, 0 or 1, got {v!r}')
    if v != 'auto':
        return v == '1'
    names = [hgnn_name('certified', r, s, rate) for r in R8 for s in HGNN_SOLO_SEEDS]
    return any(os.path.exists(os.path.join(root, t.format(m=n)))
               for n in names for t in (HGNN_CKPT, HGNN_LOG))


def hgnn_runs(rate):
    """(reward, part, seed, combinations) of every run the group evaluates."""
    out = [(rw, 'joint', s, R8) for rw, seeds in HGNN_JOINT_SEEDS.items() for s in seeds]
    if hgnn_solo_on(rate):
        out += [('certified', r, s, [r]) for r in R8 for s in HGNN_SOLO_SEEDS]
    return out


def hgnn_jobs():
    """One job per run: all test instances of its combinations at HGNN_FOLDER, CPU, one
    thread, exact campaign folder; no summary file (the evaluator writes none)."""
    rate = hgnn_rate()
    data, base = FACTORY[HGNN_FOLDER]
    meta = os.path.join(data, 'dataset_meta.json')
    if os.path.exists(meta):           # the evaluator reads the base folder from here
        with open(meta) as f:
            if json.load(f)['base_dir'] != base:
                raise SystemExit(f'{meta} names another base folder than {base}')
    names = instance_names(data)
    folder = HGNN_FOLDER + XS
    out = []
    for reward, part, s, regs in hgnn_runs(rate):
        name = hgnn_name(reward, part, s, rate)
        out.append(job(id=f'exact:hgnn_{HGNN_FOLDER}:{name}', group='hgnn',
                       campaign=f'hgnn_{HGNN_FOLDER}', env='exact', device='cpu', omp=1, tt=1,
                       model=REG.PREFIX + name, methods='hgnn', data=data, base=base, tag=XS,
                       regimes=' '.join(regs), out_root=RESULTS, folder=folder, n=len(names),
                       ckpt=HGNN_CKPT.format(m=name), log=HGNN_LOG.format(m=name),
                       cost=HGNN_SPEED * len(names) * len(regs) + JOB_OVERHEAD,
                       files=[f'{RESULTS}/{folder}/Result_{name}_{r}.npy' for r in regs]))
    return out


def hgnn_check(need):
    """The exact counterpart of every stored earlier-mask HGNN evaluation of seed 301 at
    HGNN_FOLDER (a joint run: each of its combinations; a specialist: its own), for every
    seed of HGNN_JOINT_SEEDS or HGNN_SOLO_SEEDS; the specialists only when the group
    evaluates them."""
    rate = hgnn_rate()
    solo = hgnn_solo_on(rate)
    rx = re.compile(r'^Result_hgnn-(certified|published)-(joint|[A-Z]+)-s301_([A-Z]+)\.npy$')
    d = f'{STORED}/{HGNN_FOLDER}'
    for f in sorted(os.listdir(d)):
        m = rx.match(f)
        if not m:
            continue
        reward, part, r = m.groups()
        if part != 'joint' and not solo:
            continue
        rows = np.load(f'{d}/{f}').shape[0]
        for s in (HGNN_JOINT_SEEDS[reward] if part == 'joint' else HGNN_SOLO_SEEDS):
            need(f'{RESULTS}/{HGNN_FOLDER}{XS}/Result_{hgnn_name(reward, part, s, rate)}_{r}.npy',
                 rows, ('hgnn', HGNN_FOLDER + XS))


# ----------------------------------------------------------------- legacy 304-305
LEGACY = []


def lcamp(name, data, base, tag, limit, device, omp, tt, extra, arms, pre, n30=False):
    LEGACY.append(dict(name=name, data=data, base=base, tag=tag, limit=limit, device=device,
                       omp=omp, tt=tt, extra=extra, arms=arms, pre=pre, n30=n30))


lcamp('main10', 'data/FAMILY/10x25', PPVC.format('10x25'), '', 0, 'gpu', 4, 0, '', 'main', 'small')
lcamp('blindref', 'data/FAMILY/10x25', PPVC.format('10x25'), '_blindref', 0, 'gpu', 4, 0, '', 'blindref', 'small')
lcamp('cpuref', 'data/FAMILY/10x25', PPVC.format('10x25'), '_cpuref', 0, 'cpu', 4, 4, '', 'rel8+own', 'small')
lcamp('firstmask', 'data/FAMILY/10x25', PPVC.format('10x25'), '_firstmask', 0, 'cpu', 4, 4, 'PACT_FIRST_MASK=1', 'rel8+own', 'small')
lcamp('firstmask_off', 'data/FAMILY/10x25', PPVC.format('10x25'), '_firstmask_off', 0, 'cpu', 4, 4, '', 'rel8+own', 'small')
lcamp('masktol', 'data/FAMILY/10x25', PPVC.format('10x25'), '_masktol', 0, 'gpu', 4, 0, 'PACT_MASK_TOL=1', 'rel8+own', 'small')
lcamp('oodsetup', 'data/FAMILY/10x25_oodsetup', PPVC.format('10x25'), '', 0, 'cpu', 4, 4, '', 'rel8+own', 'small')
lcamp('oodwindow', 'data/FAMILY/10x25_oodwindow', PPVC.format('10x25'), '', 0, 'cpu', 4, 4, '', 'rel8+own', 'small')
lcamp('t20', 'data/FAMILY/20x25', PPVC.format('20x25'), '', 0, 'gpu', 4, 0, '', 'rel8+own', 'small')
lcamp('t30', 'data/FAMILY/30x25', PPVC.format('30x25'), '', 0, 'gpu', 4, 0, '', 'rel8+own', 'small')
for _c in SCALE20:
    lcamp(f'scout{_c}', f'data/FAMILY/{_c}', PPVC.format(_c), '', 0, 'gpu', 1, 1, '', 'rel8', 'large')
for _s in ('50x25', '80x25'):
    lcamp(f'pactwarm{_s}', f'data/FAMILY/{_s}', PPVC.format(_s), '_pactwarm', 10, 'gpu', 1, 1, '', 'rel8+sched', 'large')
lcamp('masktol80x25', 'data/FAMILY/80x25', PPVC.format('80x25'), '_masktol', 10, 'gpu', 1, 1, 'PACT_MASK_TOL=1', 'rel8', 'large')
for _s in ('50x25', '80x25'):
    lcamp(f'n30{_s}', f'data/FAMILY/{_s}n30', PPVC.format(_s) + '-n30', '', 0, 'gpu', 1, 1, '', 'rel8', 'large', n30=True)
PRE_N = {'small': 10, 'large': 2}
PRE_REG = ('S', 'LSW')


def legacy_runs(c):
    out = [('released', None, R8)]
    if c['arms'] in ('main', 'blindref'):
        out.append(('control', None, R8))
    if c['arms'] == 'main':
        out += [('specialist', r, list(REG.SPEC_EVAL_10x25[r])) for r in R8]
    elif c['arms'] == 'blindref':
        out += [('specialist', r, (R8 if r == 'LSW' else [r])) for r in R8]
    elif c['arms'] == 'rel8+own':
        out += [('specialist', r, [r]) for r in R8]
    return out


def legacy_jobs():
    out = []
    if 'legacy45' not in GROUPS:
        return out
    for c in LEGACY:
        folder, tag = os.path.basename(c['data']) + c['tag'], c['tag']
        names = instance_names(c['data'], c['limit'])
        seeds = N30_SEEDS if c['n30'] else LEGACY_SEEDS
        sp = SPEED[size_key(c['data'])][0]
        for arm, combo, regs in legacy_runs(c):
            for s in seeds:
                m = REG.model(arm, combo, s, 'legacy')
                files = [f'{RESULTS}/{folder}/Result_greedy+{REG.PREFIX}{m}_{r}.npy' for r in regs]
                files.append(f'{RESULTS}/{folder}/summary_{REG.PREFIX}{m}{tag}.md')
                sched = '-'
                if c['arms'] == 'rel8+sched':
                    sched = cache_dir(s, os.path.basename(c['data']), 'legacy')
                    files += [f'{sched}/{r}/{n}.npz' for r in regs for n in names]
                out.append(job(id=f'legacy:{c["name"]}:{m}', group='legacy45',
                               campaign=c['name'], env='legacy', device=c['device'],
                               omp=c['omp'], tt=c['tt'], extra=c['extra'] or '-',
                               model=REG.PREFIX + m, methods='greedy', data=c['data'],
                               base=c['base'], tag=tag or '-', limit=c['limit'],
                               regimes=' '.join(regs), sched=sched, out_root=RESULTS,
                               folder=folder, n=len(names), ckpt=REG.checkpoint(arm, combo, s, 'legacy'),
                               log=REG.train_log(arm, combo, s, 'legacy'),
                               cost=sp * len(names) * len(regs) + JOB_OVERHEAD, files=files))
        if c['n30']:
            files = [f'{RESULTS}/{folder}/Result_{q}_{r}.npy' for q in RULES for r in R8]
            files.append(f'{RESULTS}/{folder}/summary_rules.md')
            out.append(job(id=f'legacy:{c["name"]}:rules', group='legacy45', campaign=c['name'],
                           env='legacy', device='cpu', omp=1, tt=1, model='rules',
                           methods=' '.join(RULES), data=c['data'], base=c['base'], tag='-',
                           regimes=' '.join(R8), out_root=RESULTS, folder=folder,
                           n=len(names), ckpt='-', log='-',
                           cost=SPEED[size_key(c['data'])][1] * len(names) * 32 + JOB_OVERHEAD,
                           files=files))
    return out


def jobs():
    """Longest first (deterministic), so that workers that take jobs in list order
    finish close together."""
    return sorted(exact_jobs() + legacy_jobs(), key=lambda j: (-j['cost'], j['id']))


# ----------------------------------------------------------------- pre-flights
def preflights():
    """Exact environment (no stored exact files): xdet runs one small exact job twice
    and requires identical arrays (determinism at one thread); xcross re-evaluates
    the stored CPU campaign 10x25_cpuref of the seed-303 released run in full with
    its own settings (four torch threads, legacy mask; 8 combinations x 100
    instances = 800 makespans) and the stored legacy rules (one thread; 10
    instances of S and LSW), bit for bit, which proves the evaluation path.
    xdet_rules runs one masked rules job of the exact environment twice and
    requires identical arrays (the seeded tie-breaks repeat). pubdet repeats xdet
    on a public set. Legacy campaigns keep the per-campaign bit comparison of
    their stored seed-303 files. The hgnn group's jobs need hgnn_cross and
    hgnn_cross_pub: backbones/hgnn/eval_family_hgnn.py on the CPU with one thread
    reproduces the stored evaluation of hgnn-certified-joint-s301 and of
    hgnn-published-joint-s301 (earlier mask) on the first two instances of S and LSW."""
    out = []
    groups_exact = {'principal', 'secondary', 'n30', 'rules', 'public', 'pubsens', 'pubeq',
                    'pubact', 'cross'} & set(GROUPS) | set(PUB_ACTFAM_COUNTERPART) & set(GROUPS)
    if groups_exact:
        det = REG.PREFIX + REG.model('released', None, REF_SEED, 'legacy')
        for s in EXACT_SEEDS:            # the first trained exact run, if any
            m = REG.model('released', None, s, 'exact')
            if os.path.exists(REG.checkpoint('released', None, s, 'exact')):
                det = REG.PREFIX + m
                break
        out.append(dict(name='xdet', kind='twice', device='cpu', omp=1, tt=1, extra='-',
                        model=det, methods='greedy', data='data/FAMILY/10x25',
                        base=PPVC.format('10x25'), tag=XS, limit=5, regimes='S LSW',
                        mask='exact', folder='10x25' + XS, stored='-',
                        cmp_tag=f'greedy+{det}'))
        m3 = REG.PREFIX + REG.model('released', None, REF_SEED, 'legacy')
        out.append(dict(name='xcross_policy', kind='stored', device='cpu', omp=4, tt=4,
                        extra='-', model=m3, methods='greedy', data='data/FAMILY/10x25',
                        base=PPVC.format('10x25'), tag='_cpuref', limit=100,
                        regimes=' '.join(R8),             # the whole campaign: 800 makespans
                        mask='legacy', folder='10x25_cpuref', stored='10x25_cpuref',
                        cmp_tag=f'greedy+{m3}'))
        out.append(dict(name='xcross_rules', kind='stored', device='cpu', omp=1, tt=1,
                        extra='-', model='rules', methods=' '.join(RULES),
                        data='data/FAMILY/10x25', base=PPVC.format('10x25'), tag='-',
                        limit=10, regimes='S LSW', mask='legacy', folder='10x25',
                        stored='10x25', cmp_tag=','.join(RULES)))
        out.append(dict(name='xdet_rules', kind='twice', device='cpu', omp=1, tt=1,
                        extra='-', model='rules', methods=' '.join(RULES),
                        data='data/FAMILY/10x25', base=PPVC.format('10x25'), tag=XS,
                        limit=10, regimes='S LSW', mask='exact', folder='10x25' + XS,
                        stored='-', cmp_tag=','.join(RULES), rule_set='masked'))
    if ({'public', 'pubsens', 'pubeq', 'pubact'} | set(PUB_ACTFAM_COUNTERPART)) & set(GROUPS):
        pm = None
        for s in REG.PUB_SEEDS:
            if os.path.exists(REG.pub_paths('released', None, s)[0]):
                pm = REG.PUB_PREFIX + REG.pub_model('released', None, s)
                break
        if pm:
            out.append(dict(name='pubdet', kind='twice', device='cpu', omp=1, tt=1,
                            extra='-', model=pm, methods='greedy',
                            data='data/PUBLIC/family/pub_Brandimarte', base='record',
                            tag=XS, limit=3, regimes='N LSW', mask='exact',
                            folder='pub_Brandimarte' + XS, stored='-', cmp_tag=f'greedy+{pm}'))
    if 'legacy45' in GROUPS:
        m3 = REG.PREFIX + REG.model('released', None, REF_SEED, 'legacy')
        for c in LEGACY:
            stored = (os.path.basename(c['data'])[:-3] if c['n30'] else
                      os.path.basename(c['data'])) + c['tag']
            out.append(dict(name=f'legacy_{c["name"]}', kind='stored', device=c['device'],
                            omp=c['omp'], tt=c['tt'], extra=c['extra'] or '-', model=m3,
                            methods='greedy', data=c['data'], base=c['base'],
                            tag=c['tag'] or '-', limit=PRE_N[c['pre']], regimes=' '.join(PRE_REG),
                            mask='legacy', folder=os.path.basename(c['data']) + c['tag'],
                            stored=stored, cmp_tag=f'greedy+{m3}'))
    if 'hgnn' in GROUPS:
        # the HGNN evaluator with the group's settings (CPU, one thread) reproduces the
        # stored earlier-mask evaluation of the seed-301 joint run of each reward, written
        # on the GPU with one thread, on the first two instances of S and LSW, bit for bit
        for reward, pname in HGNN_CROSS.items():
            name = f'hgnn-{reward}-joint-s{HGNN_CROSS_SEED}'
            out.append(dict(name=pname, kind='stored', device='cpu', omp=1, tt=1, extra='-',
                            model=REG.PREFIX + name, methods='hgnn',
                            data=f'data/FAMILY/{HGNN_FOLDER}', base=PPVC.format(HGNN_FOLDER),
                            tag='-', limit=HGNN_CROSS_N, regimes=HGNN_CROSS_REG, mask='legacy',
                            folder=HGNN_FOLDER, stored=HGNN_FOLDER, cmp_tag=name))
    for p in out:
        p.setdefault('rule_set', 'classical')
    return out


def preflight_of(j):
    """Name of the pre-flight a job depends on."""
    if j['env'] == 'legacy':
        return f'legacy_{j["campaign"]}'
    if j['group'] == 'hgnn':
        return 'hgnn'
    pub = ('public', 'pubsens', 'pubeq', 'pubact') + tuple(PUB_ACTFAM_COUNTERPART)
    return 'pubdet' if j['group'] in pub and j['model'] != 'rules' else 'exact'


KEYS_JOB = ('id', 'group', 'campaign', 'env', 'device', 'omp', 'tt', 'extra', 'model',
            'methods', 'data', 'base', 'tag', 'limit', 'regimes', 'token_mode', 'blind',
            'best_known', 'sched', 'out_root', 'folder', 'n', 'ckpt', 'log', 'pre',
            'rule_set')
KEYS_PRE = ('name', 'kind', 'device', 'omp', 'tt', 'extra', 'model', 'methods', 'data',
            'base', 'tag', 'limit', 'regimes', 'mask', 'folder', 'stored', 'cmp_tag',
            'rule_set')


def ms(path):
    """Makespan column of a result file (some stored rule files hold only it)."""
    a = np.load(path)
    return a[:, 0] if a.ndim > 1 else a


def compare(folder_a, folder_b, cmp_tag, regimes, n):
    bad, tot = [], 0
    for t in cmp_tag.split(','):
        for r in regimes.split():
            a_p, b_p = f'{folder_a}/Result_{t}_{r}.npy', f'{folder_b}/Result_{t}_{r}.npy'
            if not (os.path.exists(a_p) and os.path.exists(b_p)):
                bad.append(f'missing {a_p if not os.path.exists(a_p) else b_p}')
                continue
            a, b = ms(a_p)[:n], ms(b_p)[:n]
            tot += len(a)
            if len(a) != n or not np.array_equal(a, b):
                bad.append(f'{t} {r}: {int((a == b[:len(a)]).sum())}/{len(a)} identical')
    if bad:
        print('PREFLIGHT MISMATCH ' + '; '.join(bad))
        return 1
    print(f'preflight identical: {tot} makespans')
    return 0


# ----------------------------------------------------------------- completeness
LEGACY_ONLY = {'10x25_masktol', '10x25_firstmask', '10x25_firstmask_off', '80x25_masktol'}
REPLACED = {'50x25_pactwarm', '80x25_pactwarm'}   # exact: schedules exported on n30
NOT_EVAL_FAMILY = {'10x25_misconfig'}             # p40's own rollout campaign
RX = re.compile(r'^Result_greedy\+10x25\+family\+(.+?)_([A-Z]+)\.npy$')
RXS = re.compile(r'^summary_10x25\+family\+(.+?)(_[a-z_]+)?\.md$')
EXPECTED = []          # every path check() requires, in order (for the plan test)


def ref_names():
    """stored legacy seed-303 run name -> (kind, key): principal (arm, combo) or the
    secondary stem whose counterpart it is."""
    d = {REG.model('released', None, REF_SEED, 'legacy'): ('principal', ('released', None)),
         REG.model('control', None, REF_SEED, 'legacy'): ('principal', ('control', None))}
    d.update({REG.model('specialist', r, REF_SEED, 'legacy'): ('principal', ('specialist', r))
              for r in R8})
    for stem, (_, cp) in REG.SECONDARY_ARMS.items():
        if cp:
            d[REG.other_model(cp, REF_SEED)] = ('secondary', stem)
    return d


def check():
    ref = ref_names()
    miss, wrong, eq_fail, ok, per = [], [], [], 0, {}
    seen = set()

    def need(path, nrows, key):
        nonlocal ok
        if path in seen:
            return
        seen.add(path)
        EXPECTED.append(path)
        per.setdefault(key, [0, 0])
        per[key][1] += 1
        if not os.path.exists(path):
            miss.append(path)
            return
        if path.endswith('.npy') and nrows is not None and np.load(path).shape[0] != nrows:
            wrong.append(f'{path}: {np.load(path).shape[0]} rows, want {nrows}')
            return
        per[key][0] += 1
        ok += 1

    def exact_folder(folder):
        return REG.FOLDER_ALIAS['exact'].get(folder, folder) + XS

    ex = {'principal', 'secondary', 'rules'} & set(GROUPS)
    for folder in sorted(os.listdir(STORED)):
        d = f'{STORED}/{folder}'
        if not os.path.isdir(d) or folder.endswith(XS) or not ex:
            continue
        if folder in LEGACY_ONLY | REPLACED | NOT_EVAL_FAMILY:
            continue
        files = sorted(os.listdir(d))
        tf = exact_folder(folder)
        for f in files:
            m = RX.match(f)
            if not m or m.group(1) not in ref:
                continue
            kind, key = ref[m.group(1)]
            rows = np.load(f'{d}/{f}').shape[0]
            if kind == 'principal' and 'principal' in GROUPS:
                for s in EXACT_SEEDS:
                    new = REG.PREFIX + REG.model(*key, s, 'exact')
                    need(f'{RESULTS}/{tf}/Result_greedy+{new}_{m.group(2)}.npy', rows, ('principal', tf))
                    need(f'{RESULTS}/{tf}/summary_{new}{XS}.md', None, ('principal', tf))
            if kind == 'secondary' and 'secondary' in GROUPS:
                for s in REG.SECONDARY_SEEDS:
                    new = REG.PREFIX + REG.secondary_model(key, s)
                    need(f'{RESULTS}/{tf}/Result_greedy+{new}_{m.group(2)}.npy', rows, ('secondary', tf))
                    need(f'{RESULTS}/{tf}/summary_{new}{XS}.md', None, ('secondary', tf))
        # rules: only in the campaign folders of the principal seed-303 runs (not in
        # probe folders such as 20x25_auditprobe)
        campaign = any((m := RX.match(f)) and ref.get(m.group(1), ('',))[0] == 'principal'
                       for f in files)
        if 'rules' in GROUPS and campaign:
            for f in files:
                if re.match(r'^Result_(FIFO|MOR|SPT|MWKR)_[A-Z]+\.npy$', f):
                    need(f'{RESULTS}/{tf}/{f}', np.load(f'{d}/{f}').shape[0], ('rules', tf))
    if 'secondary' in GROUPS:
        # the arm without a stored counterpart, and the token interventions (seed 301)
        for stem, (_, cp) in REG.SECONDARY_ARMS.items():
            if cp is None:
                for f, regs in SECONDARY_EVAL[stem]:
                    n = len(instance_names(FACTORY[f][0]))
                    for s in REG.SECONDARY_SEEDS:
                        new = REG.PREFIX + REG.secondary_model(stem, s)
                        for r in regs:
                            need(f'{RESULTS}/{f}{XS}/Result_greedy+{new}_{r}.npy', n, ('secondary', f + XS))
                        need(f'{RESULTS}/{f}{XS}/summary_{new}{XS}.md', None, ('secondary', f + XS))
        new = REG.PREFIX + REG.secondary_model(E3_ARM, E3_SEED)
        for tagm, regs in e3_modes().items():
            for r in regs:
                need(f'{RESULTS}/10x25{XS}/Result_greedy+{new}{tagm}_{r}.npy', 100, ('secondary', 'tokens'))
            need(f'{RESULTS}/10x25{XS}/summary_{new}{tagm}{XS}.md', None, ('secondary', 'tokens'))

    for size, (data, _) in N30.items():
        n = len(instance_names(data))
        tf = size + XS
        if 'n30' in GROUPS:
            names = instance_names(data)
            for s in [x for x in N30_SEEDS if x in EXACT_SEEDS]:
                m = REG.PREFIX + REG.model('released', None, s, 'exact')
                cd = cache_dir(s, size, 'exact')
                for r in R8:
                    p = f'{RESULTS}/{tf}/Result_greedy+{m}_{r}.npy'
                    need(p, n, ('n30', tf))
                    q = f'{RESULTS}/{size[:5]}{XS}/Result_greedy+{m}_{r}.npy'
                    if os.path.exists(p) and os.path.exists(q) and not np.array_equal(ms(p)[:20], ms(q)):
                        eq_fail.append(f'{p}: first 20 differ from {q}')
                    res = ms(p) if os.path.exists(p) else None
                    for i, nm in enumerate(names):
                        z = f'{cd}/{r}/{nm}.npz'
                        need(z, None, ('n30', os.path.basename(cd)))
                        if res is not None and os.path.exists(z):
                            if abs(float(np.load(z, allow_pickle=True)['ms']) - res[i]) > 1e-6:
                                eq_fail.append(f'{z}: makespan differs from {p}[{i}]')
                need(f'{RESULTS}/{tf}/summary_{m}{XS}.md', None, ('n30', tf))
        if 'rules' in GROUPS:
            for q in RULES:
                for r in R8:
                    p = f'{RESULTS}/{tf}/Result_{q}_{r}.npy'
                    need(p, n, ('rules', tf))
                    o = f'{RESULTS}/{size[:5]}{XS}/Result_{q}_{r}.npy'
                    if os.path.exists(p) and os.path.exists(o) and not np.array_equal(ms(p)[:20], ms(o)):
                        eq_fail.append(f'{p}: first 20 differ from {o}')

    # no stored public results: the set x arm x regime grid
    pub_groups = [('public', st) for st in REG.PUB_SETS if 'public' in GROUPS] + \
                 [('pubsens', st) for st in REG.PUB_SENS_SETS if 'pubsens' in GROUPS]
    for g, st in pub_groups:
        n = len(instance_names(f'data/PUBLIC/family/{st}'))
        tf = st + XS
        for arm, combo, regs in PUB_RUNS:
            for s in REG.PUB_SEEDS:
                m = REG.PUB_PREFIX + REG.pub_model(arm, combo, s)
                for r in regs:
                    need(f'{PUB_RESULTS}/{tf}/Result_greedy+{m}_{r}.npy', n, (g, tf))
                need(f'{PUB_RESULTS}/{tf}/summary_{m}{XS}.md', None, (g, tf))
        for q in RULES:
            for r in R8:
                need(f'{PUB_RESULTS}/{tf}/Result_{q}_{r}.npy', n, (g, tf))
    # the equal-experience arm: the eight combinations and the summary of each seed
    if 'pubeq' in GROUPS:
        for st, _ in pub_eq_sets():
            n = len(instance_names(f'data/PUBLIC/family/{st}'))
            tf = st + XS
            for s in REG.PUB_SEEDS:
                m = REG.PUB_PREFIX + REG.pub_eq_model(s)
                for r in R8:
                    need(f'{PUB_RESULTS}/{tf}/Result_greedy+{m}_{r}.npy', n, ('pubeq', tf))
                need(f'{PUB_RESULTS}/{tf}/summary_{m}{XS}.md', None, ('pubeq', tf))
    # the active-rule arm: the eight combinations and the summary of each seed
    if 'pubact' in GROUPS:
        for st, _ in pub_act_sets():
            n = len(instance_names(f'data/PUBLIC/family/{st}'))
            tf = st + XS
            for s in REG.PUB_ACT_SEEDS:
                m = REG.PUB_PREFIX + REG.pub_act_model(s)
                for r in R8:
                    need(f'{PUB_RESULTS}/{tf}/Result_greedy+{m}_{r}.npy', n, ('pubact', tf))
                need(f'{PUB_RESULTS}/{tf}/summary_{m}{XS}.md', None, ('pubact', tf))
    # the family under the active mask: the eight combinations (a specialist its own) and
    # the summary of every run of each group
    for g in ('pubactfam', 'pubactsens'):
        if g not in GROUPS:
            continue
        for st in (REG.PUB_SETS if g == 'pubactfam' else REG.PUB_SENS_SETS):
            n = len(instance_names(f'data/PUBLIC/family/{st}'))
            tf = st + XS
            for arm, combo, s, regs in pub_actfam_runs(g, st):
                m = REG.PUB_PREFIX + REG.pub_rule_model(arm, combo, s)
                for r in regs:
                    need(f'{PUB_RESULTS}/{tf}/Result_greedy+{m}_{r}.npy', n, (g, tf))
                need(f'{PUB_RESULTS}/{tf}/summary_{m}{XS}.md', None, (g, tf))
    if 'pubacteq' in GROUPS:
        for st, _ in pub_eq_sets():
            n = len(instance_names(f'data/PUBLIC/family/{st}'))
            tf = st + XS
            for s in REG.PUB_SEEDS:
                m = REG.PUB_PREFIX + REG.pub_rule_model('eq', None, s)
                for r in R8:
                    need(f'{PUB_RESULTS}/{tf}/Result_greedy+{m}_{r}.npy', n, ('pubacteq', tf))
                need(f'{PUB_RESULTS}/{tf}/summary_{m}{XS}.md', None, ('pubacteq', tf))
    if 'hgnn' in GROUPS:
        hgnn_check(need)
    if 'cross' in GROUPS:
        cross_check(need, eq_fail)

    if 'legacy45' in GROUPS:
        legacy_check(need, eq_fail)

    for (g, tf), (a, b) in sorted(per.items()):
        print(f'  {g:10s} {tf:36s} {a:6d} / {b:6d}{"" if a == b else "  INCOMPLETE"}')
    print(f'complete: {ok} files; missing {len(miss)}; wrong length {len(wrong)}; '
          f'consistency failures {len(eq_fail)}')
    for x in (miss[:20] + wrong[:20] + eq_fail[:20]):
        print('   ', x)
    if len(miss) > 20:
        print(f'    ... and {len(miss) - 20} more missing')
    return 0 if not (miss or wrong or eq_fail) else 1


def cross_check(need, eq_fail):
    """Group cross, from the registry grid (no stored counterpart): every specialist of
    every seed of EXACT_SEEDS under every other combination, 100 rows each, and its
    summary. Where the principal 10x25 campaign evaluates the same run on the same
    combination (principal_runs), both arrays must hold identical makespans: greedy
    rollouts on one thread do not depend on the job they ran in."""
    data = FACTORY[CROSS_DATA][0]
    n = len(instance_names(data))
    tf = REG.CROSS_FOLDER + XS
    main = {combo: set(regs) for arm, combo, regs in principal_runs(CROSS_DATA)
            if arm == 'specialist'}
    for r in R8:
        for s in EXACT_SEEDS:
            m = REG.PREFIX + REG.model('specialist', r, s, 'exact')
            for c in R8:
                if c == r:
                    continue
                p = f'{RESULTS}/{tf}/Result_greedy+{m}_{c}.npy'
                need(p, n, ('cross', tf))
                q = f'{RESULTS}/{CROSS_DATA}{XS}/Result_greedy+{m}_{c}.npy'
                if c in main[r] and os.path.exists(p) and os.path.exists(q) and \
                        not np.array_equal(ms(p), ms(q)):
                    eq_fail.append(f'{p}: makespans differ from {q}')
            need(f'{RESULTS}/{tf}/summary_{m}{REG.EXACT_TAG[REG.CROSS_FOLDER]}.md', None, ('cross', tf))


def legacy_check(need, eq_fail):
    """Round-one check of the legacy seeds 304-305 and the legacy 30-instance sets."""
    ref = {REG.model('released', None, REF_SEED, 'legacy'): ('released', None),
           REG.model('control', None, REF_SEED, 'legacy'): ('control', None)}
    ref.update({REG.model('specialist', r, REF_SEED, 'legacy'): ('specialist', r) for r in R8})
    for folder in sorted(os.listdir(STORED)):
        d = f'{STORED}/{folder}'
        if not os.path.isdir(d) or folder.endswith(XS):
            continue
        files = os.listdir(d)
        refs = [(f, RX.match(f)) for f in files]
        refs = [(f, m) for f, m in refs if m and m.group(1) in ref]
        if not refs:
            continue
        sums = [(m.group(1), m.group(2) or '') for m in map(RXS.match, files)
                if m and m.group(1) in ref]
        for s in LEGACY_SEEDS:
            for f, m in refs:
                new = REG.model(*ref[m.group(1)], s, 'legacy')
                need(f'{RESULTS}/{folder}/Result_greedy+{REG.PREFIX}{new}_{m.group(2)}.npy',
                     np.load(f'{d}/{f}').shape[0], ('legacy45', folder))
            for name, tg in sums:
                new = REG.model(*ref[name], s, 'legacy')
                need(f'{RESULTS}/{folder}/summary_{REG.PREFIX}{new}{tg}.md', None, ('legacy45', folder))
    for size in ('50x25', '80x25'):
        tf = f'{size}n30'
        n = len(instance_names(f'data/FAMILY/{tf}'))
        for s in N30_SEEDS:
            m = REG.model('released', None, s, 'legacy')
            for r in R8:
                p = f'{RESULTS}/{tf}/Result_greedy+{REG.PREFIX}{m}_{r}.npy'
                need(p, n, ('legacy45', tf))
                if s in REG.DEFAULT_SEEDS and os.path.exists(p):
                    b = ms(f'{STORED}/{size}/Result_greedy+{REG.PREFIX}{m}_{r}.npy')
                    if not np.array_equal(ms(p)[:20], b):
                        eq_fail.append(f'{p}: first 20 differ from {STORED}/{size}')
            need(f'{RESULTS}/{tf}/summary_{REG.PREFIX}{m}.md', None, ('legacy45', tf))
        for q in RULES:
            for r in R8:
                p = f'{RESULTS}/{tf}/Result_{q}_{r}.npy'
                need(p, n, ('legacy45', tf))
                if os.path.exists(p) and not np.array_equal(ms(p)[:20], ms(f'{STORED}/{size}/Result_{q}_{r}.npy')):
                    eq_fail.append(f'{p}: first 20 differ from {STORED}/{size}')
        for s in LEGACY_SEEDS:
            m = REG.model('released', None, s, 'legacy')
            cd = cache_dir(s, size, 'legacy')
            rp = f'{RESULTS}/{size}_pactwarm/Result_greedy+{REG.PREFIX}{m}_'
            for r in R8:
                res = ms(rp + f'{r}.npy') if os.path.exists(rp + f'{r}.npy') else None
                for i, nm in enumerate(instance_names(f'data/FAMILY/{size}', 10)):
                    z = f'{cd}/{r}/{nm}.npz'
                    need(z, None, ('legacy45', os.path.basename(cd)))
                    if res is not None and os.path.exists(z) and \
                            abs(float(np.load(z, allow_pickle=True)['ms']) - res[i]) > 1e-6:
                        eq_fail.append(f'{z}: makespan differs from the result file')


def counts():
    js = jobs()
    agg, grp = {}, {}
    for j in js:
        k = (j['group'], j['campaign'], j['device'])
        a = agg.setdefault(k, [0, 0, 0, 0.0])
        nroll = j['n'] * len(j['regimes'].split()) * len(j['methods'].split())
        for x, v in ((0, 1), (1, len(j['files'])), (2, nroll), (3, j['cost'] / 3600)):
            a[x] += v
        g = grp.setdefault(j['group'], [0, 0, 0, 0.0])
        for x, v in ((0, 1), (1, len(j['files'])), (2, nroll), (3, j['cost'] / 3600)):
            g[x] += v
    print(f'{"group":10s} {"campaign":30s} {"device":6s} {"jobs":>5s} {"files":>6s} '
          f'{"rollouts":>9s} {"core-h":>8s}')
    for (g, c, dv), (a, b, n, h) in agg.items():
        print(f'{g:10s} {c:30s} {dv:6s} {a:5d} {b:6d} {n:9d} {h:8.1f}')
    print('per group:')
    tot = [0, 0, 0, 0.0]
    for g, (a, b, n, h) in grp.items():
        print(f'{g:10s} {"":30s} {"":6s} {a:5d} {b:6d} {n:9d} {h:8.1f}')
        tot = [tot[0] + a, tot[1] + b, tot[2] + n, tot[3] + h]
    print(f'{"total":48s}{tot[0]:5d} {tot[1]:6d} {tot[2]:9d} {tot[3]:8.1f}')


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'jobs':
        for j in jobs():
            j['pre'] = preflight_of(j)
            print('|'.join(str(j[k]) for k in KEYS_JOB) + '|' + ' '.join(j['files']))
    elif cmd == 'preflight':
        for p in preflights():
            print('|'.join(str(p[k]) for k in KEYS_PRE))
    elif cmd == 'compare':
        sys.exit(compare(*sys.argv[2:5], sys.argv[5], int(sys.argv[6])))
    elif cmd == 'check':
        sys.exit(check())
    elif cmd == 'counts':
        counts()
    else:
        sys.exit(__doc__)
