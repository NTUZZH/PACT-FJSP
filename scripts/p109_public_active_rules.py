r"""P109: the four dispatching rules of the public benchmark rolled out under the active mask.

WHY
  The public benchmark is reported under the active mask (mask_mode 'active', the condition
  of Giffler and Thompson: the mask admits a pair of an operation and a machine while its
  processing start is earlier than the earliest completion over the compatible pairs). Its
  comparison with the dispatching rules (scripts/p89_public_family.py --family active:
  section 3, the regime-N anchor of section 4, the sensitivity set of section 7 and the
  rules of section 9) needs FIFO, MOR, SPT and MWKR rolled out under that mask on every set
  the comparison under the non-delay mask reads: the six test sets of arm_registry.PUB_SETS
  and the sensitivity set of PUB_SENS_SETS. eval_family.py cannot write them: it applies
  the mask a run's configuration records to the policy only and rolls out every rule under
  the non-delay mask. This script repeats its rule rollout with the mask as a parameter and
  changes nothing else.

ROLLOUT (one instance, one combination, one rule; the construction of eval_family.py)
  record       data/PUBLIC/family/<set>/desc/<R>/<name>.desc.json, read with
               ConstraintDescriptor.from_json(record, setup_convention='attached'); the base
               instance is the record's base_instance (eval_family.py --base record);
  environment  FJSPEnvFamily(J, M, family_mode=True, blind_channels=(), mask_mode=<mask>,
               pair_extra='none', first_mask='exact', mask_ties='exact'), delta_mode from
               params.configs (resolve_delta_mode), set before the data, the type lists of the
               public instance with source 'none';
  rule         common_utils.masked_rule_action (eval_family.py --rule_set masked): the rule
               acts on the pairs the environment's mask admits, so under the active mask on
               the pairs the active mask admits; np.random.seed(t) before each rollout, t the
               tie-break seed (--seed; default params seed_test, 50, the seed of the stored
               rule arrays; eval_family.py --seed_test t);
  check        constraint_family.reference_sim.validate_schedule on the realised schedule,
               the check of eval_family.py, and one decision per operation; an infeasible
               schedule refuses the job.
  Row i of an array is the i-th record of desc/<R>/ in sorted file-name order (the order of
  eval_family.py); column 1 holds the seconds of the rollout (planning figures only: the
  rollouts are deterministic, so the arrays do not depend on the load of the core).

OUTPUTS (root ACTIVE_ROOT = test_results/PUBFAMILY/rules_act, or --out-root)
  <root>/<set>_xmask/Result_<rule>_<R>.npy   [n, 2], the layout of the stored rule arrays
  <root>/<set>_xmask/summary_rules_xmask.md  the rule summary of eval_family.py (n/n feasible,
                                             mean, regime-N gap to the best-known table) and
                                             the line 'pairs: mask_mode=<mask>'
  <root>/<set>_xmask/rules_meta.json         mask, settings, code fingerprint, counts
  The folder rules_act lies beside the <set>_xmask folders of test_results/PUBFAMILY and
  matches none of them: p89 check_duplicates globs test_results/PUBFAMILY/<set>*, p91
  load_stored and the rule comparison read one <set>_xmask folder, so no reader of the
  non-delay arrays takes these files. A job (set, combination) writes the four arrays of its
  combination through temporary names and renames them; a finished job is skipped and a
  partial one refuses. The summary and the meta record are written once all 32 arrays of a
  set exist. Writing under the non-delay mask is allowed only under --out-root outside
  test_results (the stored non-delay arrays exist); under the active mask a root inside
  test_results lies inside ACTIVE_ROOT. The meta record names the mask and the tie-break
  seed: a run into a set folder whose meta record names another mask or seed refuses, and a
  tie-break seed other than 50 needs --out-root (the default root holds the seed-50 arrays).

THE RULE COMPARISON (scripts/rulecmp_plan.py --mask active)
  The rule jobs of the comparison with the dispatching rules are this script's run with the
  campaign's folders as --out-root: the tie-break seeds 51, 52 and 53 on a test set (--seed t,
  <campaign folder>/rulecmp_ties/seed<t>/<set>_xmask) and the validation records the public
  runs validated on (--sets pub_SD2_vali_10x5+mix, VALI_SETS, no best-known table:
  <campaign folder>/rulecmp_vali/pub_SD2_vali_10x5+mix_xmask). The layout is the one
  eval_family.py writes for the same jobs under the non-delay mask.

CHECKS (tests/test_public_active_rules.py runs them on a sample)
  check-nondelay  under the non-delay mask the rollouts reproduce the stored non-delay rule
                  arrays bit for bit (makespans): with the tie-break seed 50 those of
                  test_results/PUBFAMILY/<set>_xmask (the validation records: the
                  rulecmp_vali folder of the headline set), with --seed t those of the
                  rulecmp_ties/seed<t> folder of the set (scripts/rulecmp_plan.py);
  check-p107      on SD2 10x5 the rollouts under the active and the non-delay mask equal the
                  makespans of the cases of scripts/p107_mask_rule_variants.py (the
                  folder pub10x5/<R>/<instance>.json below its DEFAULT_ROOT, read from that
                  script so that no other file names it; keys adm.active.rules and
                  adm.nondelay.rules);
  check-arrays    the rollouts under --mask with --seed reproduce the arrays of
                  --ref-root/<set>_xmask (the first --n records), bit for bit;
  every rollout   validator and decision count, as above.

USAGE (repository root; CPU only; one thread)
  python scripts/p109_public_active_rules.py plan
  python scripts/p109_public_active_rules.py run [--sets a,b] [--combos N,S] [--budget_s 450]
         [--seed t --out-root <folder>]
  python scripts/p109_public_active_rules.py check-nondelay --sets <set> [--combos ..] [--n k]
         [--seed t]
  python scripts/p109_public_active_rules.py check-p107 [--combos ..] [--n k]
  python scripts/p109_public_active_rules.py check-arrays --mask active --sets <set>
         --ref-root <folder> [--seed t] [--combos ..] [--n k]
"""
import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
if ROOT not in sys.path:
    sys.path.append(ROOT)     # params, common_utils and constraint_family

import numpy as np  # noqa: E402

import arm_registry as REG  # noqa: E402  (removes --pact-* options from sys.argv)

R8 = list(REG.COMBOS)
RULES = ('FIFO', 'MOR', 'SPT', 'MWKR')
MASKS = ('active', 'nondelay')
XS = REG.FOLDER_SUFFIX['exact']
MASK_OPTS = {'first_mask': 'exact', 'mask_ties': 'exact'}
# the root of the rule arrays under the mask of each rule word (arm_registry.PUB_RULE_WORD)
RULE_ROOT = {m: f'{REG.PUB_RESULTS}/rules_{w}' for m, w in REG.PUB_RULE_WORD.items()}
ACTIVE_ROOT = RULE_ROOT['active']
SETS = list(REG.PUB_SETS) + list(REG.PUB_SENS_SETS)
# the validation records every public run validated on (pub_vali_dir of the training
# configurations; scripts/rulecmp_plan.py Family.vali_data): rolled out only when named with
# --sets, never by default
VALI_SETS = ('pub_SD2_vali_10x5+mix',)
ALL_SETS = SETS + list(VALI_SETS)
STORED_SEED = 50                 # params.py seed_test: the tie-break seed of the stored arrays
SUMMARY = f'summary_rules{XS}.md'
META = 'rules_meta.json'
P107_CASES = 'pub10x5'           # the folder of the SD2 10x5 cases below p107's DEFAULT_ROOT
P107_SET = 'pub_SD2_test_10x5+mix'
CODE = ('eval_family.py', 'common_utils.py', 'params.py', 'constraint_family/fjsp_env_family.py',
        'constraint_family/descriptor.py', 'constraint_family/public_adapter.py',
        'constraint_family/reference_sim.py', 'constraint_family/public_types.py',
        'scripts/p109_public_active_rules.py')
# seconds per rule rollout on one core under the non-delay mask (eval_newseeds_plan.SPEED),
# a planning figure for the active mask, which admits more pairs per decision
SPEED = {'pub_SD2_test_10x5+mix': 0.072, 'pub_SD2_test_20x10+mix': 0.337,
         'pub_Brandimarte': 0.136, 'pub_Hurink_edata': 0.318, 'pub_Hurink_rdata': 0.314,
         'pub_Hurink_vdata': 0.326}
for _st, (_base, _bk, _ch) in REG.PUB_SENS_SETS.items():
    SPEED.setdefault(_st, SPEED[_base])
for _st in VALI_SETS:
    SPEED.setdefault(_st, SPEED['pub_SD2_test_10x5+mix'])


def stop(msg):
    raise SystemExit(f'REFUSING: {msg}')


def table(st):
    """Best-known table of a test or sensitivity set; None for the validation records."""
    if st in VALI_SETS:
        return None
    return REG.PUB_SETS[st] if st in REG.PUB_SETS else REG.PUB_SENS_SETS[st][1]


def folder(root, st):
    return os.path.join(root, st + XS)


def array_name(q, r):
    return f'Result_{q}_{r}.npy'


def records(st, combo, root=ROOT):
    """[(name, record)] of one set and combination in sorted file-name order."""
    d = os.path.join(root, 'data', 'PUBLIC', 'family', st, 'desc', combo)
    if not os.path.isdir(d):
        stop(f'no records in {os.path.relpath(d, root)}')
    out = []
    for f in sorted(os.listdir(d)):
        if f.endswith('.desc.json'):
            with open(os.path.join(d, f)) as fh:
                out.append((f[:-len('.desc.json')], json.load(fh)))
    return out


def fingerprint(root=ROOT):
    h = hashlib.sha256()
    for p in CODE:
        with open(os.path.join(root, p), 'rb') as f:
            h.update(f.read())
    return h.hexdigest()[:16]


# ================================================================== rollout
_CONF = {}


def _setup():
    """params.configs and the environment classes, imported once with a CPU argv (params
    parses sys.argv at import time)."""
    if not _CONF:
        os.chdir(ROOT)               # base_instance paths of the records are relative to it
        keep = sys.argv
        sys.argv = [keep[0], '--device', 'cpu']
        try:
            from params import configs
        finally:
            sys.argv = keep
        from constraint_family.descriptor import ConstraintDescriptor
        from constraint_family.fjsp_env_family import FJSPEnvFamily, resolve_delta_mode
        from constraint_family.public_adapter import load_base_instance
        from constraint_family.public_types import type_lists
        from constraint_family.reference_sim import validate_schedule
        from common_utils import masked_rule_action
        _CONF.update(configs=configs, Desc=ConstraintDescriptor, Env=FJSPEnvFamily,
                     delta=resolve_delta_mode(configs), load=load_base_instance,
                     types=type_lists, validate=validate_schedule, act=masked_rule_action,
                     seed=int(configs.seed_test))
        if configs.pub_type_source != 'none':
            stop(f'params pub_type_source is {configs.pub_type_source!r}, want none')
    return _CONF


def tie_seed(seed=None):
    """The tie-break seed of a rollout: seed, else params seed_test (50)."""
    return int(_setup()['seed'] if seed is None else seed)


def rollout(rec, rule, mask, seed=None):
    """(makespan, seconds) of one rule on one record under mask, numpy seeded with the
    tie-break seed (tie_seed); refuses an infeasible schedule or a decision count other
    than the number of operations."""
    C = _setup()
    desc = C['Desc'].from_json(rec, setup_convention='attached')
    jl, pt, meta = C['load'](rec['base_instance'])
    M = pt.shape[1]
    env = C['Env'](jl.shape[0], M, family_mode=True, blind_channels=(), mask_mode=mask,
                   pair_extra='none', **MASK_OPTS)
    env.delta_mode = C['delta']
    env.reward_delta_env = (C['delta'] != 'zero')
    if meta['op_type'] is not None:
        otl, mtl = [meta['op_type']], [meta['mch_type']]
    else:
        otl, mtl = C['types']('none', [desc], M)
    env.set_initial_data([jl], [pt], descriptor_list=[desc], op_type_list=otl,
                         mch_type_list=mtl)
    for k, v in MASK_OPTS.items():
        if env.mask_setting(k) != v:
            stop(f'the environment reads {k}={env.mask_setting(k)}, want {v}')
    if env.mask_mode != mask:
        stop(f'the environment applies the {env.mask_mode} mask, want {mask}')
    amch = np.full(pt.shape[0], -1, dtype=int)
    np.random.seed(tie_seed(seed))
    t0 = time.time()
    steps = 0
    while not env.done().all():
        a = int(C['act'](rule, env))
        amch[env.candidate[0, a // M]] = a % M
        env.step(np.array([a]))
        steps += 1
    dt = time.time() - t0
    ms = float(env.current_makespan[0])
    op_ct = env.true_op_ct[0]
    op_st = op_ct - pt[np.arange(pt.shape[0]), amch]
    viol = C['validate'](jl, pt, desc, op_st, op_ct, amch)
    if viol or steps != pt.shape[0] or (amch < 0).any():
        stop(f'{rec.get("base_instance")} {rec.get("regime")} {rule} under {mask}: infeasible '
             f'schedule or {steps} decisions for {pt.shape[0]} operations ({list(viol)[:2]})')
    return ms, dt


def rollout_combo(st, combo, mask, n=0, root=ROOT, seed=None):
    """{rule: [n, 2] array} of one set and combination (first n records, 0 = all), tie-break
    seed tie_seed(seed)."""
    recs = records(st, combo, root)
    if n:
        recs = recs[:n]
    out = {}
    for q in RULES:
        out[q] = np.array([rollout(rec, q, mask, seed) for _, rec in recs], dtype=float)
    return out, [nm for nm, _ in recs]


# ================================================================== jobs
def job_state(root, st, combo):
    """'done' when the four arrays of the combination exist, 'none' when none exists,
    'partial' otherwise."""
    have = [os.path.exists(os.path.join(folder(root, st), array_name(q, combo))) for q in RULES]
    return 'done' if all(have) else ('none' if not any(have) else 'partial')


def check_out_root(root, mask):
    """Refuses a root that would mix the new arrays with the stored ones: the folders of the
    stored arrays themselves, or the repository's test_results under the non-delay mask."""
    a = os.path.abspath(root)
    stored = os.path.abspath(os.path.join(ROOT, REG.PUB_RESULTS))
    if a == stored or a == os.path.abspath(os.path.join(ROOT, 'test_results')):
        stop(f'{root} holds the stored rule arrays; the arrays under a mask go to a folder of '
             f'their own ({ACTIVE_ROOT})')
    tr = os.path.abspath(os.path.join(ROOT, 'test_results')) + os.sep
    if mask == 'nondelay' and (a + os.sep).startswith(tr):
        stop('the non-delay rule arrays are stored; a rollout under the non-delay mask is '
             'written only under --out-root outside test_results')
    own = os.path.abspath(os.path.join(ROOT, RULE_ROOT[mask])) + os.sep \
        if mask in RULE_ROOT else None
    if mask != 'nondelay' and (a + os.sep).startswith(tr) and not (a + os.sep).startswith(own):
        stop(f'{root} lies in test_results outside {RULE_ROOT[mask]}, the folder of the rule '
             f'arrays under the {mask} mask, so it would mix them with the arrays of another mask')


def check_meta(root, st, mask, seed):
    """Refuses a set folder whose meta record names another mask or tie-break seed."""
    p = os.path.join(folder(root, st), META)
    if not os.path.exists(p):
        return
    with open(p) as f:
        m = json.load(f)
    if m.get('mask_mode') != mask or int(m.get('tie_break_seed', -1)) != seed:
        stop(f'{st}: the meta record of the folder names the {m.get("mask_mode")} mask and the '
             f'tie-break seed {m.get("tie_break_seed")}, this run the {mask} mask and seed {seed}')


def write_job(root, st, combo, mask, arrays):
    d = folder(root, st)
    os.makedirs(d, exist_ok=True)
    tmp = []
    for q in RULES:
        p = os.path.join(d, array_name(q, combo))
        t = p + '.tmp.npy'
        np.save(t, arrays[q])
        tmp.append((t, p))
    for t, p in tmp:
        if os.path.exists(p):
            stop(f'{os.path.relpath(p, ROOT)} appeared meanwhile')
        os.replace(t, p)


def summary_lines(st, mask, arrays, names, root=ROOT):
    """The rule summary of eval_family.py for one set: arrays[(rule, combo)] [n, 2]; the
    regime-N gap is n/a without a best-known table (the validation records)."""
    import csv
    bks = {}
    if table(st) is not None:
        with open(os.path.join(root, table(st))) as f:
            bks = {r['instance']: float(r['best_known_makespan'])
                   for r in csv.DictReader(f) if r['best_known_makespan']}
    n = len(names)
    L = [f'# FAMILY eval — rules{XS}', '',
         f'mask: first_mask={MASK_OPTS["first_mask"]} mask_ties={MASK_OPTS["mask_ties"]}', '',
         'rules: rule_set=masked', '', f'pairs: mask_mode={mask}', '',
         '| regime | method | mean | std | feasible | gap(cov) | gap to best known (N only) |',
         '|---|---|---|---|---|---|---|']
    for r in R8:
        for q in RULES:
            v = arrays[(q, r)][:, 0]
            bg = [100.0 * (v[i] - bks[nm]) / bks[nm] for i, nm in enumerate(names)
                  if nm in bks and r == 'N']
            g = f'{np.mean(bg):.2f}% ({len(bg)})' if bg else 'n/a'
            L.append(f'| {r} | {q} | {v.mean():.1f} | {v.std():.1f} | {n}/{n} | n/a | {g} |')
    return L


def finish_set(root, st, mask, seed=None):
    """Summary and meta record of a set once all 32 arrays exist; returns True when written
    or present (a present meta record must name the mask and the tie-break seed)."""
    d = folder(root, st)
    seed = tie_seed(seed)
    if any(job_state(root, st, r) != 'done' for r in R8):
        return False
    if os.path.exists(os.path.join(d, SUMMARY)) and os.path.exists(os.path.join(d, META)):
        check_meta(root, st, mask, seed)
        return True
    names = [nm for nm, _ in records(st, 'N')]
    for r in R8:
        if [nm for nm, _ in records(st, r)] != names:
            stop(f'{st}: combination {r} lists other records than N')
    arrays = {(q, r): np.load(os.path.join(d, array_name(q, r))) for q in RULES for r in R8}
    for k, a in arrays.items():
        if a.shape != (len(names), 2) or not np.all(np.isfinite(a)) or np.any(a[:, 0] <= 0):
            stop(f'{st} {k}: array of shape {a.shape} or with non-positive values')
    with open(os.path.join(d, SUMMARY + '.tmp'), 'w') as f:
        f.write('\n'.join(summary_lines(st, mask, arrays, names)) + '\n')
    os.replace(os.path.join(d, SUMMARY + '.tmp'), os.path.join(d, SUMMARY))
    meta = dict(generator='scripts/p109_public_active_rules.py', set=st, mask_mode=mask,
                **MASK_OPTS, rule_set='masked', rules=list(RULES), combinations=R8,
                tie_break_seed=seed, delta_mode=_setup()['delta'],
                pair_extra='none', n=len(names), names=names, record_folder=f'data/PUBLIC/family/{st}',
                best_known=table(st), code_fingerprint=fingerprint(),
                feasible=f'{32 * len(names)} of {32 * len(names)} schedules',
                note='column 1 of every array holds the seconds of the rollout, planning figures only')
    with open(os.path.join(d, META + '.tmp'), 'w') as f:
        json.dump(meta, f, indent=1)
    os.replace(os.path.join(d, META + '.tmp'), os.path.join(d, META))
    return True


def parse_list(text, allowed, what, default=None):
    xs = [x for x in (text or '').split(',') if x] or list(allowed if default is None else default)
    bad = [x for x in xs if x not in allowed]
    if bad:
        stop(f'unknown {what} {bad}; want some of {list(allowed)}')
    return xs


# ================================================================== commands
def out_root(a):
    """(shown root, absolute root) of the arrays of a command."""
    root = a.out_root or RULE_ROOT[a.mask]
    return root, (root if os.path.isabs(root) else os.path.join(ROOT, root))


def plan(a):
    root, root_abs = out_root(a)
    tot = 0.0
    print(f'[p109] rules {", ".join(RULES)} under the {a.mask} mask, root {root}'
          + (f', tie-break seed {a.seed}' if getattr(a, 'seed', None) is not None else ''))
    for st in parse_list(a.sets, ALL_SETS, 'sets', SETS):
        n = len(records(st, 'N'))
        states = {r: job_state(root_abs, st, r) for r in R8}
        left = [r for r, s in states.items() if s != 'done']
        cost = SPEED[st] * n * len(RULES) * len(left)
        tot += cost
        print(f'  {st}: {n} records; done {8 - len(left)} of 8 combinations; left '
              f'{",".join(left) or "none"} (about {cost / 60:.1f} min on one core)')
    print(f'left in total: about {tot / 3600:.2f} core-hours (non-delay speed)')
    return tot


def run(a):
    root, root_abs = out_root(a)
    check_out_root(root_abs, a.mask)
    seed = tie_seed(getattr(a, 'seed', None))
    if seed != STORED_SEED and not a.out_root:
        stop(f'the default root {root} holds the arrays of the tie-break seed {STORED_SEED}; the '
             f'seed {seed} needs --out-root')
    sets = parse_list(a.sets, ALL_SETS, 'sets', SETS)
    for st in sets:
        check_meta(root_abs, st, a.mask, seed)
    t_start = time.time()
    last = 0.0
    done = []
    for st in sets:
        for r in parse_list(a.combos, R8, 'combinations'):
            s = job_state(root_abs, st, r)
            if s == 'done':
                continue
            if s == 'partial':
                stop(f'{st} {r}: some of the four arrays exist in {root}; inspect and remove them')
            if a.budget_s and time.time() - t_start + last > a.budget_s:
                print(f'[p109] budget of {a.budget_s:g} s reached before {st} {r}')
                break
            t0 = time.time()
            arrays, _ = rollout_combo(st, r, a.mask, seed=seed)
            write_job(root_abs, st, r, a.mask, arrays)
            last = time.time() - t0
            done.append((st, r))
            print(f'[p109] {st} {r} under {a.mask}' + (f', tie-break seed {seed}'
                                                       if seed != STORED_SEED else '')
                  + f': {len(arrays["SPT"])} records x 4 rules in '
                  f'{last:.1f} s; means ' + ', '.join(f'{q} {arrays[q][:, 0].mean():.1f}'
                                                     for q in RULES), flush=True)
        else:
            if finish_set(root_abs, st, a.mask, seed):
                print(f'[p109] {st}: all 32 arrays, summary and meta record in place')
            continue
        break
    for st in sets:
        finish_set(root_abs, st, a.mask, seed)
    return done


def nondelay_folder(st, seed):
    """Folder of the stored non-delay rule arrays of a set and tie-break seed: the set's
    result folder (seed 50), the rulecmp_ties/seed<t> folder of the comparison with the
    dispatching rules (another seed), the rulecmp_vali folder of the headline set (the
    validation records, seed 50); the folders of scripts/rulecmp_plan.py."""
    if st in VALI_SETS:
        if seed != STORED_SEED:
            stop(f'the non-delay arrays of {st} exist for the tie-break seed {STORED_SEED} only')
        import rulecmp_plan as PLAN
        fam = PLAN.Family('public', K=PLAN.K_MIN)
        if fam.vali_dir != st + XS:
            stop(f'the comparison validates on {fam.vali_dir}, not {st}')
        return fam.vali_folder()
    if seed == STORED_SEED:
        return os.path.join(ROOT, REG.PUB_RESULTS, st + XS)
    import rulecmp_plan as PLAN
    if st not in PLAN.PUB_SET_INFO:
        stop(f'no stored non-delay arrays of {st} with the tie-break seed {seed}')
    if seed not in PLAN.TIE_SEEDS:
        stop(f'the comparison stores the tie-break seeds {list(PLAN.TIE_SEEDS)}, not {seed}')
    return PLAN.Family('public', K=PLAN.K_MIN, set_name=st).ties_dir(seed)


def compare(a, mask, ref_of, what, out=print):
    """Rollouts under mask with the tie-break seed of a against the arrays of the folders
    ref_of(set) (the first --n records), bit for bit."""
    bad, tot = [], 0
    seed = tie_seed(getattr(a, 'seed', None))
    for st in parse_list(a.sets, ALL_SETS, 'sets', SETS):
        ref_dir = ref_of(st, seed)
        for r in parse_list(a.combos, R8, 'combinations'):
            arrays, names = rollout_combo(st, r, mask, n=a.n, seed=seed)
            for q in RULES:
                p = os.path.join(ref_dir, array_name(q, r))
                if not os.path.exists(p):
                    stop(f'missing reference array {os.path.relpath(p, ROOT)}')
                ref = np.load(p)[:len(names), 0]
                v = arrays[q][:, 0]
                tot += len(v)
                if len(ref) != len(v) or not np.array_equal(v, ref):
                    bad.append(f'{st} {r} {q}: {int((v == ref).sum()) if len(ref) == len(v) else 0} '
                               f'of {len(v)} equal')
    out(f'[p109] {what}' + (f' (tie-break seed {seed})' if seed != STORED_SEED else '')
        + f': {tot} makespans, ' + ('all equal' if not bad else 'MISMATCH ' + '; '.join(bad[:8])))
    return tot, bad


def check_nondelay(a, out=print):
    """Rollouts under the non-delay mask against the stored rule arrays, bit for bit."""
    return compare(a, 'nondelay', nondelay_folder, 'non-delay mask against the stored arrays',
                   out)


def check_arrays(a, out=print):
    """Rollouts under --mask against the arrays of --ref-root/<set>_xmask, bit for bit."""
    if not a.ref_root:
        stop('check-arrays needs --ref-root')
    base = a.ref_root if os.path.isabs(a.ref_root) else os.path.join(ROOT, a.ref_root)
    return compare(a, a.mask, lambda st, seed: folder(base, st),
                   f'{a.mask} mask against the arrays of {a.ref_root}', out)


def p107_root():
    """The folder of the SD2 10x5 cases of scripts/p107_mask_rule_variants.py (its
    DEFAULT_ROOT and P107_CASES), relative to the repository root; the import of p107 leaves
    sys.argv as it was."""
    keep = sys.argv
    try:
        import p107_mask_rule_variants as P107
    finally:
        sys.argv = keep
    return os.path.join(P107.DEFAULT_ROOT, P107_CASES)


def check_p107(a, out=print):
    """Rollouts on SD2 10x5 under both masks against the cases of p107."""
    bad, tot = [], 0
    cases_root = p107_root()
    names = [nm for nm, _ in records(P107_SET, 'N')]
    n = a.n or len(names)
    for r in parse_list(a.combos, R8, 'combinations'):
        cases = []
        for nm in names[:n]:
            p = os.path.join(ROOT, cases_root, r, f'{nm}.json')
            if not os.path.exists(p):
                stop(f'missing case {os.path.relpath(p, ROOT)} of p107')
            with open(p) as f:
                cases.append(json.load(f))
        for mask in MASKS:
            arrays, got = rollout_combo(P107_SET, r, mask, n=n)
            if got != names[:n]:
                stop('the records of p107 and of this script are in another order')
            for q in RULES:
                want = np.array([c['adm'][mask]['rules'][q]['makespan'] for c in cases])
                v = arrays[q][:, 0]
                tot += len(v)
                if not np.array_equal(v, want):
                    bad.append(f'{r} {q} {mask}: {int((v == want).sum())} of {len(v)} equal')
    out(f'[p109] SD2 10x5 against the cases of p107 (both masks): {tot} makespans, '
        + ('all equal' if not bad else 'MISMATCH ' + '; '.join(bad[:8])))
    return tot, bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('cmd', choices=('plan', 'run', 'check-nondelay', 'check-p107', 'check-arrays'))
    ap.add_argument('--sets', default='', help='comma-separated sets (default: the seven test and '
                                               'sensitivity sets; the validation records only '
                                               'when named)')
    ap.add_argument('--combos', default='', help='comma-separated combinations (default: all eight)')
    ap.add_argument('--mask', default='active', choices=MASKS)
    ap.add_argument('--out-root', default='', help=f'root of the arrays (default {ACTIVE_ROOT})')
    ap.add_argument('--seed', type=int, default=None,
                    help=f'tie-break seed of the rules (default params seed_test, {STORED_SEED})')
    ap.add_argument('--ref-root', default='', help='check-arrays: root of the reference arrays')
    ap.add_argument('--budget_s', type=float, default=0.0,
                    help='run: start no job that would end after this many seconds')
    ap.add_argument('--n', type=int, default=0, help='checks: the first n records (0 = all)')
    a = ap.parse_args(argv)
    if a.cmd == 'plan':
        return plan(a)
    if a.cmd == 'run':
        return run(a)
    if a.cmd == 'check-nondelay':
        tot, bad = check_nondelay(a)
    elif a.cmd == 'check-arrays':
        tot, bad = check_arrays(a)
    else:
        if a.seed not in (None, STORED_SEED):
            stop(f'the cases of p107 are rolled out with the tie-break seed {STORED_SEED}')
        tot, bad = check_p107(a)
    if bad:
        raise SystemExit(1)
    return tot


if __name__ == '__main__':
    main()
