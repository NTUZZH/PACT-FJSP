"""What a rule change costs: one policy against one specialist per combination.

Three subcommands; every name and path comes from scripts/arm_registry.py.

checks   Time of the computed checks at a rule change. For each size (--size, default
         10x25,50x25n30,80x25n30) and combination (--combos) it loads the test descriptors
         data/FAMILY/<size>/desc/<R>/*.desc.json, their base instances and one stored
         schedule per instance, and times
           (a) the descriptor checks C1, C3 and C4 (check_schema_and_values,
               check_family_membership, check_magnitude_range of
               constraint_family/checks.py, in that order). run_all is not timed, because
               it also runs C2 and C5, which read declared facts, and computes the
               fingerprint of the C7 record, which belongs to the organizational step;
           (b) the schedule check C6 (check_schedule_validation) on the stored schedule.
         Each check is called once as a warm-up and then --repeats times (default 20); the
         time of an instance is the median of the timed calls. Schedules: --schedules
         policy (default) reads the policy's exported schedules
         (arm_registry.schedule_cache, --seed, default the selected seed); --schedules rule
         reads the best-rule schedules of or_solution/FAMILY/warmstart_cache_xmask_<size>,
         which exist before the campaign (exact environment only). The campaign exports
         policy schedules at 50x25n30 and 80x25n30 only, so a policy run names the sizes
         timed on the rule schedules with --rule_sizes (for example 10x25); the source of
         every size is stored in the record. A policy size without exported schedules is
         refused before anything is timed.
         Evidence as in scripts/p75_deploy_cost.py: one thread in every numerical library,
         idle-core samples before and after each size (p75.idle_check), cpu_share of the
         timed calls (CPU seconds over wall seconds), affinity mask and load average. A size
         is valid when its cores were idle before and after, cpu_share >= p75.SHARE_MIN,
         C1 and C3 pass and C6 passes on every schedule. The record
         results/rule_change_checks.json (arm_registry.out) is written in every case; an
         invalid record exits non-zero, and infer refuses it. --allow_busy (tests only)
         measures on busy cores and is stored in the record. An existing record is refused
         without --force.

count    What is kept under change control, counted from the registry and the disk, for one
         policy and for one specialist per combination in the active environment (--seed,
         default the selected seed of each arm and combination): models that serve the
         eight combinations, parameters per model (from the checkpoint; refused unless
         PARAMS), checkpoint bytes, training configuration files, and validation
         rollouts. Validating a model on a combination it serves is one greedy rollout on
         each validation instance of that combination; the validation set has vali_size
         instances of the configuration, capped by the trainer (train_family.py). The
         checks of Table tab:checks are listed by kind from CHECK_KINDS, which is compared
         with the identifiers and names run_all returns. Writes
         results/rule_change_count.json (arm_registry.out). A missing checkpoint,
         configuration or validation record is refused with the file named.

infer    Macros manuscript/macros_rulechange_cost.tex (prefix Rcc, letters only),
         results/rule_change_cost.json and results/rule_change_cost.md, all through
         arm_registry.out. Inputs: the training-cost records of scripts/run_training_cost.sh
         (--records, or every train_log/training_cost/*.json other than the copies
         <name>.first.json that training_cost_evidence.py rejudge keeps), the checks record,
         the count record, and the updates and episodes of the configurations of the
         repeated runs.
         An invalid or stub record, a record outside train_log/training_cost, two records of
         one arm and combination, and inputs of another environment are refused. The sum,
         median and ratios over the specialists are written only when all eight are
         measured (RccCatalogue...); otherwise only the measured ones and their number.
         --require all refuses when any input is missing; without it the blocks whose
         inputs exist are written and the skipped ones are listed. A refusal removes the
         three outputs.

Usage (repository root, one core, no GPU):
  PACT_ENV=exact PACT_OUT=<folder> taskset -c <core> python scripts/p99_rule_change_cost.py checks --rule_sizes 10x25
  PACT_ENV=exact PACT_OUT=<folder> taskset -c <core> python scripts/p99_rule_change_cost.py checks --schedules rule
  PACT_ENV=exact PACT_OUT=<folder> python scripts/p99_rule_change_cost.py count
  PACT_ENV=exact PACT_OUT=<folder> python scripts/p99_rule_change_cost.py infer [--require all]
"""
import argparse
import glob
import json
import math
import os
import re
import statistics
import sys
import time

NUMPY_VARS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')
if __name__ == '__main__':
    for _v in NUMPY_VARS:                      # one thread, before numpy is imported
        os.environ[_v] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''    # no CUDA context in this process

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import arm_registry as REG  # noqa: E402  (pops --pact-* options)
import training_cost_evidence as TCE  # noqa: E402

PARAMS = 32786
DEFAULT_SIZES = '10x25,50x25n30,80x25n30'
OUT_MACROS = 'manuscript/macros_rulechange_cost.tex'
OUT_JSON = 'results/rule_change_cost.json'
OUT_MD = 'results/rule_change_cost.md'
CHECKS_JSON = 'results/rule_change_checks.json'
COUNT_JSON = 'results/rule_change_count.json'
MACRO_RX = re.compile(r'^Rcc[A-Za-z]+$')
# Table tab:checks (manuscript/sections/s5_system.tex) by kind; tests compare the
# identifiers and names with those run_all returns.
CHECK_KINDS = (('C1', 'schema_and_values', 'computed'),
               ('C2', 'setup_convention', 'declared'),
               ('C3', 'family_membership', 'computed'),
               ('C4', 'magnitude_range', 'computed'),
               ('C5', 'applicability', 'declared'),
               ('C6', 'schedule_validation', 'computed'),
               ('C7', 'source_reconciliation', 'organizational'))
KIND_TEXT = {'computed': 'computed by code', 'declared': 'declared by the plant',
             'organizational': 'organizational procedure'}
DESC_CHECKS = ('check_schema_and_values', 'check_family_membership', 'check_magnitude_range')
SCHED_CHECK = 'check_schedule_validation'


class Refuse(SystemExit):
    pass


def refuse(why, outs=()):
    """Remove the outputs, so no stale or partial number survives, and stop."""
    for p in outs:
        if p and os.path.exists(p):
            os.remove(p)
            print(f'removed {p} (refused)')
    raise Refuse(f'*** REFUSING ***\n{why}')


# ================================================================== shared
def modules(size):
    return int(size.split('x')[0])


def size_word(size):
    import p75_deploy_cost as P75        # the words of \DeployLatency<Word>
    return P75.size_word(modules(size))


def hours2(x):
    return f'{float(x):.2f}'


def u1(x):
    return f'{float(x):.1f}'


def ms_decimals(spread_ms):
    """Decimals that put the last printed digit at the order of magnitude of the spread,
    from zero to three."""
    if not spread_ms or spread_ms <= 0:
        return 3
    return min(3, max(0, -math.floor(math.log10(spread_ms))))


def verify_check_kinds():
    """CHECK_KINDS against the identifiers and names run_all returns on a stored
    descriptor. Refuses on any difference."""
    from constraint_family import checks
    from constraint_family.descriptor import ConstraintDescriptor
    from ppvc_instance_generator import load_instance
    dp = sorted(glob.glob(os.path.join(ROOT, 'data', 'FAMILY', '10x25', 'desc', 'LSW',
                                       '*.desc.json')))[0]
    rec = json.load(open(dp))
    _, pt, _ = load_instance(os.path.join(ROOT, rec['base_instance']))
    got = [(r.id, r.name) for r in checks.run_all(
        ConstraintDescriptor.from_json(rec, setup_convention='attached'), pt)]
    want = [(i, n) for i, n, _ in CHECK_KINDS]
    if got != want:
        refuse(f'the checks of run_all {got} differ from CHECK_KINDS {want}')
    return got


# ================================================================== checks
def schedule_folder(source, size, seed):
    if source == 'rule':
        if REG.env() != 'exact':
            refuse('--schedules rule reads the exact-mask rule schedules '
                   'warmstart_cache_xmask_<size>; set PACT_ENV=exact')
        return f'or_solution/FAMILY/warmstart_cache_xmask_{size}'
    return REG.schedule_cache(seed, size)


def instance_pairs(size, combo, folder, n):
    """[(descriptor path, schedule path)] of the instances of a combination that have a
    stored schedule, at most n."""
    dps = sorted(glob.glob(os.path.join(ROOT, 'data', 'FAMILY', size, 'desc', combo,
                                        '*.desc.json')))
    if not dps:
        refuse(f'no descriptors under data/FAMILY/{size}/desc/{combo}')
    out = []
    for dp in dps:
        sp = os.path.join(ROOT, folder, combo, os.path.basename(dp)[:-len('.desc.json')] + '.npz')
        if os.path.isfile(sp):
            out.append((dp, sp))
        if n and len(out) >= n:
            break
    if not out:
        refuse(f'no stored schedule for {combo} at {size} under {folder}/{combo}/ '
               f'({REG.describe()})')
    return out, len(dps)


def time_calls(fn, repeats):
    """One warm-up call, then `repeats` timed calls: (seconds per call, CPU seconds and
    wall seconds of the timed calls, result of the last call)."""
    fn()
    ts = []
    c0, w0 = time.process_time(), time.perf_counter()
    out = None
    for _ in range(repeats):
        t = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t)
    return ts, time.process_time() - c0, time.perf_counter() - w0, out


def ms_stats(ts):
    import numpy as np
    ms = np.asarray(ts) * 1000.0
    return {'median': float(np.median(ms)),
            'iqr': float(np.percentile(ms, 75) - np.percentile(ms, 25)),
            'min': float(ms.min()), 'max': float(ms.max())}


def cmd_checks(a):
    import numpy as np
    import p75_deploy_cost as P75
    from constraint_family import checks
    from constraint_family.descriptor import ConstraintDescriptor
    from ppvc_instance_generator import load_instance

    REG.banner()
    out = os.path.join(ROOT, REG.out(CHECKS_JSON))
    if os.path.exists(out) and not a.force:
        refuse(f'{out} exists; pass --force to replace it')
    sizes = [s.strip() for s in a.size.split(',') if s.strip()]
    combos = [c.strip() for c in a.combos.split(',') if c.strip()]
    bad = [c for c in combos if c not in REG.COMBOS]
    if bad or a.repeats < 1:
        refuse(f'combinations {bad} or --repeats {a.repeats}')
    rule_sizes = [s.strip() for s in (a.rule_sizes or '').split(',') if s.strip()]
    if rule_sizes and a.schedules != 'policy':
        refuse('--rule_sizes applies to --schedules policy; --schedules rule already times '
               'every size on the rule schedules')
    if [s for s in rule_sizes if s not in sizes]:
        refuse(f'--rule_sizes {rule_sizes} must be among --size {sizes}')
    sources = {s: 'rule' if a.schedules == 'rule' or s in rule_sizes else 'policy'
               for s in sizes}
    seed = None
    if 'policy' in sources.values():
        try:
            seed = REG.select_seed('released')[0] if a.seed == 'select' else int(a.seed)
        except REG.MissingResult as e:
            refuse(str(e))
    folders = {s: schedule_folder(sources[s], s, seed) for s in sizes}
    absent = [s for s in sizes if sources[s] == 'policy'
              and not os.path.isdir(os.path.join(ROOT, folders[s]))]
    if absent:
        fix = ('time every size on the rule schedules with --schedules rule'
               if len(absent) == len(sizes) else
               f'time {",".join(absent)} on the rule schedules with --rule_sizes '
               f'{",".join(absent + rule_sizes)}')
        refuse(f'no exported policy schedules at {",".join(absent)} '
               f'({", ".join(folders[s] for s in absent)}). The campaign exports them at '
               f'50x25n30 and 80x25n30 after its evaluation; {fix}')
    plan = {s: {c: instance_pairs(s, c, folders[s], a.n) for c in combos} for s in sizes}
    cores = P75.affinity()
    rec = {'kind': 'rule_change_checks', 'generator': 'scripts/p99_rule_change_cost.py checks',
           'environment': REG.env(), 'schedules': a.schedules, 'policy_seed': seed,
           'schedule_sources': sources, 'schedule_folders': folders,
           'combinations': combos, 'repeats': a.repeats,
           'warmup_calls': 1,
           'timed': {'descriptor': list(DESC_CHECKS), 'schedule': SCHED_CHECK},
           'threads': {v: os.environ.get(v) for v in NUMPY_VARS},
           'affinity': cores, **P75.core_info(cores), 'hostname': os.uname().nodename,
           'started': time.strftime('%Y-%m-%d %H:%M:%S'), 'loadavg_start': P75.loadavg(),
           'allow_busy': bool(a.allow_busy), 'share_min': P75.SHARE_MIN, 'sizes': {}}
    if len(cores) != 1:
        print(f'note: affinity {cores}; the timed calls run on one thread', flush=True)
    valid = True
    for size in sizes:
        print(f'\n== {size} ({sources[size]} schedules, {folders[size]})', flush=True)
        before = P75.idle_check(cores, a.allow_busy, f'{size} before')
        rows, cpu, wall = [], 0.0, 0.0
        for combo in combos:
            pairs, n_desc = plan[size][combo]
            for dp, sp in pairs:
                drec = json.load(open(dp))
                desc = ConstraintDescriptor.from_json(drec, setup_convention='attached')
                jl, pt, _ = load_instance(os.path.join(ROOT, drec['base_instance']))
                z = np.load(sp, allow_pickle=False)
                amch = np.asarray(z['amch'], dtype=np.int64)
                ct = np.asarray(z['ct'], dtype=float)
                if amch.shape != (pt.shape[0],) or ct.shape != (pt.shape[0],):
                    refuse(f'{sp}: {amch.shape[0]} operations, the instance has {pt.shape[0]}')

                def dfn():
                    return (checks.check_schema_and_values(desc, pt, jl),
                            checks.check_family_membership(desc),
                            checks.check_magnitude_range(desc, pt))

                def sfn():
                    return checks.check_schedule_validation(desc, jl, pt, amch, ct)

                dts, dc, dw, dres = time_calls(dfn, a.repeats)
                sts, sc, sw, sres = time_calls(sfn, a.repeats)
                cpu += dc + sc
                wall += dw + sw
                rows.append({'combination': combo,
                             'instance': os.path.basename(dp)[:-len('.desc.json')],
                             'operations': int(pt.shape[0]),
                             'descriptor_ms': ms_stats(dts), 'schedule_ms': ms_stats(sts),
                             'C1': dres[0].status, 'C3': dres[1].status,
                             'C4': dres[2].status, 'C6': sres.status})
            print(f'  {size} {combo:4s} {len(pairs):3d} instances (of {n_desc} descriptors)'
                  f'  descriptor checks median '
                  f'{statistics.median(r["descriptor_ms"]["median"] for r in rows if r["combination"] == combo):.3f} ms'
                  f'  schedule check median '
                  f'{statistics.median(r["schedule_ms"]["median"] for r in rows if r["combination"] == combo):.3f} ms',
                  flush=True)
        after = P75.idle_check(cores, True, f'{size} after', record_only=True)
        share = cpu / wall if wall > 0 else 0.0
        reasons = []
        if not before['idle'] or not after['idle']:
            reasons.append('the cores were not idle before or after the size')
        if share < P75.SHARE_MIN:
            reasons.append(f'cpu_share {share:.3f} below {P75.SHARE_MIN}')
        for k, want in (('C1', ('PASS',)), ('C3', ('PASS',)), ('C6', ('PASS',))):
            badr = [f"{r['combination']}/{r['instance']}={r[k]}" for r in rows if r[k] not in want]
            if badr:
                reasons.append(f'{k} not PASS on {len(badr)} instance(s): {badr[:3]}')
        d_med = [r['descriptor_ms']['median'] for r in rows]
        s_med = [r['schedule_ms']['median'] for r in rows]
        S = {'instances': len(rows), 'modules': modules(size), 'schedules': sources[size],
             'descriptor_ms_median': statistics.median(d_med), 'descriptor_ms_max': max(d_med),
             'descriptor_spread_ms': statistics.median(r['descriptor_ms']['iqr'] for r in rows),
             'schedule_ms_median': statistics.median(s_med), 'schedule_ms_max': max(s_med),
             'schedule_spread_ms': statistics.median(r['schedule_ms']['iqr'] for r in rows),
             'c4_warn': sum(r['C4'] == 'WARN' for r in rows),
             'cpu_seconds': cpu, 'wall_seconds': wall, 'cpu_share': share,
             'idle_before': before, 'idle_after': after,
             'valid': not reasons, 'reasons': reasons, 'per_instance': rows}
        rec['sizes'][size] = S
        valid = valid and not reasons
        print(f"{size}: descriptor checks median {S['descriptor_ms_median']:.3f} ms (max "
              f"{S['descriptor_ms_max']:.3f}), schedule check median "
              f"{S['schedule_ms_median']:.3f} ms (max {S['schedule_ms_max']:.3f}), "
              f"cpu_share {share:.3f}, C4 warnings {S['c4_warn']}"
              + ('' if not reasons else '  INVALID: ' + '; '.join(reasons)), flush=True)
    rec['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
    rec['loadavg_end'] = P75.loadavg()
    rec['valid'] = valid
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump(rec, f, indent=1)
    print(f'wrote {out} (valid={valid})')
    if not valid:
        raise Refuse(f'INVALID measurement; {out} records it, and infer will refuse it')
    return 0


# ================================================================== count
def trainer_vali_cap():
    """The cap train_family.py puts on vali_size (self.vali_size = min(config.vali_size, k))."""
    src = open(os.path.join(ROOT, 'train_family.py')).read()
    m = re.search(r'self\.vali_size\s*=\s*min\(config\.vali_size,\s*(\d+)\)', src)
    if not m:
        refuse('the cap of the validation set was not found in train_family.py')
    return int(m.group(1))


def state_dict_numel(path):
    import torch
    sd = torch.load(path, map_location='cpu', weights_only=True)
    return int(sum(t.numel() for t in sd.values()))


def cmd_count(a):
    REG.banner()
    env = REG.env()
    out = os.path.join(ROOT, REG.out(COUNT_JSON))
    if os.path.exists(out):
        os.remove(out)
    kinds = verify_check_kinds()
    cap = trainer_vali_cap()

    def seed_of(arm, combo):
        return REG.select_seed(arm, combo)[0] if a.seed == 'select' else int(a.seed)

    try:
        serve = {'policy': {R: ('released', None, seed_of('released', None)) for R in REG.COMBOS},
                 'specialists': {R: ('specialist', R, seed_of('specialist', R))
                                 for R in REG.COMBOS}}
    except (REG.MissingResult, ValueError) as e:
        refuse(f'{e} ({REG.describe()})')
    sides = {}
    for side, m in serve.items():
        models = {}
        for R, (arm, c, s) in m.items():
            ck = REG.checkpoint(arm, c, s)
            models.setdefault(ck, {'arm': arm, 'combination': c, 'seed': s,
                                   'run': REG.model(arm, c, s), 'checkpoint': ck,
                                   'config': REG.config_path(arm, c, s), 'serves': []})
            models[ck]['serves'].append(R)
        for info in models.values():
            for p in (info['checkpoint'], info['config']):
                if not os.path.isfile(os.path.join(ROOT, p)):
                    refuse(f'missing {p} ({REG.describe()})')
            info['params'] = state_dict_numel(os.path.join(ROOT, info['checkpoint']))
            if info['params'] != PARAMS:
                refuse(f"{info['checkpoint']} holds {info['params']} parameters, want {PARAMS}")
            info['checkpoint_bytes'] = os.path.getsize(os.path.join(ROOT, info['checkpoint']))
            info['config_bytes'] = os.path.getsize(os.path.join(ROOT, info['config']))
            cfg = json.load(open(os.path.join(ROOT, info['config'])))
            info['vali_instances'] = min(int(cfg['vali_size']), cap)
        ms = list(models.values())
        sides[side] = {
            'models': len(ms), 'covers': sorted({R for x in ms for R in x['serves']},
                                                key=REG.COMBOS.index),
            'params_per_model': sorted({x['params'] for x in ms}),
            'checkpoint_bytes': sum(x['checkpoint_bytes'] for x in ms),
            'config_files': len({x['config'] for x in ms}),
            'config_bytes': sum(x['config_bytes'] for x in ms),
            'validation_pairs': sum(len(x['serves']) for x in ms),
            'validation_rollouts': sum(len(x['serves']) * x['vali_instances'] for x in ms),
            'validation_instances': sorted({x['vali_instances'] for x in ms}),
            'per_model': ms}
        if sides[side]['covers'] != list(REG.COMBOS):
            refuse(f'the {side} do not cover the eight combinations: {sides[side]["covers"]}')
    P, S = sides['policy'], sides['specialists']
    equal = (P['validation_pairs'] == S['validation_pairs']
             and P['validation_rollouts'] == S['validation_rollouts'])
    by_kind = {k: [i for i, _, kk in CHECK_KINDS if kk == k] for k in KIND_TEXT}
    rec = {'kind': 'rule_change_count', 'generator': 'scripts/p99_rule_change_cost.py count',
           'environment': env, 'seed_option': a.seed, 'policy': P, 'specialists': S,
           'validation_equal': equal, 'validation_cap_of_trainer': cap,
           'checks': [{'id': i, 'name': n, 'kind': KIND_TEXT[k]} for i, n, k in CHECK_KINDS],
           'checks_by_kind': by_kind, 'run_all_checks': kinds,
           'definitions': {
               'models': 'distinct checkpoints that serve the eight combinations',
               'params': 'summed numel of the tensors of the checkpoint state dict',
               'config_files': 'stored training configurations of these models',
               'validation_rollouts': 'one greedy rollout per validation instance of every '
                                      'combination a model serves; validation instances = '
                                      'vali_size of the configuration capped by the trainer'}}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump(rec, f, indent=1)
    print(f"policy: {P['models']} model(s), {P['checkpoint_bytes']} bytes, "
          f"{P['config_files']} configuration file(s), {P['validation_pairs']} model-combination "
          f"pairs, {P['validation_rollouts']} validation rollouts")
    print(f"specialists: {S['models']} model(s), {S['checkpoint_bytes']} bytes, "
          f"{S['config_files']} configuration file(s), {S['validation_pairs']} model-combination "
          f"pairs, {S['validation_rollouts']} validation rollouts")
    print('validation: ' + ('the two sides are equal' if equal else 'the two sides differ'))
    print('checks by kind: ' + '; '.join(f'{KIND_TEXT[k]} {",".join(v)}' for k, v in by_kind.items()))
    print(f'wrote {out}')
    return 0


# ================================================================== infer
def load_records(a, outs):
    """{('released', None) | ('specialist', R): (path, record, config)} of the usable
    training-cost records; refuses on any record that may not be used."""
    root = os.path.abspath(a.root)
    if a.records:
        paths = [os.path.abspath(p) for p in a.records]
    else:
        d = a.records_dir or os.path.join(root, TCE.RECORD_DIR)
        # <name>.first.json is the copy that rejudge keeps of a record, not a record of its own
        paths = [p for p in sorted(glob.glob(os.path.join(d, '*.json')))
                 if not p.endswith(TCE.FIRST_SUFFIX)]
    got = {}
    for p in paths:
        try:
            rec = json.load(open(p))
        except (OSError, ValueError) as e:
            refuse(f'{p}: {e}', outs)
        probs = TCE.check_record(rec)
        if probs:
            refuse(f'{p} may not be used: ' + '; '.join(probs), outs)
        if rec.get('environment') != REG.env():
            refuse(f"{p} was measured in the {rec.get('environment')} environment, this "
                   f'reading is in the {REG.env()} environment (mixed environments)', outs)
        r = rec['repeats']
        key = (r['arm'], r.get('combination'))
        want = REG.model(r['arm'], r.get('combination'), r['seed'])
        if r['run'] != want:
            refuse(f"{p} repeats {r['run']}, the registry names {want}", outs)
        if key in got:
            refuse(f'{p} and {got[key][0]} both measure {key}; keep one', outs)
        cp = os.path.join(root, rec['files'].get('config', ''))
        if not rec['files'].get('config') or not os.path.isfile(cp):
            refuse(f"{p}: the configuration of the repeated run is missing "
                   f"({rec['files'].get('config')})", outs)
        cfg = json.load(open(cp))
        if int(cfg['max_updates']) != rec['updates']:
            refuse(f"{p}: {rec['updates']} updates, its configuration {cfg['max_updates']}", outs)
        got[key] = (p, rec, cfg)
    gpus = {v[1]['host'].get('gpu') for v in got.values()}
    if len(gpus) > 1:
        refuse(f'records from different GPUs {sorted(map(str, gpus))}', outs)
    return got


def load_json(path, kind, outs):
    if not os.path.isfile(path):
        return None
    rec = json.load(open(path))
    if rec.get('kind') != kind:
        refuse(f"{path}: kind {rec.get('kind')!r}, want {kind!r}", outs)
    if rec.get('environment') != REG.env():
        refuse(f"{path} was written in the {rec.get('environment')} environment, this reading "
               f'is in the {REG.env()} environment (mixed environments)', outs)
    return rec


def block_training(got, L, V):
    pol = got.get(('released', None))
    specs = {R: got[('specialist', R)] for R in REG.COMBOS if ('specialist', R) in got}
    if pol:
        _, rec, cfg = pol
        h = rec['wall_seconds'] / 3600.0
        V['policy_hours'] = h
        L.append(('RccPolicyHours', hours2(h), f"h, wall clock of {rec['name']} "
                  f"(repeats {rec['repeats']['run']}), GPU {rec['host'].get('gpu')}"))
        L.append(('RccPolicySecPerUpdate', f"{rec['seconds_per_update']:.2f}",
                  's per update, mean over the updates'))
        L.append(('RccPolicyUpdates', str(int(cfg['max_updates'])), 'updates'))
        L.append(('RccPolicyEpisodes', str(int(cfg['max_updates']) * int(cfg['num_envs'])),
                  f"episodes = updates x {cfg['num_envs']} environments"))
    for R, (_, rec, _) in specs.items():
        L.append((f'RccSpecHours{R}', hours2(rec['wall_seconds'] / 3600.0),
                  f"h, wall clock of {rec['name']} (repeats {rec['repeats']['run']})"))
        L.append((f'RccSpecSecPerUpdate{R}', f"{rec['seconds_per_update']:.2f}", 's per update'))
    if specs:
        L.append(('RccSpecMeasuredN', str(len(specs)), 'specialists with a valid record, of 8'))
        upd = {int(c['max_updates']) for _, _, c in specs.values()}
        eps = {int(c['max_updates']) * int(c['num_envs']) for _, _, c in specs.values()}
        if len(upd) == 1 and len(eps) == 1:
            L.append(('RccSpecUpdates', str(upd.pop()), 'updates of one specialist'))
            L.append(('RccSpecEpisodes', str(eps.pop()), 'episodes of one specialist'))
        else:
            for R, (_, _, c) in specs.items():
                L.append((f'RccSpecUpdates{R}', str(int(c['max_updates'])), 'updates'))
    if len(specs) == len(REG.COMBOS):
        hs = [specs[R][1]['wall_seconds'] / 3600.0 for R in REG.COMBOS]
        us = [int(specs[R][2]['max_updates']) for R in REG.COMBOS]
        V['catalogue_hours'] = sum(hs)
        L.append(('RccCatalogueHours', hours2(sum(hs)), 'h, sum over the eight specialists'))
        L.append(('RccSpecMedianHours', hours2(statistics.median(hs)),
                  'h, median over the eight specialists'))
        L.append(('RccCatalogueUpdates', str(sum(us)), 'updates, sum over the eight specialists'))
        srt = sorted(hs)
        L.append(('RccSpecMinHours', hours2(srt[0]), 'h, smallest of the eight specialists'))
        L.append(('RccSpecMaxHours', hours2(srt[-1]), 'h, largest of the eight specialists'))
        if pol:
            # how many specialists train as long as the policy: every set of `above`
            # specialists trains longer than the policy (the shortest runs are summed), and
            # every set of `below` specialists trains for less time (the longest are summed)
            ph, n = V['policy_hours'], len(srt)
            above = next((k for k in range(1, n + 1) if sum(srt[:k]) > ph), None)
            below = max((k for k in range(1, n + 1) if sum(srt[n - k:]) < ph), default=0)
            if above is not None:
                L.append(('RccSpecSetAbovePolicyN', str(above),
                          'smallest count of specialists of which every set trains longer '
                          'than the policy'))
            L.append(('RccSpecSetBelowPolicyN', str(below),
                      'largest count of specialists of which every set trains for less time '
                      'than the policy'))
            L.append(('RccCatalogueOverPolicyHours', u1(sum(hs) / V['policy_hours']),
                      'eight specialists over the policy, training hours'))
            L.append(('RccCatalogueOverPolicyUpdates',
                      u1(sum(us) / int(pol[2]['max_updates'])),
                      'eight specialists over the policy, updates'))
    used = ([pol[1]] if pol else []) + [rec for _, rec, _ in specs.values()]
    means = [(rec.get('samples') or {}).get('foreign_gpu_util_mean') for rec in used]
    if used and all(isinstance(m, (int, float)) for m in means):
        # evidence that each run had the GPU to itself: the largest run-average of the
        # GPU use of the other processes, against the limit of the record
        L.append(('RccMeasuredRuns', str(len(used)), 'training runs with a valid record'))
        L.append(('RccGpuOtherMeanMax', f'{max(means):.2f}',
                  '%, largest average GPU use of the other processes over one run'))
        L.append(('RccGpuOtherMeanLimit', f'{TCE.FOREIGN_UTIL_MEAN_MAX:g}',
                  '%, limit of that average for a valid record'))
    return bool(pol), specs


def block_checks(C, L):
    names = []
    for size, S in C['sizes'].items():
        w = size_word(size)
        for q, stem in (('descriptor', 'DescChecks'), ('schedule', 'ScheduleCheck')):
            d = ms_decimals(S[f'{q}_spread_ms'])
            L.append((f'Rcc{stem}Ms{w}', f"{S[f'{q}_ms_median']:.{d}f}",
                      f"ms, median over {S['instances']} instances at {size}, "
                      f"{d} decimals for a spread of {S[f'{q}_spread_ms']:.4f} ms"))
            L.append((f'Rcc{stem}MaxMs{w}', f"{S[f'{q}_ms_max']:.{d}f}",
                      f'ms, largest instance time at {size}'))
        L.append((f'RccChecksInstances{w}', str(S['instances']), f'instances timed at {size}'))
        names.append(size)
    return names


def block_count(K, L):
    P, S = K['policy'], K['specialists']
    if P['params_per_model'] != [PARAMS] or S['params_per_model'] != [PARAMS]:
        refuse(f"parameters per model {P['params_per_model']}, {S['params_per_model']}")
    L.append(('RccPolicyModels', str(P['models']), 'models that serve the eight combinations'))
    L.append(('RccSpecModels', str(S['models']), 'models that serve the eight combinations'))
    L.append(('RccParams', str(PARAMS), 'parameters per model, read from every checkpoint'))
    L.append(('RccPolicyCheckpointKB', f"{P['checkpoint_bytes'] / 1024.0:.0f}", 'KiB'))
    L.append(('RccSpecCheckpointKB', f"{S['checkpoint_bytes'] / 1024.0:.0f}",
              'KiB, sum over the eight checkpoints'))
    L.append(('RccPolicyConfigFiles', str(P['config_files']), 'training configurations'))
    L.append(('RccSpecConfigFiles', str(S['config_files']), 'training configurations'))
    if len(P['validation_instances']) == 1 and P['validation_instances'] == S['validation_instances']:
        L.append(('RccValiInstances', str(P['validation_instances'][0]),
                  'validation instances per combination'))
    L.append(('RccPolicyValiPairs', str(P['validation_pairs']), 'model-combination pairs'))
    L.append(('RccSpecValiPairs', str(S['validation_pairs']), 'model-combination pairs'))
    L.append(('RccPolicyValiRollouts', str(P['validation_rollouts']), 'validation rollouts'))
    L.append(('RccSpecValiRollouts', str(S['validation_rollouts']), 'validation rollouts'))
    by = K['checks_by_kind']
    L.append(('RccChecksComputed', str(len(by['computed'])), ','.join(by['computed'])))
    L.append(('RccChecksDeclared', str(len(by['declared'])), ','.join(by['declared'])))
    L.append(('RccChecksOrganizational', str(len(by['organizational'])),
              ','.join(by['organizational'])))


def collision_check(names, out_mac, outs):
    rx = re.compile(r'\\(?:newcommand|renewcommand|providecommand|def)\s*\{?\\([A-Za-z]+)')
    files = sorted(glob.glob(os.path.join(ROOT, 'manuscript', 'macros_*.tex')) +
                   glob.glob(os.path.join(ROOT, 'paper', 'macros_*.tex')) +
                   [os.path.join(ROOT, d, 'macros.tex') for d in ('manuscript', 'paper')])
    # the file this run writes and its copy in the manuscript folder are the same macros
    own = {os.path.abspath(out_mac),
           os.path.abspath(os.path.join(ROOT, 'manuscript', os.path.basename(out_mac)))}
    for f in files:
        if not os.path.exists(f) or os.path.abspath(f) in own:
            continue
        clash = set(names) & set(rx.findall(open(f, errors='replace').read()))
        if clash:
            refuse(f'macro name collision with {os.path.relpath(f, ROOT)}: {sorted(clash)[:5]}',
                   outs)


def cmd_infer(a):
    REG.banner()
    env = REG.env()
    outs = [os.path.join(ROOT, REG.out(p)) for p in (OUT_MACROS, OUT_JSON, OUT_MD)]
    for p in outs:                       # a refusal below leaves no stale output
        if os.path.exists(p):
            os.remove(p)
    got = load_records(a, outs)
    C = load_json(os.path.join(ROOT, REG.out(CHECKS_JSON)), 'rule_change_checks', outs)
    K = load_json(os.path.join(ROOT, REG.out(COUNT_JSON)), 'rule_change_count', outs)
    if C is not None and C.get('valid') is not True:
        refuse(f'{CHECKS_JSON} is invalid: ' + '; '.join(
            f"{s}: {', '.join(v['reasons'])}" for s, v in C['sizes'].items() if v['reasons']), outs)
    missing = []
    if ('released', None) not in got:
        missing.append('training record of the policy')
    miss_spec = [R for R in REG.COMBOS if ('specialist', R) not in got]
    if miss_spec:
        missing.append('training records of the specialists ' + ','.join(miss_spec))
    if C is None:
        missing.append(CHECKS_JSON)
    elif [s for s in DEFAULT_SIZES.split(',') if s not in C['sizes']]:
        missing.append('checks at the sizes ' + ','.join(
            s for s in DEFAULT_SIZES.split(',') if s not in C['sizes']))
    if K is None:
        missing.append(COUNT_JSON)
    if a.require == 'all' and missing:
        refuse('--require all: missing ' + '; '.join(missing), outs)

    L, V, written, skipped = [], {}, [], []
    has_pol, specs = block_training(got, L, V)
    if has_pol or specs:
        written.append(f"training ({'policy' if has_pol else 'no policy'}, "
                       f'{len(specs)} of 8 specialists)')
    else:
        skipped.append('training: no usable training-cost record')
    if len(specs) < len(REG.COMBOS):
        skipped.append(f'catalogue sum, median and ratios: {len(specs)} of 8 specialists measured')
    if C is not None:
        written.append('checks at ' + ', '.join(block_checks(C, L)))
    else:
        skipped.append(f'checks: no {CHECKS_JSON}')
    if K is not None:
        block_count(K, L)
        written.append('count')
    else:
        skipped.append(f'count: no {COUNT_JSON}')
    if not L:
        refuse('no input of any block exists: ' + '; '.join(skipped), outs)
    names = [n for n, _, _ in L]
    bad = [n for n in names if not MACRO_RX.match(n)]
    if bad or len(names) != len(set(names)):
        refuse(f'macro names must be Rcc plus letters and unique: {bad or names}', outs)
    collision_check(names, outs[0], outs)

    import macro_io
    head = ['% Generated by scripts/p99_rule_change_cost.py -- do not hand-edit',
            f'% What a rule change costs, {env} environment. Training hours: wall clock of one '
            'run with the GPU and its cores to itself (scripts/run_training_cost.sh); '
            'check times: one thread on one reserved core (results/rule_change_checks.json).',
            '% Blocks written: ' + '; '.join(written) + '.',
            *(['% Blocks skipped: ' + '; '.join(skipped) + '.'] if skipped else [])]
    macro_io.emit(outs[0], head + [f'\\newcommand{{\\{n}}}{{{v}}}  % {c}' for n, v, c in L])
    res = {'generator': 'scripts/p99_rule_change_cost.py infer', 'environment': env,
           'written': written, 'skipped': skipped, 'missing': missing,
           'macros': {n: v for n, v, _ in L},
           'records': {f'{k[0]}{"" if k[1] is None else " " + k[1]}':
                       {'file': os.path.relpath(p, os.path.abspath(a.root)), 'name': r['name'],
                        'repeats': r['repeats']['run'], 'hours': r['wall_seconds'] / 3600.0,
                        'seconds_per_update': r['seconds_per_update'],
                        'updates': int(c['max_updates']),
                        'episodes': int(c['max_updates']) * int(c['num_envs']),
                        'done_in_hours': r.get('done_in_hours'), 'gpu': r['host'].get('gpu')}
                       for k, (p, r, c) in got.items()},
           'checks_schedules': (C.get('schedule_sources') or C.get('schedules')) if C else None,
           'count_validation_equal': K.get('validation_equal') if K else None}
    with open(outs[1], 'w') as f:
        json.dump(res, f, indent=1)
    md = ['# What a rule change costs (generated by scripts/p99_rule_change_cost.py infer)', '',
          f'Environment: {env}.', '', 'Blocks written: ' + '; '.join(written) + '.']
    if skipped:
        md += ['', 'Blocks skipped: ' + '; '.join(skipped) + '.']
    if K is not None:
        md += ['', 'Validation of the models on the combinations they serve: '
               + ('the two sides are equal.' if K['validation_equal'] else 'the two sides differ.')]
    if C is not None:
        src = C.get('schedule_sources') or {s: C['schedules'] for s in C['sizes']}
        md += ['', 'Schedules of the C6 timing: '
               + ', '.join(f'{s} {v}' for s, v in src.items()) + '.']
    md += ['', '| macro | value | meaning |', '|---|---|---|']
    md += [f'| `\\{n}` | {v} | {c} |' for n, v, c in L]
    with open(outs[2], 'w') as f:
        f.write('\n'.join(md) + '\n')
    print('\n'.join(md))
    print(f'\nwrote {outs[1]}, {outs[2]}')
    return 0


# ================================================================== main
def parse(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest='command', required=True)
    c = sub.add_parser('checks')
    c.add_argument('--size', default=DEFAULT_SIZES)
    c.add_argument('--combos', default=','.join(REG.COMBOS))
    c.add_argument('--n', type=int, default=None,
                   help='instances per combination (default every one with a schedule)')
    c.add_argument('--repeats', type=int, default=20)
    c.add_argument('--schedules', choices=('policy', 'rule'), default='policy')
    c.add_argument('--seed', default='select', help="policy schedules: seed or 'select'")
    c.add_argument('--rule_sizes', default='',
                   help='with --schedules policy: sizes timed on the rule schedules')
    c.add_argument('--allow_busy', action='store_true', help='tests only')
    c.add_argument('--force', action='store_true')
    k = sub.add_parser('count')
    k.add_argument('--seed', default='select', help="training seed of every model, or 'select'")
    i = sub.add_parser('infer')
    i.add_argument('--records', nargs='*', default=None)
    i.add_argument('--records_dir', default=None)
    i.add_argument('--root', default=ROOT, help='root the record paths are relative to')
    i.add_argument('--require', choices=('all',), default=None)
    return ap.parse_args(argv)


def main(argv=None):
    a = parse(sys.argv[1:] if argv is None else argv)
    sys.argv = sys.argv[:1]          # params.py (imported by the environment code) parses argv
    os.chdir(ROOT)
    try:
        return {'checks': cmd_checks, 'count': cmd_count, 'infer': cmd_infer}[a.command](a)
    except Refuse as e:
        print(e, file=sys.stderr)
        return 2
    except REG.MissingResult as e:
        print(f'*** REFUSING ***\n{e}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
