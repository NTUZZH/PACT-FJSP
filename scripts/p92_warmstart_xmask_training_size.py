r"""Exact-mask warm-start cache of the training size (10 modules).

Builds or_solution/FAMILY/warmstart_cache_xmask_10x25/ for instance_000 to
instance_029 of the eight combinations of data/FAMILY/10x25 (240 cells), in
the format of the production-size caches
or_solution/FAMILY/warmstart_cache_xmask_<n>x25n30 (scripts/
p86_extend_scale_sets.py wsmasked):

  <combination>/instance_<iii>.npz  amch, ct, ms, rule, rule_names, rule_ms
  manifest.json                     per cell: the chosen rule, its makespan
                                    and the four makespans
  dataset_meta.json                 set, rule set, mask, rules, procedure,
                                    tie rule, file format, generator

The construction is the code of p86, imported and not copied:
  p86.masked_rollout          FJSPEnvFamily(J, M, family_mode=True,
                              first_mask='exact', mask_ties='exact') with no
                              type lists, numpy seed 50 right before the
                              rollout, common_utils.masked_rule_action
  p86._eligible               reference_sim.validate_schedule (eligibility)
                              and the C6 status
  p86.load_stored_descriptor  the descriptor, attached setup convention
  best of four                the shortest eligible makespan; equal makespans
                              go to the earlier rule of FIFO, MOR, SPT, MWKR
                              (first argmin, as in p86 wsmasked and p12)
One difference from the production build: there the SPT schedules came from
the earlier exact-mask build with heuristic_select_action (the same rule);
here all four rules, SPT included, are rolled out with masked_rule_action,
the path p86 wsmasked takes for SPT wherever SPT was not the chosen rule.

Checks, printed; the cache is written only when (a) to (c) pass:
  (a) the makespan of every rule schedule equals column 0 of
      test_results/FAMILY/10x25_xmask/Result_<RULE>_<combination>.npy at that
      instance (eval_family.py --rule_set masked under the exact mask; shape
      (100, 2), column 0 the makespan, column 1 the rollout seconds). A
      difference is reported (rule, cells, amount) and stops the build.
  (b) every rule schedule: reference_sim.validate_schedule empty, C6 PASS,
      constraint_family.cpsat_arms.validate_all empty
  (c) every rule call of every rollout: env.mask_setting('first_mask') and
      env.mask_setting('mask_ties') read 'exact' and env._first_mask is set
  (d) after writing: 242 files; every npz re-read and checked with
      cpsat_arms.load_hint_npz + validate_all, with the memetic runner's own
      loader (eval_memetic_family.load_stored_schedule) and against the
      manifest; a sha256 list written next to the cache
      (or_solution/FAMILY/warmstart_cache_xmask_10x25.sha256, paths from the
      repository root, `sha256sum -c` from there) and re-checked.
The folder is assembled under a temporary sibling name and renamed only when
every check has passed, so a failed build leaves no partial cache.

Usage (repository root, one core, a few minutes):
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
    python scripts/p92_warmstart_xmask_training_size.py [--check-only]
    [--instances 0-29] [--stage DIR] [--force] [--max-minutes 20]
--check-only builds and checks and writes nothing but the stage (any
--instances allowed); a written cache always holds instances 0-29. --stage
keeps each rule rollout (amch, ct, ms, the mask check of the rollout) in DIR
and reads it back in a later call, so the build can be split, e.g.
--check-only --instances 0-14 --stage DIR, then --stage DIR; the checks (a),
(b) and the re-read run in every call on every schedule, read back or not,
and a stage made with other code (sha256 of p92, p86, common_utils,
fjsp_env_family, reference_sim, checks) is refused. --force moves an existing
cache aside (<folder>.replaced_<time>) instead of refusing.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time

_ARGV = sys.argv[1:]            # p86 strips sys.argv at import (params.py parses it)
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
os.environ['CUDA_VISIBLE_DEVICES'] = ''

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, ROOT)
import p86_extend_scale_sets as p86  # noqa: E402  (chdir to ROOT, configs.device = cpu)
import numpy as np  # noqa: E402

SIZE = '10x25'
SET = f'data/FAMILY/{SIZE}'
BASE = 'data/PPVC/10x25+ppvc-mixed'
CACHE = f'or_solution/FAMILY/warmstart_cache_xmask_{SIZE}'
SUMS = CACHE + '.sha256'
STORED = f'test_results/FAMILY/{SIZE}_xmask'
LEGACY = 'or_solution/FAMILY/warmstart_cache'      # classical rules, legacy mask (p12, p44)
N_SET = 100                                        # test instances of the set
N_CACHE = 30                                       # instance_000 to instance_029
RULES = p86.RULE_ORDER                             # FIFO, MOR, SPT, MWKR
COMBOS = p86.REGIMES_ALL                           # N, L, S, W, LS, LW, SW, LSW


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--instances', default=f'0-{N_CACHE - 1}')
    ap.add_argument('--check-only', action='store_true')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--max-minutes', type=float, default=20.0,
                    help='stop after the first instance when the estimate is above this')
    ap.add_argument('--stage', default=None,
                    help='folder of the rule rollouts, so that the build can run in '
                         'several calls (outside data/, or_solution/, test_results/)')
    a = ap.parse_args(argv)
    if a.stage:
        a.stage = os.path.abspath(a.stage)
        for d in ('data', 'or_solution', 'test_results', 'train_log', 'trained_network'):
            if p86._inside(a.stage, os.path.join(ROOT, d)):
                ap.error(f'--stage inside {d}/')
    lo, _, hi = a.instances.partition('-')
    a.idx = list(range(int(lo), int(hi or lo) + 1))
    if not a.check_only and a.idx != list(range(N_CACHE)):
        ap.error(f'a written cache holds instances 0-{N_CACHE - 1}; use --check-only '
                 f'for another range')
    if max(a.idx) >= N_SET:
        ap.error(f'the set has {N_SET} instances')
    return a


def meta_text():
    """dataset_meta.json: the keys of p86.masked_meta_text, stated for this
    size, this instance range and this procedure."""
    d = {'set': SET, 'base_dir': BASE, 'n_instances': N_CACHE,
         'instances': (f'instance_000 to instance_{N_CACHE - 1:03d} of the {N_SET} test '
                       f'instances of {SET} (the instances of the timed campaign at the '
                       f'training size)'),
         'regimes': list(COMBOS), 'rule_set': 'masked',
         'first_mask': 'exact', 'mask_ties': 'exact',
         'rules': list(RULES),
         'procedure': ('best of FIFO, MOR, SPT and MWKR, each rolled out with '
                       'common_utils.masked_rule_action through FJSPEnvFamily '
                       "(first_mask='exact', mask_ties='exact'; numpy seed 50 "
                       'right before each rollout, as in '
                       'scripts/p12_cpsat_fairwarm.py _rollout) by '
                       'scripts/p86_extend_scale_sets.py masked_rollout, and checked '
                       'with constraint_family.reference_sim.validate_schedule; all '
                       'four rules, SPT included, rolled out with the masked rule '
                       'set; every rule makespan equal to '
                       f'{STORED}/Result_<RULE>_<regime>.npy (eval_family.py '
                       '--rule_set masked, exact mask)'),
         'tie_rule': ('the shortest valid makespan; equal makespans go to the '
                      'earlier rule in the order FIFO, MOR, SPT, MWKR (the rule '
                      'of p12_cpsat_fairwarm._warm_start)'),
         'files': ('<regime>/<instance>.npz with amch, ct, ms, rule, rule_names, '
                   'rule_ms (the format of p12_cpsat_fairwarm._warm_start); '
                   'manifest.json lists the chosen rule and the four makespans'),
         'generator': ('scripts/p92_warmstart_xmask_training_size.py (the rollout, '
                       'eligibility and tie rule of scripts/p86_extend_scale_sets.py '
                       'wsmasked)')}
    return json.dumps(d, indent=1)


def load_cell(r, i):
    name = p86.inst(i)
    with open(os.path.join(SET, 'desc', r, name + '.desc.json')) as f:
        rec = json.load(f)
    stem = rec['base_instance']
    if stem != os.path.join(BASE, name):
        sys.exit(f'{SET} {r}/{name}: base_instance {stem!r}, expected {BASE}/{name}')
    jl, pt, _ = p86.load_instance(stem)
    return jl, pt, p86.load_stored_descriptor(rec)


def stored_arrays():
    """{(rule, combination): makespans of the 100 test instances}; checks that
    each array is (100, 2) with integer makespans and that its mean equals the
    run's own summary file."""
    means = {}
    with open(os.path.join(STORED, 'summary_rules_xmask.md')) as f:
        for line in f:
            p = [x.strip() for x in line.split('|')]
            if len(p) > 4 and p[1] in COMBOS and p[2] in RULES:
                means[(p[2], p[1])] = float(p[3])
    out = {}
    for rule in RULES:
        for r in COMBOS:
            a = np.load(os.path.join(STORED, f'Result_{rule}_{r}.npy'))
            if a.shape != (N_SET, 2) or not np.all(a[:, 0] == np.round(a[:, 0])):
                sys.exit(f'{STORED}/Result_{rule}_{r}.npy: shape {a.shape}, not (100, 2) '
                         f'with integer makespans in column 0')
            if abs(round(float(a[:, 0].mean()), 1) - means[(rule, r)]) > 1e-9:
                sys.exit(f'{STORED}/Result_{rule}_{r}.npy: mean {a[:, 0].mean():.2f}, '
                         f'summary {means[(rule, r)]}')
            out[(rule, r)] = a[:, 0]
    return out, len(means)


STAGE_CODE = ('scripts/p92_warmstart_xmask_training_size.py', 'scripts/p86_extend_scale_sets.py',
              'common_utils.py', 'constraint_family/fjsp_env_family.py',
              'constraint_family/reference_sim.py', 'constraint_family/checks.py')


def open_stage(stage):
    """A stage folder holds one npz per rule rollout (amch, ct, ms and the mask
    check of that rollout), as p86 wsmasked keeps its rule rollouts in its
    scratch folder, so that the build can run in several calls. It records the
    sha256 of the code it was made with and is refused under other code."""
    code = {}
    for f in STAGE_CODE:
        with open(os.path.join(ROOT, f), 'rb') as fh:
            code[f] = hashlib.sha256(fh.read()).hexdigest()
    mark = os.path.join(stage, 'STAGE.json')
    if os.path.exists(mark):
        with open(mark) as f:
            old = json.load(f)
        if old != code:
            sys.exit(f'REFUSING stage {stage}: made with other code '
                     f'({[k for k in code if old.get(k) != code[k]]}); use a new folder')
    else:
        os.makedirs(stage, exist_ok=True)
        if os.listdir(stage):
            sys.exit(f'REFUSING stage {stage}: not empty and no STAGE.json')
        with open(mark, 'w') as f:
            json.dump(code, f, indent=1)


def build(idx, max_minutes, stage=None):
    """Every rule rollout and check of the cells. Returns the cells and the
    check counts; nothing is written here except the stage files."""
    from constraint_family import cpsat_arms as A
    import common_utils
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass
    if stage:
        open_stage(stage)
    original = common_utils.masked_rule_action
    seen = {}

    def rollout(rule, r, i, jl, pt, desc):
        """(amch, ct, ms, rule calls, mask exact at every call, loaded)."""
        p = os.path.join(stage, rule, r, p86.inst(i) + '.npz') if stage else None
        if p and os.path.exists(p):
            z = np.load(p)
            return (z['amch'], z['ct'], float(z['ms']), int(z['calls']),
                    bool(z['mask_ok']), True)
        seen.update(calls=0, bad=[])
        amch, ct, ms = p86.masked_rollout(rule, jl, pt, desc)
        mask_ok = seen['calls'] > 0 and not seen['bad']
        if p:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            np.savez(p + '.tmp.npz', amch=amch, ct=ct, ms=ms, calls=seen['calls'],
                     mask_ok=mask_ok)
            os.replace(p + '.tmp.npz', p)
        return amch, ct, ms, seen['calls'], mask_ok, False

    def recording(method, env):
        # every rule call: the mask settings the environment uses at that step
        seen['calls'] += 1
        s = (env.mask_setting('first_mask'), env.mask_setting('mask_ties'),
             bool(getattr(env, '_first_mask', False)))
        if s != ('exact', 'exact', True):
            seen['bad'].append(s)
        return original(method, env)

    # masked_rollout imports masked_rule_action from common_utils at call time
    common_utils.masked_rule_action = recording
    cells, n_exact, n_calls, invalid, n_loaded = {}, 0, 0, [], 0
    t0 = time.perf_counter()
    try:
        for k, i in enumerate(idx):
            for r in COMBOS:
                name = p86.inst(i)
                jl, pt, desc = load_cell(r, i)
                case = A.load_case(SIZE, r, name)
                cand = {}
                for rule in RULES:
                    amch, ct, ms, calls, mask_ok, loaded = rollout(rule, r, i, jl, pt, desc)
                    n_loaded += loaded
                    # checked here on every call, also for a schedule read from the stage
                    ok, c6 = p86._eligible(jl, pt, desc, amch, ct)
                    v = A.validate_all(case, *A.schedule_from_hint(case, amch, ct))
                    n_exact += mask_ok
                    n_calls += calls
                    if not ok or c6 != 'PASS' or v:
                        invalid.append(f'{rule} {r}/{name}: reference valid {ok}, C6 {c6}, '
                                       f'validate_all {v[:1]}')
                    cand[rule] = (amch, ct, ms, ok)
                rule_ms = [cand[x][2] if cand[x][3] else float('inf') for x in RULES]
                kb = int(np.argmin(rule_ms))             # first minimum: p12's tie rule
                if not np.isfinite(rule_ms[kb]):
                    sys.exit(f'{r}/{name}: no valid masked rule schedule')
                cells[(r, i)] = dict(cand=cand, rule_ms=rule_ms, best=RULES[kb])
            el = time.perf_counter() - t0
            est = el / (k + 1) * len(idx)
            print(f'[p92] {p86.inst(i)} done: {(k + 1) * len(COMBOS) * len(RULES)} rollouts '
                  f'({n_loaded} read from the stage), {el:.0f} s, estimate for {len(idx)} '
                  f'instances {est / 60:.1f} min', flush=True)
            if k == 0 and est > 60 * max_minutes:
                sys.exit(f'STOP: estimate {est / 60:.1f} min is above {max_minutes:g} min')
    finally:
        common_utils.masked_rule_action = original
    return cells, dict(n_exact=n_exact, n_calls=n_calls, invalid=invalid, n_loaded=n_loaded,
                       seconds=time.perf_counter() - t0)


def check_stored(cells, stored):
    """Check (a): rule makespans against the evaluator's arrays."""
    diff, n = [], 0
    for (r, i), c in cells.items():
        for rule in RULES:
            n += 1
            got, want = c['cand'][rule][2], float(stored[(rule, r)][i])
            if abs(got - want) > 1e-9:
                diff.append((rule, r, i, got, want))
    return n, diff


def write_cache(cells, force):
    """Assemble the folder under a temporary sibling, re-read and check every
    file there, then rename it to CACHE."""
    from constraint_family import cpsat_arms as A
    from baselines.memetic.instance import Instance
    import eval_memetic_family as EM
    if os.path.exists(CACHE) and not force:
        sys.exit(f'REFUSING: {CACHE} exists (pass --force to move it aside)')
    tmp = f'{CACHE}.building_{os.getpid()}'
    if os.path.exists(tmp):
        sys.exit(f'REFUSING: {tmp} exists')
    os.makedirs(tmp)
    try:
        manifest = {}
        for r in COMBOS:                                   # regime-major, as p86 wsmasked
            os.makedirs(os.path.join(tmp, r))
            for i in range(N_CACHE):
                c, name = cells[(r, i)], p86.inst(i)
                b = c['best']
                amch, ct, _, _ = c['cand'][b]
                ms = c['rule_ms'][RULES.index(b)]
                np.savez(os.path.join(tmp, r, name + '.npz'), amch=amch, ct=ct, ms=ms,
                         rule=b, rule_names=list(RULES), rule_ms=c['rule_ms'])
                manifest[f'{r}/{name}'] = {'rule': b, 'ms': ms,
                                           'rule_ms': dict(zip(RULES, c['rule_ms']))}
        with open(os.path.join(tmp, 'manifest.json'), 'w') as f:
            f.write(json.dumps(manifest, indent=1))
        with open(os.path.join(tmp, 'dataset_meta.json'), 'w') as f:
            f.write(meta_text())

        # (d) re-read every file
        files = sorted(os.path.relpath(os.path.join(d, x), tmp)
                       for d, _, fs in os.walk(tmp) for x in fs)
        want = sorted([os.path.join(r, p86.inst(i) + '.npz') for r in COMBOS
                       for i in range(N_CACHE)] + ['manifest.json', 'dataset_meta.json'])
        if files != want:
            raise RuntimeError(f'file set: extra {sorted(set(files) - set(want))[:3]}, '
                               f'missing {sorted(set(want) - set(files))[:3]}')
        with open(os.path.join(tmp, 'manifest.json')) as f:
            man = json.load(f)
        bad = []
        for r in COMBOS:
            for i in range(N_CACHE):
                name = p86.inst(i)
                p = os.path.join(tmp, r, name + '.npz')
                z = np.load(p, allow_pickle=True)
                if sorted(z.files) != sorted(p86.WS_KEYS):
                    bad.append(f'{r}/{name}: keys {z.files}')
                case = A.load_case(SIZE, r, name)
                (amch, st, ct), info = A.load_hint_npz(p, case)
                v = A.validate_all(case, amch, st, ct)
                if v:
                    bad.append(f'{r}/{name}: validate_all {v[:1]}')
                jl, pt, desc = load_cell(r, i)
                _, _, em = EM.load_stored_schedule(tmp, r, name, Instance(jl, pt, desc))
                m = man[f'{r}/{name}']
                if not (m['rule'] == info['rule'] == em['rule']
                        and m['ms'] == info['makespan'] == float(em['makespan'])
                        == float(ct.max()) and m['rule_ms'] == info['all_rules']
                        == em['rule_makespans']):
                    bad.append(f'{r}/{name}: manifest {m}, npz {info}, runner {em}')
        if bad:
            raise RuntimeError(f'{len(bad)} files fail the re-read: {bad[:5]}')
    except BaseException:
        shutil.rmtree(tmp)                                 # the folder this call made
        raise
    if os.path.exists(CACHE):
        aside = f'{CACHE}.replaced_{time.strftime("%Y%m%d-%H%M%S")}'
        os.rename(CACHE, aside)
        print(f'[p92] existing {CACHE} moved to {aside}', flush=True)
    os.rename(tmp, CACHE)
    lines = []
    for x in want:
        p = os.path.join(CACHE, x)
        with open(p, 'rb') as f:
            lines.append(f'{hashlib.sha256(f.read()).hexdigest()}  {p}')
    with open(SUMS, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    ok = 0
    for line in open(SUMS):
        h, p = line.rstrip('\n').split('  ', 1)
        with open(p, 'rb') as f:
            ok += hashlib.sha256(f.read()).hexdigest() == h
    return len(want), ok, len(lines)


def summary(cells, idx):
    """Mean makespan of each rule, the rule chosen and the gain over SPT per
    combination, and the comparison with the legacy-mask classical cache."""
    print(f'\n| Combination | FIFO | MOR | SPT | MWKR | Rules chosen | Best below SPT '
          f'| Legacy cache mean | Exact cache mean | Exact longer / equal / shorter |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    for r in COMBOS:
        cs = [cells[(r, i)] for i in idx]
        mean = [np.mean([c['rule_ms'][k] for c in cs]) for k in range(len(RULES))]
        ch = {}
        for c in cs:
            ch[c['best']] = ch.get(c['best'], 0) + 1
        best = np.array([c['rule_ms'][RULES.index(c['best'])] for c in cs])
        spt = np.array([c['rule_ms'][RULES.index('SPT')] for c in cs])
        leg = np.array([float(np.load(os.path.join(LEGACY, r, p86.inst(i) + '.npz'),
                                      allow_pickle=True)['ms']) for i in idx])
        print(f'| {r} | ' + ' | '.join(f'{m:.1f}' for m in mean)
              + f' | {", ".join(f"{k} {v}" for k, v in sorted(ch.items()))} '
              f'| {int((best < spt).sum())} | {leg.mean():.1f} | {best.mean():.1f} '
              f'| {int((best > leg).sum())} / {int((best == leg).sum())} / '
              f'{int((best < leg).sum())} |')


def main():
    a = parse_args(_ARGV)
    print(f'[p92] {SET}, instances {a.idx[0]:03d}-{a.idx[-1]:03d}, {len(COMBOS)} combinations, '
          f'rules {", ".join(RULES)}; cwd {os.getcwd()}; affinity '
          f'{sorted(os.sched_getaffinity(0))}', flush=True)
    stored, n_means = stored_arrays()
    print(f'[p92] {STORED}: {len(stored)} arrays of shape ({N_SET}, 2), column 0 integer '
          f'makespans; {n_means} means equal to summary_rules_xmask.md', flush=True)
    cells, st = build(a.idx, a.max_minutes, a.stage)
    n_roll = len(cells) * len(RULES)
    n_cmp, diff = check_stored(cells, stored)
    print(f'\n(a) rule makespans equal to {STORED}/Result_<RULE>_<combination>.npy: '
          f'{n_cmp - len(diff)} of {n_cmp}')
    differ = {(x[0], x[1], x[2]) for x in diff}
    for rule in RULES:
        print(f'    {rule}: ' + ', '.join(
            f'{r} {sum((rule, r, i) not in differ for i in a.idx)}/{len(a.idx)}'
            for r in COMBOS))
    for rule, r, i, got, want in diff:
        print(f'    DIFFERS {rule} {r}/{p86.inst(i)}: built {got:g}, stored {want:g} '
              f'({got - want:+g})')
    print(f'(b) rule schedules valid (reference_sim, C6 PASS, validate_all): '
          f'{n_roll - len(st["invalid"])} of {n_roll}')
    for x in st['invalid'][:20]:
        print(f'    INVALID {x}')
    print(f'(c) rollouts with exact first_mask and mask_ties at every rule call: '
          f'{st["n_exact"]} of {n_roll} ({st["n_calls"]} rule calls checked)')
    print(f'    build time {st["seconds"]:.0f} s for {n_roll} rollouts on one core '
          f'({st["n_loaded"]} read from the stage {a.stage}, the rest rolled out in this call)'
          if a.stage else
          f'    build time {st["seconds"]:.0f} s for {n_roll} rollouts on one core')
    summary(cells, a.idx)
    if diff or st['invalid'] or st['n_exact'] != n_roll:
        print('\nSTOP: a check failed; nothing written')
        return 1
    if a.check_only:
        print('\n--check-only: nothing written')
        return 0
    n_files, n_ok, n_sums = write_cache(cells, a.force)
    print(f'\n(d) {CACHE}: {n_files} files (240 npz, manifest.json, dataset_meta.json), '
          f'every npz re-read: keys, validate_all, the memetic runner\'s loader and the '
          f'manifest agree; {SUMS}: {n_sums} sha256 lines, {n_ok} re-checked equal')
    return 0


if __name__ == '__main__':
    sys.exit(main())
