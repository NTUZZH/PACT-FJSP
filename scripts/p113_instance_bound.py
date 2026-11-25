r"""P113: the makespan estimate at the empty schedule against the shortest schedules found
(exact environment, factory benchmark, training size 10x25).

QUANTITY
  For a test instance and a rule combination c, C^(s_0;c) is the makespan estimate of the
  environment at the empty schedule: FJSPEnvFamily.set_initial_data computes init_quality on
  the normalized time axis, converted to hours by dividing by the environment's slope, with
  the reward settings (delta_mode, reward_delta_env) and the mask options of the selected
  policy run's configuration. For every (instance, combination) with a valid 600 s record of
  both searches, CP-SAT started from a rule schedule (test_results/FAMILY/10x25_cparms/
  evidence_rule600s/<arm>/<R>/instance_<i>.json, the pilot's arm) and the memetic algorithm
  (or_solution/FAMILY/10x25_memetic600s/<R>/instance_<i>_seed1.json), read through the
  functions of scripts/p94_timed_inference.py (records that scripts/timing_validity.py
  accepts), the script reports
      found  = 100 (min(final makespan of CP-SAT, final makespan of the memetic algorithm)
                    / C^(s_0;c) - 1)
      policy = 100 (greedy makespan of the selected policy run / C^(s_0;c) - 1)
  per combination and pooled over all pairs: the mean, the smallest and the largest value.
  C^(s_0;c) never exceeds the makespan of a feasible schedule; the script stops (nothing
  written) when it exceeds any makespan of the comparison on any instance.

OUTPUTS (arm_registry.out under PACT_OUT)
  manuscript/macros_ibound.tex   macro_io.emit; names Ibound<quantity><R|All>, letters only,
                                none defined in another .tex file under manuscript/
  results/instance_bound.json   every pair with its estimate and makespans
  results/instance_bound.md     tables
  Values print with one decimal.

CHECK (--check, read only)
  exit 0 when the greedy arrays of the selected policy run exist with one row per test
  instance on the eight combinations and at least one instance of each combination has a
  valid record of both searches.

Usage: PACT_ENV=exact PACT_SEEDS=301-305 PACT_OUT=out/exact python scripts/p113_instance_bound.py
       python scripts/p113_instance_bound.py --check
"""
import argparse
import glob
import json
import os
import re
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import arm_registry as REG  # noqa: E402  (pops --pact-* options from sys.argv)

_ARGV = sys.argv[1:]
sys.argv = sys.argv[:1] + ['--device', 'cpu']

import numpy as np  # noqa: E402

from macro_io import emit  # noqa: E402

ENV = 'exact'
SIZE = '10x25'
DATA = 'data/FAMILY/10x25'
BUDGET = 600
TOKENS = (f'memetic_rule@{BUDGET}', f'cpsat_rule@{BUDGET}')
RUNNER = {'memetic_rule': 'memetic', 'cpsat_rule': 'cpsat'}
LAYOUT = {'memetic_rule': r'^or_solution/FAMILY/10x25_memetic600s/[A-Z]+/instance_\d{3}_seed1\.json$',
          'cpsat_rule': r'^test_results/FAMILY/10x25_cparms/evidence_rule600s/[a-z_]+/[A-Z]+/'
                        r'instance_\d{3}\.json$'}
R8 = list(REG.COMBOS)
OUT_FILES = {'macros': 'manuscript/macros_ibound.tex', 'json': 'results/instance_bound.json',
             'md': 'results/instance_bound.md'}
PREFIX = 'Ibound'
# counts of the solver's own bound, which this comparison does not use
NOT_WRITTEN = ('IboundSolverTighterN', 'IboundSolverTighterOf')
TOL = 1e-6


class Refused(Exception):
    """An input or a value that this script does not accept."""


# ----------------------------------------------------------------- the estimate
def run_config(seed):
    p = REG.config_path('released', None, seed, ENV)
    if not os.path.exists(p):
        raise Refused(f'no configuration {p} of the selected policy run')
    return json.load(open(p))


def make_env(n_j, n_m, cfg):
    import torch
    import fjsp_env_same_op_nums
    from params import configs
    from constraint_family.fjsp_env_family import FJSPEnvFamily
    configs.device = 'cpu'                    # the state tensors of the environment
    fjsp_env_same_op_nums.EnvState.device = torch.device('cpu')
    env = FJSPEnvFamily(n_j, n_m, family_mode=True,
                        mask_mode=cfg.get('mask_mode') or 'nondelay',
                        pair_extra=cfg.get('pair_extra') or 'none',
                        first_mask=cfg.get('first_mask'), mask_ties=cfg.get('mask_ties'))
    env.delta_mode = cfg.get('delta_mode')
    env.reward_delta_env = cfg.get('reward_delta_env', True)
    return env


def estimate(name, combo, cfg):
    """C^(s_0;c) in hours for one test instance and combination, by the environment."""
    from constraint_family.descriptor import ConstraintDescriptor
    from ppvc_instance_generator import load_instance
    rec = json.load(open(f'{DATA}/desc/{combo}/{name}.desc.json'))
    desc = ConstraintDescriptor.from_json(rec, setup_convention='attached')
    jl, pt, meta = load_instance(rec['base_instance'])
    env = make_env(jl.shape[0], pt.shape[1], cfg)
    env.set_initial_data([jl], [pt], descriptor_list=[desc], op_type_list=[meta['op_type']],
                         mch_type_list=[meta['mch_type']])
    return float(env.init_quality[0]) / env._slope


def instance_names():
    names = sorted(os.path.basename(p)[:-len('.desc.json')]
                   for p in glob.glob(f'{DATA}/desc/N/*.desc.json'))
    for r in R8:
        other = sorted(os.path.basename(p)[:-len('.desc.json')]
                       for p in glob.glob(f'{DATA}/desc/{r}/*.desc.json'))
        if other != names:
            raise Refused(f'the descriptors of {r} do not name the instances of N')
    if not names:
        raise Refused(f'no descriptors under {DATA}/desc')
    return names


# ----------------------------------------------------------------- records
def records():
    """{(combo, idx): {token arm: cell}} of the 600 s records at 10x25, the selected policy
    seed and the pilot's CP-SAT arm, through the readers of p94."""
    import p94_timed_inference as T
    P = T.P
    a = T.build_parser().parse_args([])
    table = P.build_parser().parse_args(['table'])
    slots = [P.parse_cores(s) for s in table.slots]
    chosen, why = T.select_cpsat_arm('test_results/FAMILY', slots, table.bars)
    if chosen is None:
        raise Refused(f'the pilot selection of the CP-SAT arm is unavailable: '
                      f'{why[0] if why else "-"}')
    seed, _ = T.select_policy_seed()
    try:
        camp = T.make_campaign([SIZE], list(TOKENS), a, seed, chosen)
        cells, counts, gaps = T.read_cells([camp], (float(BUDGET),), 'start')
    except T.Refusal as e:
        raise Refused(f'the records cannot be read: {e}')
    out = {}
    for (arm, size, combo, idx), x in cells.items():
        if size != SIZE or arm not in RUNNER:
            continue
        if not re.match(LAYOUT[arm], x['record']):
            raise Refused(f'{x["record"]}: not the record layout of {arm} ({LAYOUT[arm]})')
        out.setdefault((combo, int(idx)), {})[arm] = x
    return out, chosen, seed, counts


def both_valid(recs):
    return {k: v for k, v in recs.items()
            if all(v.get(arm, {}).get('state') == 'valid' for arm in RUNNER)}


def policy_arrays(seed, n):
    out = {}
    for r in R8:
        try:
            out[r] = REG.load('released', None, seed, SIZE, r, n)
        except REG.MissingResult as e:
            raise Refused(str(e))
    return out


# ----------------------------------------------------------------- analysis
def distance(m, c_hat):
    """100 (m / c_hat - 1); exactly 0 when m equals c_hat within TOL (the estimate is
    computed on the normalized axis and carries rounding of about 1e-7 h)."""
    if abs(m - c_hat) <= TOL * max(1.0, abs(m)):
        return 0.0
    return 100.0 * (m / c_hat - 1.0)


def summary(v):
    v = np.asarray(v, dtype=float)
    return dict(n=int(v.size), mean=float(v.mean()), min=float(v.min()), max=float(v.max()))


def analyse():
    if REG.env() != ENV:
        raise Refused(f'the comparison is that of the {ENV} environment; the active '
                      f'environment is {REG.env()} (set PACT_ENV={ENV})')
    names = instance_names()
    recs, cpsat_arm, seed, counts = records()
    pairs = both_valid(recs)
    if not pairs:
        raise Refused('no instance has a valid 600 s record of both searches')
    cfg = run_config(seed)
    from constraint_family.fjsp_env_family import resolve_delta_mode
    from types import SimpleNamespace
    mode = resolve_delta_mode(SimpleNamespace(delta_mode=cfg.get('delta_mode'),
                                              reward_delta_env=cfg.get('reward_delta_env', True)))
    pol = policy_arrays(seed, len(names))
    rows, viol = [], []
    for (combo, idx), v in sorted(pairs.items(), key=lambda kv: (R8.index(kv[0][0]), kv[0][1])):
        name = names[idx]
        c_hat = estimate(name, combo, cfg)
        mem = v['memetic_rule']['final_makespan']
        cps = v['cpsat_rule']['final_makespan']
        p = float(pol[combo][idx])
        for what, m in (('memetic algorithm', mem), ('CP-SAT', cps), ('policy', p)):
            if c_hat > m + TOL * max(1.0, abs(m)):
                viol.append(f'{combo}/{name}: estimate {c_hat:.6f} h above the {what} makespan '
                            f'{m:.6f} h')
        best = min(mem, cps)
        rows.append(dict(combo=combo, instance=name, estimate=c_hat, memetic=mem, cpsat=cps,
                         shortest=best, policy=p, found=distance(best, c_hat),
                         policy_gap=distance(p, c_hat), at_estimate=distance(best, c_hat) == 0.0,
                         records={arm: v[arm]['record'] for arm in RUNNER}))
    if viol:
        raise Refused(f'{len(viol)} makespans lie below the estimate at the empty schedule; '
                      f'first: {viol[0]}')
    per = {}
    for r in R8 + ['All']:
        sel = [x for x in rows if r == 'All' or x['combo'] == r]
        if not sel:
            raise Refused(f'{r}: no instance with a valid record of both searches')
        per[r] = dict(found=summary([x['found'] for x in sel]),
                      policy=summary([x['policy_gap'] for x in sel]),
                      at_estimate=sum(x['at_estimate'] for x in sel),
                      instances=sorted({x['instance'] for x in sel}))
    return dict(generator='scripts/p113_instance_bound.py', setting=REG.describe(), size=SIZE,
                budget_s=BUDGET, cpsat_arm=cpsat_arm, policy_seed=seed, delta_mode=mode,
                mask=dict(first_mask=cfg.get('first_mask'), mask_ties=cfg.get('mask_ties')),
                record_states=counts, per=per, pairs=rows)


# ----------------------------------------------------------------- outputs
def defined_macros():
    """{name: file} of every macro defined in a .tex file under manuscript/, the output file
    left out."""
    rx = re.compile(r'\\(?:newcommand|renewcommand|providecommand)\*?\s*\{?\s*\\([A-Za-z@]+)'
                    r'|\\def\s*\\([A-Za-z@]+)')
    own = os.path.basename(OUT_FILES['macros'])
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, 'manuscript', '**', '*.tex'), recursive=True)):
        if os.path.basename(f) == own:
            continue
        text = '\n'.join(re.sub(r'(?<!\\)%.*', '', x)
                         for x in open(f, errors='replace').read().splitlines())
        for a, b in rx.findall(text):
            out.setdefault(a or b, os.path.relpath(f, ROOT))
    return out


def macro_lines(res):
    L = [f'% Generated by scripts/p113_instance_bound.py ({res["setting"]}) -- do not hand-edit',
         f'% Makespan estimate at the empty schedule C^(s_0;c) against the shortest schedule '
         f'found by CP-SAT started from a rule schedule ({res["cpsat_arm"]}) or the memetic '
         f'algorithm in {res["budget_s"]} s, and against the greedy schedule of the selected '
         f'policy run (seed {res["policy_seed"]}), at {res["size"]}: 100 (makespan / C^ - 1), %, '
         f'over the instances with a valid record of both searches.']
    for r in R8 + ['All']:
        x = res['per'][r]
        s = (f'\\newcommand{{\\{PREFIX}Pairs{r}}}{{{x["found"]["n"]}}}'
             f'\\newcommand{{\\{PREFIX}AtEstimate{r}}}{{{x["at_estimate"]}}}')
        for word, key in (('Found', 'found'), ('Policy', 'policy')):
            s += (f'\\newcommand{{\\{PREFIX}{word}{r}}}{{{x[key]["mean"]:.1f}}}'
                  f'\\newcommand{{\\{PREFIX}{word}Min{r}}}{{{x[key]["min"]:.1f}}}'
                  f'\\newcommand{{\\{PREFIX}{word}Max{r}}}{{{x[key]["max"]:.1f}}}')
        L.append(s + f'  % {"pooled" if r == "All" else r}: pairs; pairs whose shortest '
                     f'schedule found equals the estimate; mean, smallest and largest distance '
                     f'of the shortest schedule found and of the policy schedule')
    return L


def check_names(lines):
    names = re.findall(r'\\newcommand\{\\([^}]*)\}', '\n'.join(lines))
    bad = [x for x in names if not re.fullmatch(r'[A-Za-z]+', x) or not x.startswith(PREFIX)]
    if bad:
        raise Refused(f'macro names that are not letters only with the prefix {PREFIX}: {bad[:5]}')
    if len(names) != len(set(names)) or set(names) & set(NOT_WRITTEN):
        raise Refused('duplicate macro names, or a name of NOT_WRITTEN in the output')
    have = defined_macros()
    clash = sorted(x for x in names if x in have)
    if clash:
        raise Refused('macro names already defined: '
                      + ', '.join(f'\\{x} ({have[x]})' for x in clash[:5]))
    return names


def markdown(res):
    md = [f'# Makespan estimate at the empty schedule against the shortest schedules found '
          f'({res["setting"]})', '',
          f'Size {res["size"]}; CP-SAT arm {res["cpsat_arm"]} and the memetic algorithm, '
          f'{res["budget_s"]} s records that scripts/timing_validity.py accepts; policy seed '
          f'{res["policy_seed"]} (greedy); estimate with delta_mode {res["delta_mode"]}, mask '
          f'options {res["mask"]}. Values: 100 (makespan / C^(s_0;c) - 1), %.', '',
          '| comb. | pairs | found at the estimate | found: mean [min, max] | '
          'policy: mean [min, max] |', '|---|---|---|---|---|']
    for r in R8 + ['All']:
        x = res['per'][r]
        md.append(f'| {"pooled" if r == "All" else r} | {x["found"]["n"]} | {x["at_estimate"]} | '
                  f'{x["found"]["mean"]:.1f} [{x["found"]["min"]:.1f}, {x["found"]["max"]:.1f}] '
                  f'| {x["policy"]["mean"]:.1f} [{x["policy"]["min"]:.1f}, '
                  f'{x["policy"]["max"]:.1f}] |')
    md += ['', 'Record states (arm|size|combination):', '']
    md += [f'- {k}: ' + ', '.join(f'{s} {n}' for s, n in sorted(v.items()))
           for k, v in sorted(res['record_states'].items())]
    return md


def outputs():
    return {k: REG.out(v) for k, v in OUT_FILES.items()}


def remove_outputs(outs):
    for p in outs.values():
        if os.path.exists(p):
            os.remove(p)


def run():
    if REG.env() != ENV:
        raise Refused(f'the comparison is that of the {ENV} environment; the active '
                      f'environment is {REG.env()} (set PACT_ENV={ENV})')
    if os.environ.get('PACT_OUT') is None:
        raise Refused('set PACT_OUT (the output root of the chain, e.g. out/exact)')
    outs = outputs()
    remove_outputs(outs)
    try:
        res = analyse()
        lines = macro_lines(res)
        names = check_names(lines)
        emit(outs['macros'], lines)
        with open(outs['json'], 'w') as f:
            json.dump(res, f, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
        md = markdown(res)
        with open(outs['md'], 'w') as f:
            f.write('\n'.join(md) + '\n')
    except BaseException:
        remove_outputs(outs)
        raise
    return res, names, md


def check():
    """Read-only precondition of the chain step; returns the exit code."""
    if REG.env() != ENV:
        print(f'p113: the active environment is {REG.env()}, not {ENV}')
        return 1
    names = instance_names()
    recs, chosen, seed, _ = records()
    pairs = both_valid(recs)
    per = {r: sum(1 for (c, _) in pairs if c == r) for r in R8}
    why = []
    try:
        policy_arrays(seed, len(names))
    except Refused as e:
        why.append(str(e).splitlines()[0])
    why += [f'{r}: no instance with a valid record of both searches' for r in R8 if not per[r]]
    print(f'p113: CP-SAT arm {chosen}, policy seed {seed}; instances with a valid {BUDGET} s '
          f'record of both searches: ' + ', '.join(f'{r} {per[r]}' for r in R8))
    print('p113: inputs ' + ('present' if not why else f'missing: {why[0]}'))
    return 0 if not why else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description='the makespan estimate at the empty schedule '
                                             'against the shortest schedules found')
    ap.add_argument('--check', action='store_true', help='read-only precondition of the chain')
    a = ap.parse_args(_ARGV if argv is None else argv)
    os.chdir(ROOT)
    try:
        if a.check:
            sys.exit(check())
        REG.banner()
        res, names, md = run()
    except Refused as e:
        sys.exit(f'*** REFUSING TO CERTIFY ***\n{e}')
    print('\n'.join(md))
    print(f'\n{len(names)} macros; wrote ' + ', '.join(outputs().values()))


if __name__ == '__main__':
    main()
