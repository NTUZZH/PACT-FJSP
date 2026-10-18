r"""An instance lower bound from the certified estimate at the empty schedule.

Theorem 1 states that the estimate C^(s;c) of Eq. (3) never exceeds the
makespan of any feasible completion of s. At the empty schedule s_0 no
operation is realized, so C^(s_0;c) is a function of instance data alone:

  per job i:  c^_{i,j} = c^_{i,j-1} + a_L * lag_{i,j-1} + min_k p_ijk
                         + a_S * min_lambda S[lambda, class(i,j)]
  C^(s_0;c)  = max_i c^_{i,n_i}

and it is a lower bound on the optimal makespan of that instance under c.
This script computes it for all 100 test instances x 8 regimes at 10x25 and
reports, per regime:

  (i)   mean of 100*(C_policy / C^ - 1): an UPPER bound on the PACT policy's
        optimality gap (policy pooled over its three released seeds, per seed
        per instance, then averaged);
  (ii)  the same for credited CP-SAT at 30 s (all 100) and 300 s (the fixed
        30-instance prefix), credited exactly as p13_e5_table_fair.py credits;
  (iii) mean of LB_solver / C^ at 300 s, where LB_solver is the CP-SAT proven
        lower bound ('bound' field), so the reader knows which bound is tighter.

TIME AXIS. The environment keeps a normalized axis (pt scaled by the slope
1/(pt_max - pt_min + 1e-8), pt_min == 0 asserted by the environment); its
initial bound init_quality is converted back to hours by dividing by that
slope. The value is cross-checked instance by instance against an independent
numpy evaluation of Eq. (3) on the true-hour axis read from the files, on ALL
800 instance-regime pairs (the brief asks for two; all are cheap).

GUARD. The bound is admissible, so no makespan any method reports may fall
below it. A violation is a refusal, not a footnote.

CPU only: the environment's state tensors are placed on configs.device, which
is forced to cpu here, and CUDA is hidden before torch is imported.

Writes paper/macros_ibound.tex and results/instance_bound.md.
Usage: python3 scripts/p64_instance_bound.py
"""
import glob
import json
import os
import re
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.argv = [sys.argv[0], '--device', 'cpu']
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import numpy as np  # noqa: E402

from macro_io import emit, refuse  # noqa: E402
from constraint_family.descriptor import (ConstraintDescriptor,  # noqa: E402
                                          compute_token)
from constraint_family.fjsp_env_family import FJSPEnvFamily  # noqa: E402
from ppvc_instance_generator import load_instance  # noqa: E402

REGIMES = ['N', 'L', 'S', 'W', 'LS', 'LW', 'SW', 'LSW']
SETUP = ('S', 'LS', 'SW', 'LSW')
DESC = 'data/FAMILY/10x25/desc'
BASE = 'data/PPVC/10x25+ppvc-mixed'
RES = 'test_results/FAMILY/10x25'
OR = 'or_solution/FAMILY'
POLICY = 'greedy+10x25+family+jointblind-v1'
SEEDS = ('', '-s302', '-s303')
MACROS = 'paper/macros_ibound.tex'
MD = 'results/instance_bound.md'
_REPAIRED = json.load(open(f'{OR}/warmstart_cache_repaired/makespans.json'))


def env_bound(jl, pt, meta, desc):
    """C^(s_0;c) as the environment computes it at reset, in hours."""
    env = FJSPEnvFamily(jl.shape[0], pt.shape[1], family_mode=True)
    env.delta_mode = 'min'
    env.reward_delta_env = True
    env.set_initial_data([jl], [pt], descriptor_list=[desc],
                         op_type_list=[meta['op_type']],
                         mch_type_list=[meta['mch_type']])
    return float(env.init_quality[0]) / env._slope


def hand_bound(jl, pt, desc):
    """Independent evaluation of Eq. (3) at s_0 on the true-hour axis."""
    pt = np.asarray(pt, dtype=float)
    a_l = 1.0 if desc.has_lag else 0.0
    a_s = 1.0 if desc.has_setup else 0.0
    if not desc.has_lag and np.any(desc.lag != 0):
        raise SystemExit('descriptor carries a nonzero lag with LAG inactive')
    min_pt = np.where(pt > 0, pt, np.inf).min(axis=1)
    if a_s:
        env_col = np.asarray(desc.setup, dtype=float).min(axis=0)   # over sources
        delta = env_col[desc.op_class]
    else:
        delta = np.zeros(len(min_pt))
    best, first = 0.0, 0
    for n in np.asarray(jl, dtype=int):
        c = 0.0
        for j in range(first, first + n):
            lag_prev = desc.lag[j - 1] if j > first else 0.0
            c = c + a_l * lag_prev + min_pt[j] + a_s * delta[j]
        best = max(best, c)
        first += n
    return best


def npy(tag, r):
    p = f'{RES}/Result_{tag}_{r}.npy'
    if not os.path.exists(p):
        return None
    a = np.load(p)
    return (a[:, 0] if a.ndim > 1 else a).astype(float)


def cp_tier(r, b):
    """{instance: (credited, solver bound)} with p13's credit rule."""
    rerun = b in (5, 30)
    src = (f'{OR}/10x25_fairwarmR{b}s/{r}/*.json' if rerun
           else f'{OR}/10x25_fairwarm{b}s/{r}/*.json')
    out = {}
    for f in sorted(glob.glob(src)):
        d = json.load(open(f))
        d = d[0] if isinstance(d, list) else d
        obj = d.get('objective')
        if obj is None:
            raise SystemExit(f'{f}: null objective')
        if rerun:
            c = min(obj, d['warm_makespan'])
        else:
            c = min(_REPAIRED[r][d['instance']]['repaired_ms'],
                    min(obj, d['warm_makespan']))
        out[d['instance']] = (float(c), float(d['bound']))
    return out


# ------------------------------------------------------------------ bounds
bound, hand_err = {}, 0.0
names_by_regime = {}
for r in REGIMES:
    paths = sorted(glob.glob(f'{DESC}/{r}/*.desc.json'))
    if len(paths) != 100:
        refuse(MACROS, f'{r}: {len(paths)} descriptors, expected 100')
    names, vals = [], []
    for p in paths:
        name = os.path.basename(p).replace('.desc.json', '')
        desc = ConstraintDescriptor.from_json(json.load(open(p)))
        jl, pt, meta = load_instance(os.path.join(BASE, name))
        e = env_bound(jl, pt, meta, desc)
        h = hand_bound(jl, pt, desc)
        hand_err = max(hand_err, abs(e - h))
        if abs(e - h) > 1e-4:
            refuse(MACROS, f'{r}/{name}: environment bound {e:.6f} h != hand '
                           f'evaluation {h:.6f} h')
        names.append(name)
        vals.append(h)                     # exact integer-hour value
    names_by_regime[r] = names
    bound[r] = np.array(vals)
print(f'environment vs hand evaluation of C^(s_0;c): max |diff| = {hand_err:.2e} h '
      f'over {sum(len(v) for v in bound.values())} instance-regime pairs')

# ------------------------------------------------------------------ gaps
res, viol = {}, []
for r in REGIMES:
    B = bound[r]
    names = names_by_regime[r]
    pol = [npy(f'{POLICY}{s}', r) for s in SEEDS]
    if any(p is None or len(p) != 100 for p in pol):
        refuse(MACROS, f'policy results missing on {r}')
    pol = np.stack(pol)                                       # [3, 100]
    for k in range(3):
        if np.any(pol[k] < B - 1e-6):
            viol.append(f'policy seed {k} on {r}')
    pol_gap = 100.0 * (pol / B[None, :] - 1.0)               # [3, 100]

    t30, t300 = cp_tier(r, 30), cp_tier(r, 300)
    if len(t30) != 100:
        refuse(MACROS, f'CP 30 s on {r}: {len(t30)} of 100')
    idx = {n: i for i, n in enumerate(names)}
    c30 = np.array([t30[n][0] for n in names])
    i300 = sorted(idx[n] for n in t300)
    c300 = np.array([t300[names[i]][0] for i in i300])
    lb300 = np.array([t300[names[i]][1] for i in i300])
    lb30 = np.array([t30[n][1] for n in names])
    B300 = B[i300]
    for lab, v, b in (('CP 30 s', c30, B), ('CP 300 s', c300, B300)):
        if np.any(v < b - 1e-6):
            viol.append(f'{lab} on {r}')
    res[r] = dict(
        B=float(B.mean()),
        pol=float(pol_gap.mean()),
        pol_max=float(pol_gap.mean(axis=0).max()),
        pol300=float(pol_gap[:, i300].mean()),
        cp30=float((100.0 * (c30 / B - 1.0)).mean()),
        cp300=float((100.0 * (c300 / B300 - 1.0)).mean()),
        lbratio=float((lb300 / B300).mean()),
        lbratio30=float((lb30 / B).mean()),
        lb_tighter=int((lb300 > B300 + 1e-6).sum()),
        ours_tighter=int((B300 > lb300 + 1e-6).sum()),
        tight_cp30=int((np.abs(c30 - B) < 1e-6).sum()),
        # the solver's gap to its OWN bound, for comparison with (ii)
        cp300_own=float((100.0 * (c300 / lb300 - 1.0)).mean()),
        cp300_best=float((100.0 * (c300 / np.maximum(lb300, B300) - 1.0)).mean()),
        # PACT against the better of the two bounds on the prefix
        pol_best=float((100.0 * (pol[:, i300] / np.maximum(lb300, B300)[None, :]
                                 - 1.0)).mean()),
        n300=len(i300))
if viol:
    refuse(MACROS, 'a reported makespan falls BELOW the admissible bound: '
           + '; '.join(viol) + '. Either Theorem 1 or the reading of the files '
           'is wrong; nothing is emitted.')
print('admissibility guard: no policy or CP-SAT makespan below C^(s_0;c)')


def pooled(key, regs):
    return float(np.mean([res[r][key] for r in regs]))   # equal n per regime


NOSET = [r for r in REGIMES if r not in SETUP]

# ------------------------------------------------------------------ macros
lines = ['% auto-generated by scripts/p64_instance_bound.py -- do not hand-edit',
         '% Instance lower bound = the certified estimate at the empty schedule,',
         '% C^(s_0;c) of Eq. (3). Gaps: mean of 100*(makespan/C^ - 1), an UPPER',
         '% bound on the optimality gap. LbRatio: mean CP-SAT 300 s proven bound',
         '% divided by C^ (>1 = the solver bound is tighter).']
for r in REGIMES:
    x = res[r]
    lines.append(
        f'\\newcommand{{\\IboundPol{r}}}{{{x["pol"]:.1f}}}'
        f'\\newcommand{{\\IboundCpThirty{r}}}{{{x["cp30"]:.1f}}}'
        f'\\newcommand{{\\IboundCpThreeHundred{r}}}{{{x["cp300"]:.1f}}}'
        f'\\newcommand{{\\IboundLbRatio{r}}}{{{x["lbratio"]:.2f}}}'
        f'  % {r}: gap to C^ (%) of PACT, CP 30 s, CP 300 s ({x["n300"]}-inst prefix);'
        f' solver bound / C^')
for tag, regs in (('All', REGIMES), ('Setup', list(SETUP)), ('NoSetup', NOSET)):
    lines.append(
        f'\\newcommand{{\\IboundPol{tag}}}{{{pooled("pol", regs):.1f}}}'
        f'\\newcommand{{\\IboundCpThirty{tag}}}{{{pooled("cp30", regs):.1f}}}'
        f'\\newcommand{{\\IboundCpThreeHundred{tag}}}{{{pooled("cp300", regs):.1f}}}'
        f'\\newcommand{{\\IboundLbRatio{tag}}}{{{pooled("lbratio", regs):.2f}}}'
        f'\\newcommand{{\\IboundPolPrefix{tag}}}{{{pooled("pol300", regs):.1f}}}'
        f'  % pooled over {",".join(regs)}; PolPrefix = PACT on the same 300 s prefix as the solver')
n_lb_tighter = sum(res[r]['lb_tighter'] for r in REGIMES)
n300_tot = sum(res[r]['n300'] for r in REGIMES)
lines.append(f'\\newcommand{{\\IboundSolverTighterN}}{{{n_lb_tighter}}}'
             f'\\newcommand{{\\IboundSolverTighterOf}}{{{n300_tot}}}'
             '  % 300 s instance-regime pairs where the solver bound exceeds C^')

mine = set(re.findall(r'\\newcommand\{\\([A-Za-z]+)\}', '\n'.join(lines)))
for f in glob.glob('paper/macros_*.tex') + ['paper/macros.tex']:
    if os.path.abspath(f) == os.path.abspath(MACROS):
        continue
    clash = mine & set(re.findall(r'\\newcommand\{\\([A-Za-z]+)\}', open(f).read()))
    if clash:
        refuse(MACROS, f'macro name collision with {f}: {sorted(clash)}')
emit(MACROS, lines)

# ------------------------------------------------------------------ report
hdr = ('| regime | mean C^ (h) | PACT gap to C^ (%) | PACT, 300 s prefix (%) '
       '| CP 30 s gap to C^ (%) | CP 300 s gap to C^ (%) | CP 300 s gap to own bound (%) '
       '| PACT gap to max(bound, C^), prefix (%) '
       '| solver bound / C^ (300 s) | solver bound / C^ (30 s) | solver bound tighter (of n) |')
tab = [hdr, '|' + '---|' * 11]
for r in REGIMES:
    x = res[r]
    tab.append(f'| {r} | {x["B"]:.1f} | {x["pol"]:.1f} | {x["pol300"]:.1f} | '
               f'{x["cp30"]:.1f} | {x["cp300"]:.1f} | {x["cp300_own"]:.1f} | '
               f'{x["pol_best"]:.1f} | {x["lbratio"]:.3f} | {x["lbratio30"]:.3f} | '
               f'{x["lb_tighter"]}/{x["n300"]} |')
for tag, regs in (('all 8', REGIMES), ('setup', list(SETUP)), ('no setup', NOSET)):
    tab.append(f'| *{tag}* | | {pooled("pol", regs):.1f} | {pooled("pol300", regs):.1f} | '
               f'{pooled("cp30", regs):.1f} | {pooled("cp300", regs):.1f} | '
               f'{pooled("cp300_own", regs):.1f} | {pooled("pol_best", regs):.1f} | '
               f'{pooled("lbratio", regs):.3f} | '
               f'{pooled("lbratio30", regs):.3f} | |')
print('\n'.join(tab))

pa, ca = pooled('pol', REGIMES), pooled('cp300', REGIMES)
lr = pooled('lbratio', REGIMES)
with open(MD, 'w') as f:
    f.write('# Instance lower bound from the certified estimate at the empty schedule\n\n'
            'C^(s_0;c) is Eq. (3) evaluated before any operation is scheduled: per job, '
            'the sum over its operations of the predecessor lag (if LAG is active), the '
            'shortest compatible processing time and the cheapest inbound setup (if '
            'SETUP is active), maximized over jobs. It was computed by the environment '
            'at reset (normalized axis, divided by the slope) and independently in '
            f'numpy on the hour axis; the two agree to {hand_err:.1e} h on all 800 '
            'instance-regime pairs. No policy or CP-SAT makespan falls below it.\n\n'
            'Gap to C^ = mean of 100*(makespan/C^ - 1), an upper bound on the optimality '
            'gap. PACT pooled over three seeds on all 100 instances; CP-SAT credited '
            'min(repaired hint, search) as in Table III; 300 s on the 30-instance prefix '
            '(the PACT column restricted to the same prefix is given for comparison). '
            'Solver bound = CP-SAT BestObjectiveBound.\n\n')
    f.write('\n'.join(tab) + '\n\n')
    f.write('## Plain reading\n\n')
    pb = pooled('pol_best', REGIMES)
    loose = pa >= 10
    f.write(
        f'Pooled over the eight regimes, the PACT makespan is {pa:.1f}% above '
        f'C^(s_0;c), and the 300 s CP-SAT schedule is {ca:.1f}% above it. '
        f'These percentages are upper bounds on the optimality gap. ')
    if loose:
        f.write(
            'They are tens of percent even for the 300 s solver, so the bound is too '
            'loose to report as an optimality certificate. Most of the distance '
            'measures the bound, not the schedule: Eq. (3) follows each job chain '
            'alone and ignores competition for machines, and it gives WINDOW no '
            'credit at all, so it is loosest on N, S, W and SW, where lags do not '
            'dominate the job chain. It is close only where long lags make the '
            'job chain the binding constraint (L, LW). ')
    else:
        f.write('The bound is tight enough to report as an upper bound on the '
                'optimality gap. ')
    f.write(
        f'The CP-SAT proven bound at 300 s averages {lr:.2f} times C^(s_0;c) and '
        f'exceeds it on {n_lb_tighter} of {n300_tot} instance-regime pairs, so the '
        f'solver bound is the tighter of the two. Measured against the better of '
        f'the two bounds on the 30-instance prefix, the PACT gap bound falls to '
        f'{pb:.1f}%. C^(s_0;c) earns its place as the admissible shaping potential '
        f'of Theorem 1, and it is not the bound to quote for an optimality gap.\n')
print(f'wrote {MD}')
