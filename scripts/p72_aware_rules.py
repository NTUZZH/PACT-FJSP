"""Three constraint-aware dispatching rules beside the four textbook ones.

Why. FIFO, MOR, SPT and MWKR rank jobs or pairs by processing time and work
counts only. None of them reads a changeover, an outage or a post-operation
lag, so a reader can ask whether PACT's margin is a margin over rules that are
blind to the constraints, or over any rule at all. This script answers with
three rules that read exactly the quantities the textbook rules ignore, run
through the same environment, seed and feasibility check as p14.

The rules (one decision = one (job, machine) pair, as in SPT):

  ECT   earliest completion time. Among the COMPATIBLE pairs (every unfinished
        job's candidate on every machine that can process it), choose the one
        with the earliest completion = processing start + processing time, where
        the processing start is the environment's own pair start: block start
        max(job ready, machine free), pushed past every outage the block
        [block, block + setup + p] would overlap, plus the attached setup from
        the machine's last routing class. This is the Giffler-Thompson
        earliest-completion pair, so its action set is the active-schedule set
        of mask_mode='active', a superset of the policy's non-delay set.
        Why not the non-delay set: next_schedule_time is the minimum pair start
        and the mask removes every pair starting later, so every legal pair
        starts at the same instant and min(start + p) over legal pairs is
        min(p) over legal pairs, i.e. SPT with the same tie set and the same
        random draws. That variant is kept as ECTND for the identity check.
        Times: the normalized pair start of env._compute_pair_free_time()
        (recomputed, not env.pair_free_time, which initial_vars leaves at zero
        before the first step and so ignores the cold-start setup and outages
        at the first decision) plus env.candidate_pt, divided by env._slope and
        rounded to 1e-6 h. All instance times are integer hours, so ties are
        exact ties in hours.

  SSPT  shortest setup plus processing. Among the LEGAL pairs (dynamic_pair_mask
        false, the policy's action set), minimize the setup into the candidate
        on that machine plus its processing time. Setup from
        env._setup_into_candidates(true_side=True), processing time from
        env.true_op_pt, both in true hours. With SETUP inactive the setup term
        is exactly 0, the key is the true processing time, and the rule is SPT
        with the same tie set and the same random draws.

  LLF   longest post-operation lag first, ties by shortest processing time.
        Among the LEGAL pairs, keep those whose candidate operation carries the
        longest post-operation lag (env.true_op_lag, true hours), then the
        shortest true processing time among them. Pair-level, the definition of
        the sister project's rule (FJSP-DRL-2/common_utils.py). With LAG
        inactive every lag is 0 and the rule is SPT. Pair-level instead of
        job-level on purpose: a job-level rule takes its machine from
        available_mch_list_for_job, whose earliest-start tiebreak selects
        nothing under the non-delay mask (p25), so the machine would be drawn
        at random.

Remaining ties are drawn with np.random.choice under np.random.seed(50), set
after set_initial_data exactly as p14 and eval_family do. Every rollout ends
on env.done() and its realized schedule is re-checked by
constraint_family.reference_sim.validate_schedule; a group with any violation
is not saved and the run exits non-zero.

Instances: every descriptor per regime at 10x25, 20x25 and 30x25; the first
--n_prod (default 10) descriptors in sorted order at 50x25 and 80x25, the
production-batch convention of p70. Result files are
test_results/FAMILY/<size>/Result_<RULE>_<regime>.npy, 1-D makespans where
Result_SPT_N.npy is 1-D (p14 sizes) and [makespan, decision-loop seconds] where
it is 2-D (eval_family sizes). The seconds are bookkeeping, not a latency
measurement. The script refuses to start if any target file exists, unless
--force.

Summary (after a run, or alone with --summary_only): per size and regime the
mean makespan of each rule, of SPT and of PACT (per-instance mean over the
seed files Result_greedy+10x25+family+jointblind-v1{,-s302,-s303}_<regime>.npy
that exist), and the mean per-instance gap 100*(rule - ref)/ref, positive =
the rule is WORSE, with a paired-bootstrap 95% CI (20000 resamples, one
default_rng(7) stream drawn in the fixed order size, regime, rule, SPT then
PACT, then the pooled rows per size; p11's estimator). SPT and PACT rows are
sliced to the instances the rule ran on. Writes results/aware_rules.json,
results/aware_rules.md and, when 10x25 and 80x25 are both complete,
paper/macros_aware.tex.

Usage:
  python scripts/p72_aware_rules.py [--sizes 10x25,20x25,30x25,50x25,80x25]
      [--rules ECT,SSPT,LLF] [--procs 8] [--n_prod 10] [--force]
  python scripts/p72_aware_rules.py --summary_only
  python scripts/p72_aware_rules.py --smoke
"""
import os

# one thread per process, set before numpy is imported so forked workers
# inherit it; no GPU is ever needed by a dispatching rule
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import glob
import json
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
os.chdir(ROOT)
_ARGV = sys.argv[1:]
sys.argv = sys.argv[:1]

import numpy as np

NEW_RULES = ('ECT', 'SSPT', 'LLF')
ALLOWED = NEW_RULES + ('ECTND',)
REGIMES = ('N', 'L', 'S', 'W', 'LS', 'LW', 'SW', 'LSW')
SIZES = ('10x25', '20x25', '30x25', '50x25', '80x25')
PROD = ('50x25', '80x25')
SEED = 50                       # the seed every published rule rollout used
POLICY = 'greedy+10x25+family+jointblind-v1'
POLICY_RE = re.compile(r'^Result_greedy\+10x25\+family\+jointblind-v1'
                       r'(-s30[23])?_([A-Z]+)\.npy$')
N_BOOT = 20000
MACROS = 'paper/macros_aware.tex'
JSON_OUT = 'results/aware_rules.json'
MD_OUT = 'results/aware_rules.md'
SIZEW = {'10x25': 'Ten', '80x25': 'Eighty'}


def inst_base(size):
    return f'data/PPVC/{size}+ppvc-mixed'


def result_path(size, tag, regime):
    return f'test_results/FAMILY/{size}/Result_{tag}_{regime}.npy'


# ---------------------------------------------------------------------------
# the rules
# ---------------------------------------------------------------------------

def _draw(key):
    """Uniform draw among the minima of a flattened [J*M] key (SPT's draw)."""
    key = key.reshape(-1)
    return int(np.random.choice(np.where(key == key.min())[0]))


def select_action(rule, env):
    """Action index job*M + machine for one decision of `rule` (see module doc)."""
    if rule == 'SPT':
        from common_utils import heuristic_select_action
        return int(heuristic_select_action('SPT', env))
    legal = ~env.dynamic_pair_mask[0]                              # [J, M]
    if rule in ('ECT', 'ECTND'):
        start = env._compute_pair_free_time()[0]                   # normalized
        done_h = np.round((start + env.candidate_pt[0]) / env._slope, 6)
        allowed = (~env.candidate_process_relation[0] if rule == 'ECT'
                   else legal)
        return _draw(np.where(allowed, done_h, np.inf))
    tp = env.true_op_pt[0][env.candidate[0]].astype(float)         # [J, M] h
    if rule == 'SSPT':
        s = env._setup_into_candidates(true_side=True)[0]          # [J, M] h
        return _draw(np.where(legal, s + tp, np.inf))
    if rule == 'LLF':
        lag = env.true_op_lag[0][env.candidate[0]].astype(float)   # [J] h
        lagp = np.where(legal, lag[:, None], -np.inf)
        longest = legal & (lagp == lagp.max())
        return _draw(np.where(longest, tp, np.inf))
    raise ValueError(f'unknown rule {rule}')


# ---------------------------------------------------------------------------
# one rollout
# ---------------------------------------------------------------------------

def _init_worker():
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    try:
        import torch
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass                # already set in this process
    except ImportError:
        pass


def rollout(task):
    """task = (size, regime, rule, k, desc_path, instance_stem).
    Returns (size, regime, rule, k, makespan, seconds, violations)."""
    size, regime, rule, k, dp, stem = task
    from params import configs
    configs.device = 'cpu'
    from constraint_family.descriptor import ConstraintDescriptor
    from constraint_family.fjsp_env_family import FJSPEnvFamily
    from constraint_family.reference_sim import validate_schedule
    from ppvc_instance_generator import load_instance
    jl, pt, _ = load_instance(stem)
    with open(dp) as f:
        desc = ConstraintDescriptor.from_json(json.load(f))
    M = pt.shape[1]
    env = FJSPEnvFamily(jl.shape[0], M, family_mode=True)
    env.set_initial_data([jl], [pt], descriptor_list=[desc])
    amch = np.full(pt.shape[0], -1, dtype=int)
    np.random.seed(SEED)
    t0 = time.perf_counter()
    while not env.done().all():
        a = select_action(rule, env)
        amch[env.candidate[0, a // M]] = a % M
        env.step(np.array([a]))
    dt = time.perf_counter() - t0
    viol = []
    if (amch < 0).any():
        viol.append('an operation was never assigned a machine')
    else:
        op_ct = env.true_op_ct[0]
        op_st = op_ct - pt[np.arange(pt.shape[0]), amch]
        viol = validate_schedule(jl, pt, desc, op_st, op_ct, amch)
    return size, regime, rule, k, float(env.current_makespan[0]), dt, viol[:3]


# ---------------------------------------------------------------------------
# task list, smoke test, campaign
# ---------------------------------------------------------------------------

def descriptors(size, regime, n_prod):
    descs = sorted(glob.glob(f'data/FAMILY/{size}/desc/{regime}/*.desc.json'))
    if size in PROD:
        descs = descs[:n_prod]
    if not descs:
        sys.exit(f'no descriptors under data/FAMILY/{size}/desc/{regime}')
    out = []
    for dp in descs:
        stem = os.path.join(inst_base(size),
                            os.path.basename(dp).replace('.desc.json', ''))
        if not os.path.exists(stem + '.fjs'):
            sys.exit(f'instance missing for {dp}: {stem}.fjs')
        out.append((dp, stem))
    return out


def stored(size, tag, regime):
    p = result_path(size, tag, regime)
    if not os.path.exists(p):
        return None
    a = np.load(p)
    return (a[:, 0] if a.ndim > 1 else a).astype(float)


def smoke():
    """2 instances of N and LSW at 10x25, in this process, nothing written."""
    _init_worker()
    rules = ('SPT', 'ECTND') + NEW_RULES
    ok = True
    t_all = time.perf_counter()
    for regime in ('N', 'LSW'):
        ref = stored('10x25', 'SPT', regime)
        ms = {}
        for rule in rules:
            ms[rule] = []
            for k, (dp, stem) in enumerate(descriptors('10x25', regime, 0)[:2]):
                _, _, _, _, m, dt, viol = rollout(
                    ('10x25', regime, rule, k, dp, stem))
                ms[rule].append(m)
                print(f'{regime:4s} {rule:6s} inst {k}: makespan {m:7.1f}  '
                      f'{dt:6.2f} s  feasible={not viol}', flush=True)
                if viol:
                    ok = False
                    print(f'   VIOLATION: {viol}')
        # the driver must reproduce the published SPT numbers
        if ref is None or not np.array_equal(np.array(ms['SPT']), ref[:2]):
            ok = False
            print(f'   FAIL {regime}: SPT {ms["SPT"]} vs stored '
                  f'{None if ref is None else ref[:2].tolist()}')
        else:
            print(f'   ok {regime}: SPT reproduces Result_SPT_{regime}.npy[:2]')
        # identities that must hold exactly
        must = ['ECTND', 'SSPT', 'LLF'] if regime == 'N' else []
        for rule in must:
            if ref is not None and np.array_equal(np.array(ms[rule]), ref[:2]):
                print(f'   ok {regime}: {rule} == stored SPT')
            else:
                ok = False
                print(f'   FAIL {regime}: {rule} {ms[rule]} != stored SPT')
    print(f'smoke total {time.perf_counter() - t_all:.1f} s')
    if not ok:
        sys.exit('SMOKE TEST FAILED')
    print('SMOKE TEST PASSED (no files written)')


def campaign(sizes, rules, procs, n_prod, force):
    tasks, expected, two_col = [], defaultdict(int), {}
    for size in sizes:
        spt_n = result_path(size, 'SPT', 'N')
        if not os.path.exists(spt_n):
            sys.exit(f'{spt_n} missing: cannot fix the output convention')
        two_col[size] = np.load(spt_n).ndim == 2
        for regime in REGIMES:
            for rule in rules:
                for k, (dp, stem) in enumerate(descriptors(size, regime, n_prod)):
                    tasks.append((size, regime, rule, k, dp, stem))
                    expected[(size, regime, rule)] += 1
    clash = [result_path(s, r, g) for (s, g, r) in expected
             if os.path.exists(result_path(s, r, g))]
    if clash and not force:
        sys.exit('REFUSING: result files already exist (pass --force to '
                 'overwrite):\n  ' + '\n  '.join(clash))
    # largest instances first so the long rollouts do not trail the pool
    tasks.sort(key=lambda t: -SIZES.index(t[0]))
    print(f'{len(tasks)} rollouts on {procs} procs', flush=True)

    got = defaultdict(dict)
    bad = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=procs, initializer=_init_worker) as ex:
        futs = [ex.submit(rollout, t) for t in tasks]
        for i, fu in enumerate(as_completed(futs), 1):
            size, regime, rule, k, ms, dt, viol = fu.result()
            key = (size, regime, rule)
            got[key][k] = (ms, dt, viol)
            if viol:
                bad.append((key, k, viol))
            if len(got[key]) == expected[key]:
                rows = [got[key][j] for j in range(expected[key])]
                if any(r[2] for r in rows):
                    print(f'  NOT SAVED {key}: infeasible schedule', flush=True)
                else:
                    ms_arr = np.array([r[0] for r in rows])
                    arr = (np.stack([ms_arr, np.array([r[1] for r in rows])], 1)
                           if two_col[size] else ms_arr)
                    np.save(result_path(size, rule, regime), arr)
            if i % 200 == 0 or i == len(tasks):
                print(f'  [{i}/{len(tasks)}] {(time.time() - t0) / 60:.1f} min',
                      flush=True)
    if bad:
        for key, k, viol in bad[:20]:
            print(f'VIOLATION {key} inst {k}: {viol}')
        sys.exit(f'{len(bad)} infeasible rollouts; affected groups not saved')
    print(f'campaign done in {(time.time() - t0) / 60:.1f} min')


# ---------------------------------------------------------------------------
# summary, markdown, macros
# ---------------------------------------------------------------------------

rng = np.random.default_rng(7)


def ci(d, B=N_BOOT):
    """Paired-bootstrap mean and 95% percentile interval (p11's estimator)."""
    n = len(d)
    idx = rng.integers(0, n, (B, n))
    bs = d[idx].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def pact(size, regime):
    """Per-instance mean over the PACT seed files found for (size, regime)."""
    d = f'test_results/FAMILY/{size}'
    names = sorted(os.path.basename(p) for p in
                   glob.glob(f'{d}/Result_greedy+10x25+family+jointblind-v1*_{regime}.npy'))
    names = [n for n in names
             if (m := POLICY_RE.match(n)) and m.group(2) == regime]
    if not names:
        return None, []
    arrs = []
    for n in names:
        a = np.load(os.path.join(d, n))
        arrs.append((a[:, 0] if a.ndim > 1 else a).astype(float))
    if len({len(a) for a in arrs}) != 1:
        sys.exit(f'PACT seed files disagree in length at {size}/{regime}: {names}')
    return np.mean(arrs, axis=0), names


def summarize(rules):
    rules = [r for r in rules if r in NEW_RULES] or list(NEW_RULES)
    sizes = [s for s in SIZES
             if all(os.path.exists(result_path(s, r, g))
                    for r in rules for g in REGIMES)]
    if not sizes:
        print('summary: no size has every rule and regime on disk; nothing written')
        return
    out = {'convention': 'gap = mean over instances of 100*(rule - ref)/ref, '
                         'positive = rule worse; paired bootstrap 95% CI, '
                         f'{N_BOOT} resamples, default_rng(7)',
           'policy': POLICY, 'rules': rules, 'sizes': {}}
    identity_fail = []
    for size in sizes:
        S = {'regimes': {}, 'pooled': {}}
        pooled = {r: {'spt': [], 'pact': []} for r in rules}
        for regime in REGIMES:
            R = {}
            spt = stored(size, 'SPT', regime)
            pol, names = pact(size, regime)
            vals = {r: stored(size, r, regime) for r in rules}
            n = len(vals[rules[0]])
            if spt is None or len(spt) < n:
                sys.exit(f'SPT missing or short at {size}/{regime}')
            if pol is None or len(pol) < n:
                sys.exit(f'PACT seeds missing or short at {size}/{regime}')
            spt, pol = spt[:n], pol[:n]
            R.update(n=n, pact_files=names, mean_SPT=float(spt.mean()),
                     mean_PACT=float(pol.mean()))
            for r in rules:
                v = vals[r]
                assert len(v) == n, (size, regime, r)
                gs = 100 * (v - spt) / spt
                gp = 100 * (v - pol) / pol
                m, lo, hi = ci(gs)
                mp, lop, hip = ci(gp)
                R[f'mean_{r}'] = float(v.mean())
                R[f'gap_SPT_{r}'] = [m, lo, hi]
                R[f'gap_PACT_{r}'] = [mp, lop, hip]
                pooled[r]['spt'].append(gs)
                pooled[r]['pact'].append(gp)
                # identities the definitions force
                must = ((r == 'SSPT' and 'S' not in regime) or
                        (r == 'LLF' and 'L' not in regime))
                if must:
                    same = bool(np.array_equal(v, spt))
                    R[f'identity_{r}_eq_SPT'] = same
                    if not same:
                        identity_fail.append((size, regime, r))
            S['regimes'][regime] = R
        for r in rules:
            gs = np.concatenate(pooled[r]['spt'])
            gp = np.concatenate(pooled[r]['pact'])
            S['pooled'][r] = {'gap_SPT': list(ci(gs)), 'gap_PACT': list(ci(gp)),
                              'n': int(len(gs))}
        best = min(rules, key=lambda r: S['pooled'][r]['gap_SPT'][0])
        S['best_rule'] = best
        S['best_beats_SPT_regimes'] = [
            g for g in REGIMES if S['regimes'][g][f'gap_SPT_{best}'][0] < 0]
        S['best_beats_SPT_regimes_ci'] = [
            g for g in REGIMES if S['regimes'][g][f'gap_SPT_{best}'][2] < 0]
        out['sizes'][size] = S
    out['identity_failures'] = identity_fail

    os.makedirs('results', exist_ok=True)
    with open(JSON_OUT, 'w') as f:
        json.dump(out, f, indent=1)
    print(f'wrote {JSON_OUT}')
    write_md(out)
    if identity_fail:
        print('IDENTITY FAILURES (rule should equal SPT here):', identity_fail)
    write_macros(out, refuse_reason=('identity checks failed: '
                                     f'{identity_fail}' if identity_fail else None))


def _g(t):
    return f'{t[0]:+.2f} [{t[1]:+.2f}, {t[2]:+.2f}]'


def write_md(out):
    rules = out['rules']
    L = ['# Constraint-aware dispatching rules (scripts/p72_aware_rules.py)', '',
         f'Gap convention: {out["convention"]}. PACT = per-instance mean over '
         f'the {POLICY} seed files found. Rules: ECT = earliest completion '
         'over compatible pairs (active set); SSPT = shortest setup plus '
         'processing over legal pairs; LLF = longest post-operation lag first, '
         'ties by SPT, over legal pairs.', '']
    for size, S in out['sizes'].items():
        L += [f'## {size}', '',
              '| regime | n | SPT | PACT | ' + ' | '.join(rules) + ' |',
              '|---|---|---|---|' + '---|' * len(rules)]
        for g in REGIMES:
            R = S['regimes'][g]
            L.append(f'| {g} | {R["n"]} | {R["mean_SPT"]:.1f} | '
                     f'{R["mean_PACT"]:.1f} | ' +
                     ' | '.join(f'{R[f"mean_{r}"]:.1f}' for r in rules) + ' |')
        L += ['', '| regime | ' + ' | '.join(f'{r} vs SPT' for r in rules) +
              ' | ' + ' | '.join(f'{r} vs PACT' for r in rules) + ' |',
              '|---|' + '---|' * (2 * len(rules))]
        for g in REGIMES:
            R = S['regimes'][g]
            L.append(f'| {g} | ' +
                     ' | '.join(_g(R[f'gap_SPT_{r}']) for r in rules) + ' | ' +
                     ' | '.join(_g(R[f'gap_PACT_{r}']) for r in rules) + ' |')
        P = S['pooled']
        L.append('| pooled | ' + ' | '.join(_g(P[r]['gap_SPT']) for r in rules) +
                 ' | ' + ' | '.join(_g(P[r]['gap_PACT']) for r in rules) + ' |')
        L += ['', f'Best new rule (lowest pooled gap to SPT): {S["best_rule"]}; '
              f'mean gap to SPT below zero in {len(S["best_beats_SPT_regimes"])}/8 '
              f'regimes {S["best_beats_SPT_regimes"]}, CI below zero in '
              f'{len(S["best_beats_SPT_regimes_ci"])}/8.', '']
    if out['identity_failures']:
        L += ['IDENTITY FAILURES: ' + str(out['identity_failures']), '']
    with open(MD_OUT, 'w') as f:
        f.write('\n'.join(L) + '\n')
    print(f'wrote {MD_OUT}')


def write_macros(out, refuse_reason=None):
    from macro_io import emit, refuse
    if refuse_reason:
        refuse(MACROS, refuse_reason)
    missing = [s for s in SIZEW if s not in out['sizes']]
    if missing:
        print(f'macros not written: sizes {missing} incomplete')
        return
    lines = []
    for size, w in SIZEW.items():
        S = out['sizes'][size]
        b = S['best_rule']
        gs, gp = S['pooled'][b]['gap_SPT'], S['pooled'][b]['gap_PACT']
        lines += [
            (f'AwareBest{w}', b, f'{size}: lowest pooled gap to SPT'),
            (f'AwareGapSpt{w}', f'{gs[0]:.1f}', f'{size}: pooled gap of {b} to SPT, %'),
            (f'AwareGapSpt{w}Lo', f'{gs[1]:.1f}', '95% CI low'),
            (f'AwareGapSpt{w}Hi', f'{gs[2]:.1f}', '95% CI high'),
            (f'AwareGapPact{w}', f'{gp[0]:.1f}', f'{size}: pooled gap of {b} to PACT, %'),
            (f'AwareGapPact{w}Lo', f'{gp[1]:.1f}', '95% CI low'),
            (f'AwareGapPact{w}Hi', f'{gp[2]:.1f}', '95% CI high'),
            (f'AwareBeatsSpt{w}', str(len(S['best_beats_SPT_regimes'])),
             f'{size}: regimes (of 8) where {b} has a mean gap to SPT below 0'),
            (f'AwareBeatsSptCI{w}', str(len(S['best_beats_SPT_regimes_ci'])),
             f'{size}: regimes (of 8) where the CI of that gap lies below 0'),
        ]
    emit(MACROS, lines, header='Generated by scripts/p72_aware_rules.py from '
                               'test_results/FAMILY/<size>/Result_{ECT,SSPT,LLF}_*.npy')
    with open(MACROS) as f:
        print(f.read())


# ---------------------------------------------------------------------------

def main():
    a = list(_ARGV)
    if '--smoke' in a:
        smoke()
        return
    rules = (a[a.index('--rules') + 1].split(',') if '--rules' in a
             else list(NEW_RULES))
    bad = [r for r in rules if r not in ALLOWED]
    if bad:
        sys.exit(f'unknown rules {bad}; allowed {ALLOWED}')
    if '--summary_only' in a:
        summarize(rules)
        return
    sizes = (a[a.index('--sizes') + 1].split(',') if '--sizes' in a
             else list(SIZES))
    bad = [s for s in sizes if s not in SIZES]
    if bad:
        sys.exit(f'unknown sizes {bad}; allowed {SIZES}')
    procs = int(a[a.index('--procs') + 1]) if '--procs' in a else 8
    n_prod = int(a[a.index('--n_prod') + 1]) if '--n_prod' in a else 10
    campaign(sizes, rules, procs, n_prod, '--force' in a)
    summarize(rules)


if __name__ == '__main__':
    main()
