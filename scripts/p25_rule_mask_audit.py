"""P25: what action set do the dispatching rules actually search?

The manuscript said, in three places, that all four rules "select their machine
under the same dynamic feasibility mask the policy's actor receives, so their
action set is identical to the policy's". For SPT that is true. For FIFO, MOR
and MWKR it is not, and the claim was load-bearing: those three are three of the
four ingredients of the per-instance rule PORTFOLIO, which is both the "pool"
column of Table I and the incumbent CP-SAT is warm-started from.

Two defects, both in common_utils.available_mch_list_for_job:

  1. The machine tiebreak is vacuous, provably. The environment sets
     next_schedule_time = min(pair_free_time) over all compatible pairs
     (fjsp_env_same_op_nums.py:480) and then masks every pair whose
     pair_free_time exceeds it (:539). So a pair is legal iff its pair_free_time
     EQUALS next_schedule_time -- every legal machine ties, always. The
     "ties broken by earliest legal start" in the docstring selects nothing, and
     np.random.choice picks the machine uniformly at random. On a FLEXIBLE job
     shop these three rules therefore have no machine-selection rule at all.

  2. The mask is abandoned. Job choice ranges over READY jobs, which is a strict
     superset of jobs holding a legal pair: a changeover that will not fit, or an
     outage in progress, can block every machine of a ready job. When that
     happens the function falls back to the STATIC compatibility relation ranked
     by raw mch_free_time -- a clock its own docstring calls "not a legal-start
     test under SETUP/WINDOW". Those are actions the actor cannot take; it sees
     them as -inf.

SPT escapes both because it ranks (job, machine) PAIRS jointly inside the mask.
It is also the strongest of the four in every regime and at every size, so every
"best rule" comparison in the paper is against the one rule that is clean. That
is why the headline survives; it is not why the sentence was true.

This script measures both defects and the size of the baseline they cost, and
emits macros_rules.tex. It does not change any reported result: the .npy files
on disk remain the rules as published. It exists so the manuscript's disclosure
of what those rules are is a recomputable number rather than a claim.

REPAIRED rule = the rule the manuscript used to describe: priority key applied
over jobs that hold a legal pair, machine chosen inside the mask by shortest
processing time (exactly what SPT already does).

Usage: python scripts/p25_rule_mask_audit.py [--regimes N,L,...] [--limit K]
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_ARGV = sys.argv[1:]


def _pop(flag, default=None):
    if flag in _ARGV:
        i = _ARGV.index(flag)
        v = _ARGV[i + 1]
        del _ARGV[i:i + 2]
        return v
    return default


ARG_REGIMES = _pop('--regimes', 'N,L,S,W,LS,LW,SW,LSW').split(',')
ARG_LIMIT = int(_pop('--limit', '100'))
sys.argv = ['x', '--device', 'cpu']

import numpy as np
from params import configs  # noqa: F401  (parses sys.argv; sets device)
from constraint_family.descriptor import ConstraintDescriptor
from constraint_family.fjsp_env_family import FJSPEnvFamily
from ppvc_instance_generator import load_instance
import common_utils

BASE, INST = 'data/FAMILY/10x25', 'data/PPVC/10x25+ppvc-mixed'
RULES = ['FIFO', 'MOR', 'SPT', 'MWKR']
TWO_STAGE = ['FIFO', 'MOR', 'MWKR']          # the ones that pick a job first
SETUP_BEARING = ['S', 'LS', 'SW', 'LSW']     # where the defect bites


def repaired_action(rule, env):
    """The rule the manuscript described: job AND machine inside the mask."""
    M = env.number_of_machines
    legal = ~env.dynamic_pair_mask[0]                 # pairs that can START now
    if rule == 'SPT':                                 # already joint-in-mask
        pt = np.where(legal, env.candidate_pt[0], np.inf)
        return int(np.random.choice(np.where(pt.reshape(-1) == pt.min())[0]))
    # next_schedule_time is the MIN pair_free_time, so some pair always ties it:
    # the legal-job set is never empty while the episode runs, and the static
    # fallback of common_utils is never needed.
    legal_jobs = np.where(legal.any(axis=1))[0]
    ops = env.candidate[0][legal_jobs]
    if rule == 'FIFO':
        key = env.candidate_free_time[0][legal_jobs]
        pick = legal_jobs[common_utils.min_element_index(key)]
    elif rule == 'MOR':
        key = env.op_match_job_left_op_nums[0][ops]
        pick = legal_jobs[common_utils.max_element_index(key)]
    else:  # MWKR
        key = env.op_match_job_remain_work[0][ops]
        pick = legal_jobs[common_utils.max_element_index(key)]
    j = int(np.random.choice(np.atleast_1d(pick).ravel()))
    pt = np.where(legal[j], env.candidate_pt[0][j], np.inf)
    return j * M + int(np.random.choice(np.where(pt == pt.min())[0]))


def cases(regime):
    out = []
    for dp in sorted(glob.glob(f'{BASE}/desc/{regime}/*.desc.json'))[:ARG_LIMIT]:
        name = os.path.basename(dp).replace('.desc.json', '')
        with open(dp) as f:
            desc = ConstraintDescriptor.from_json(json.load(f))
        out.append((desc, load_instance(os.path.join(INST, name))))
    return out


def rollout(rule, desc, inst, repaired, count=None):
    jl, pt, meta = inst
    np.random.seed(200)
    env = FJSPEnvFamily(jl.shape[0], pt.shape[1], family_mode=True)
    env.set_initial_data([jl], [pt], descriptor_list=[desc],
                         op_type_list=[meta['op_type']],
                         mch_type_list=[meta['mch_type']])
    while not env.done().all():
        if repaired:
            a = repaired_action(rule, env)
        else:
            a = common_utils.heuristic_select_action(rule, env)
            if count is not None:
                count['n'] += 1
                # did the rule commit a job that CANNOT start on any machine now?
                j = a // env.number_of_machines
                if not (~env.dynamic_pair_mask[0, j]).any():
                    count['escape'] += 1
                # is the legal-machine set fully tied? (the vacuous tiebreak)
                for jj in np.where(~env.dynamic_pair_mask[0].any(axis=1) == 0)[0]:
                    st = env.pair_free_time[0, jj][~env.dynamic_pair_mask[0, jj]]
                    if st.size:
                        count['sets'] += 1
                        count['tied'] += int(np.all(st == st[0]))
        env.step(np.array([a]))
    return float(env.current_makespan[0])



# The analysis + macro-writing body runs only when this file is executed
# directly. Guarded so `from p25_rule_mask_audit import repaired_action` (used by
# scripts/p41_repaired_warm_cache.py) imports the rule logic without side effects.
if __name__ == '__main__':
    escape = {r: {} for r in RULES}
    gain = {r: {} for r in RULES}
    repaired_ms = {}          # regime -> rule -> per-instance repaired makespans
    tied_sets = tied_hits = 0

    for reg in ARG_REGIMES:
        cs = cases(reg)
        repaired_ms[reg] = {}
        for rule in RULES:
            c = {'n': 0, 'escape': 0, 'sets': 0, 'tied': 0}
            pub = np.array([rollout(rule, d, i, False, c) for d, i in cs])
            rep = np.array([rollout(rule, d, i, True) for d, i in cs])
            repaired_ms[reg][rule] = rep
            escape[rule][reg] = 100.0 * c['escape'] / max(c['n'], 1)
            gain[rule][reg] = 100.0 * (pub.mean() - rep.mean()) / pub.mean()
            tied_sets += c['sets']
            tied_hits += c['tied']
        print(f'{reg:4} escape% ' +
              ' '.join(f'{r}={escape[r][reg]:5.1f}' for r in RULES) +
              '  | repair gain% ' +
              ' '.join(f'{r}={gain[r][reg]:+5.2f}' for r in RULES), flush=True)

    esc_setup = [escape[r][g] for r in TWO_STAGE
                 for g in SETUP_BEARING if g in escape[r]]
    gain_all = [gain[r][g] for r in TWO_STAGE for g in gain[r]]
    spt_gain = max(abs(v) for v in gain['SPT'].values())
    tied_pct = 100.0 * tied_hits / max(tied_sets, 1)

    # How often is the four-rule PORTFOLIO just SPT? Table I's footnote (b) used to
    # call the pool "stronger than any single rule". A portfolio cannot be WEAKER
    # than its best member, but it is only STRICTLY stronger where some other rule
    # wins an instance -- and SPT dominates the other three per-instance nearly
    # everywhere. Read from the published arrays on disk, which are what Table I
    # prints; no rollout needed.
    RES = 'test_results/FAMILY/10x25'


    def _disk(tag, r):
        a = np.load(f'{RES}/Result_{tag}_{r}.npy')
        return (a[:, 0] if a.ndim > 1 else a).astype(float)


    pool_is_spt = 0
    for reg in ARG_REGIMES:
        try:
            cols = {r: _disk(r, reg) for r in RULES}
        except FileNotFoundError:
            continue
        pool = np.min(np.stack([cols[r] for r in RULES]), axis=0)
        pool_is_spt += int(np.array_equal(pool, cols['SPT']))

    # The one conclusion the two-stage rules reach: on the DAGGERED regimes the 5s
    # CP-SAT cell IS the portfolio (the search improved on its warm start on zero
    # instances), so a weak portfolio inflates the policy's margin there. Re-measure
    # that margin against a REPAIRED portfolio and report whether the conclusion --
    # "an exact solve under ten seconds returns nothing better than the rule it
    # started from, and the policy does" -- survives it. Reported, not hidden: this
    # is the number the E5 paragraph now quotes.
    DAGGERED = ['S', 'LS', 'SW', 'LSW']
    J = 'greedy+10x25+family+joint-v1'
    # The "repaired portfolio" this block reports must be THE hint the solver was
    # handed -- the same per-instance values the daggered 5s cells of Table I
    # credit (the fairwarmR re-solves improved on none of them) -- and NOT a
    # fresh re-roll of the same rules by this script: the re-roll's uniform
    # tie-breaks land differently per instance and drift the mean by up to
    # 0.3 h, which would make this table disagree with Table I about cells the
    # prose says are equal. The re-rolls above stay: they feed the per-rule
    # repair-gain macros, which no other table cross-references.
    HINT_CACHE = 'or_solution/FAMILY/warmstart_cache_repaired/makespans.json'
    _hint = json.load(open(HINT_CACHE))
    marg_old, marg_new, ahead_old, ahead_new = [], [], 0, 0
    marg_rows = []
    for reg in DAGGERED:
        if reg not in ARG_REGIMES:
            continue
        try:
            pol = np.mean([_disk(f'{J}{s}', reg)
                           for s in ('', '-s302', '-s303')], axis=0).mean()
            old_pool = np.min(np.stack([_disk(r, reg) for r in RULES]),
                              axis=0).mean()
        except FileNotFoundError:
            continue
        hint_vals = [v['repaired_ms'] for v in _hint[reg].values()]
        assert len(hint_vals) == len(repaired_ms[reg][RULES[0]]), \
            f'{reg}: hint cache holds {len(hint_vals)} instances, audit ran ' \
            f'{len(repaired_ms[reg][RULES[0]])}'
        new_pool = float(np.mean(hint_vals))
        own_pool = np.min(np.stack([repaired_ms[reg][r] for r in RULES]),
                          axis=0).mean()
        print(f'  hint-vs-reroll drift {reg:4}: hint {new_pool:7.2f}  '
              f'own re-roll {own_pool:7.2f}  (tie-break drift '
              f'{own_pool - new_pool:+.2f} h; the hint is what Table I credits)')
        marg_old.append(old_pool - pol)
        marg_new.append(new_pool - pol)
        ahead_old += int(old_pool > pol)
        ahead_new += int(new_pool > pol)
        # The main paper promises the supplement these margins regime by regime, so
        # they have to leave here as macros. A range in the prose is not the promise.
        marg_rows.append((reg, pol, old_pool, new_pool))
        print(f'  CP5s {reg:4} policy {pol:7.2f} | portfolio published {old_pool:7.2f} '
              f'(margin {old_pool - pol:+5.2f}) | repaired {new_pool:7.2f} '
              f'(margin {new_pool - pol:+5.2f})')

    print()
    print(f'legal-machine sets fully tied : {tied_pct:.1f}% ({tied_hits}/{tied_sets})'
          '  -> the machine tiebreak selects nothing')
    print(f'two-stage rules outside the mask, setup regimes: '
          f'{min(esc_setup):.0f}-{max(esc_setup):.0f}% of decisions')
    print(f'repair gain, two-stage rules  : up to {max(gain_all):.1f}%')
    print(f'repair gain, SPT (the control): {spt_gain:.3f}%  (must be ~0)')

    out = [
        '% Auto-generated by scripts/p25_rule_mask_audit.py -- do not hand-edit.',
        '% What the dispatching rules actually search. SPT ranks (job, machine) pairs',
        '% jointly inside the policy mask; FIFO/MOR/MWKR pick a job first and can',
        '% commit it to a machine that cannot start it now. SPT is the best rule in',
        '% every regime, so every "best rule" comparison in the paper is clean.',
        f'\\newcommand{{\\RuleTiedPct}}{{{tied_pct:.0f}}}'
        '  % legal-machine sets that are fully tied (the tiebreak is vacuous)',
        f'\\newcommand{{\\RuleEscapeLo}}{{{min(esc_setup):.0f}}}'
        f'\\newcommand{{\\RuleEscapeHi}}{{{max(esc_setup):.0f}}}'
        '  % FIFO/MOR/MWKR decisions outside the actor mask, setup-bearing regimes (%)',
        f'\\newcommand{{\\RuleRepairMax}}{{{max(gain_all):.0f}}}'
        '  % most a mask-restricted repair improves FIFO/MOR/MWKR (%)',
        f'\\newcommand{{\\RuleRepairSpt}}{{{spt_gain:.1f}}}'
        '  % ... and SPT, the control, which is already mask-restricted (%)',
        # \RuleRepairMax is a max over the THREE two-stage rules, so it is nobody's
        # number in particular -- it happens to be MOR's on S. Quoting it as "MWKR is
        # weaker by up to this much" is wrong, and the E6 caption did exactly that.
        # Emit the per-rule maxima so a caption can name the rule it is talking about.
        f'\\newcommand{{\\RuleRepairMwkr}}{{{max(gain["MWKR"].values()):.0f}}}'
        f'\\newcommand{{\\RuleRepairFifo}}{{{max(gain["FIFO"].values()):.0f}}}'
        f'\\newcommand{{\\RuleRepairMor}}{{{max(gain["MOR"].values()):.0f}}}'
        '  % per-rule maximum repair gain, over all regimes (%)',
        f'\\newcommand{{\\RuleAuditSize}}{{10}}'
        '  % the shop size the mask audit was run at (modules); E6 is 20 and 30',
        f'\\newcommand{{\\PoolIsSptN}}{{{pool_is_spt}}}'
        f'\\newcommand{{\\PoolN}}{{{len(ARG_REGIMES)}}}'
        '  % regimes where the 4-rule portfolio is EXACTLY SPT, per-instance',
    ]
    if marg_new:
        out.append(
            f'\\newcommand{{\\CpFiveAheadRepaired}}{{{ahead_new}}}'
            f'\\newcommand{{\\CpFiveDaggerN}}{{{len(marg_new)}}}'
            f'  % daggered regimes where the policy still leads a REPAIRED portfolio'
            f' at 5s (published margins {min(marg_old):.1f}-{max(marg_old):.1f} h,'
            f' repaired {min(marg_new):.1f}-{max(marg_new):.1f} h)')
        # Regime by regime, because that is what the main text sends the reader here
        # for. Emitting only the range would leave the pointer as broken as it was.
        out.append('% Per-regime, for the repaired-portfolio table in the supplement:')
        for reg, pol, old_pool, new_pool in marg_rows:
            out.append(
                f'\\newcommand{{\\Rep{reg}Pol}}{{{pol:.1f}}}'
                f'\\newcommand{{\\Rep{reg}Pub}}{{{old_pool:.1f}}}'
                f'\\newcommand{{\\Rep{reg}Rep}}{{{new_pool:.1f}}}'
                f'\\newcommand{{\\Rep{reg}MPub}}{{{old_pool - pol:+.2f}}}'
                f'\\newcommand{{\\Rep{reg}MRep}}{{{new_pool - pol:+.2f}}}'
                f'  % {reg}: policy, portfolio published, portfolio repaired, and the'
                f' two margins (h)')
    with open('paper/macros_rules.tex', 'w') as f:
        f.write('\n'.join(out) + '\n')
    print('\nwrote paper/macros_rules.tex')
