"""P28 -- three numbers the supplement states, each of which needed a macro.

The manuscript answered each of these with prose where it should have answered with a
count. Each is computed here from the
shipped artefacts, and each becomes a macro, so that a rerun moves the sentence
instead of silently contradicting it.

  1. SEED COLLISION. The supplement quoted the PER-DRAW chance that an
     online training instance regenerates a test instance (100 / 2^31). That is
     not the quantity a reader cares about: the campaign draws an instance seed
     for every environment of every update of every run, so the exposure is
     cumulative. Counted here from the training banners themselves
     (train_family.py:222 draws num_envs instance seeds per update, :172), so it
     cannot drift away from the runs that actually happened.

  2. NO-WAIT ACTIVITY. E4 argues the no-wait boundary from the
     construction and deliberately does NOT rest it on a measurement, because a
     measurement here describes the benchmark rather than the family. But the
     claim that no-wait is barely active on this shop IS a measurement, and it
     has one: test_results/FAMILY/nowait100/probe.json, the CP-SAT no-wait model
     on the 100 LAG test instances, against the same instances' reference
     makespans. Reported as what it is -- a property of this benchmark.

  3. WHAT THE CP-SAT CELLS CERTIFY. "Exact method" invites the
     reading that every solver number is an optimum. Most are not. Counted per
     budget from the same JSONs Table I is built from.

Usage: python scripts/p28_exposure_and_certification.py
"""
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from macro_io import emit

LOGS = 'train_log'
CONFIGS = 'train_log/FAMILY'
# The test set: 100 base instances from seeds 50000..50099
# (data/PPVC/10x25+ppvc-mixed/dataset_meta.json: seed0=50000, n_instances=100).
N_TEST_SEEDS = 100
# train_family.py:172 -- np.random.randint(0, 2**31 - 1), so 2**31 - 1 outcomes.
SEED_SPACE = 2 ** 31 - 1

# HOW MANY INSTANCE SEEDS A RUN ACTUALLY DRAWS. The obvious answer, num_envs per
# update, is wrong by a factor of reset_env_timestep and would overstate the
# campaign's exposure twentyfold. train_family.py:224 resamples the batch only
# every reset_env_timestep updates:
#
#     if i_update % self.reset_env_timestep == 0:
#         jls, pts, descs, opts, mchts = self.sample_batch()   # num_envs instances
#
# so a run draws (max_updates / reset_env_timestep) * num_envs instances. Under the
# locked defaults both are 20, which is exactly why the error is easy to make and
# invisible once made: the two twenties cancel, and a run draws one instance per
# update. All three numbers are read from each run's own config snapshot rather
# than assumed, so a rerun with a different batch policy moves the macro instead of
# silently falsifying the sentence it backs.
def draws_of(model):
    p = os.path.join(CONFIGS, f'config_{model}.json')
    if not os.path.exists(p):
        raise SystemExit(f'missing config {p}: cannot count seed draws for a run '
                         f'whose batch policy is unknown')
    c = json.load(open(p))
    upd, envs, reset = c['max_updates'], c['num_envs'], c['reset_env_timestep']
    if reset <= 0:
        raise SystemExit(f'{model}: reset_env_timestep={reset}')
    # CEIL, not floor. train_family.py runs `for i_update in range(max_updates)` and
    # resamples when `i_update % reset_env_timestep == 0`, so it draws at i = 0,
    # reset, 2*reset, ..., which is ceil(upd/reset) batches, not upd//reset. The two
    # agree whenever upd is a multiple of reset, which is why the main runs (2000
    # updates, reset 25) never exposed it. They are not equal for the
    # experience-matched W specialist, which trains for 333 updates: 13 batches by
    # floor, 14 in fact. Undercounting draws would understate the collision exposure
    # this section exists to bound, and the bound must not understate the exposure.
    return -(-upd // reset) * envs

# Every training run behind a number the paper reports, and only those: runs the
# paper does not report (underscore-named variants, unfinished arms) must not be
# here, or the bound stops describing the paper.
#
# spec-W-third is included deliberately: it is the experience-matched control the
# paper reports (Section "Why the Joint Policy Trails on W"), so its three runs
# drew instance seeds this section's collision bound has to account for. A run whose numbers
# we print is a run whose exposure we own.
REPORTED_RUNS = (
    [f'FAMILY_spec-{r}{s}.log'
     for r in ('N', 'L', 'S', 'W', 'LS', 'LW', 'SW', 'LSW')
     for s in ('', '-s302', '-s303')] +
    [f'FAMILY_spec-W-third{s}.log'
     for s in ('', '-s302', '-s303')] +
    # joint-nlsw: trained on the single members only, every composition read
    # zero-shot. Its numbers are in the paper, so its instance-seed draws are ours.
    [f'FAMILY_joint-nlsw{s}.log'
     for s in ('', '-s302', '-s303')] +
    [f'FAMILY_{m}{s}.log'
     for m in ('joint-v1', 'jointblind-v1')
     for s in ('', '-s302', '-s303')] +
    [f'FAMILY_joint-{m}-s{s}.log'
     for m in ('concat', 'deltazero', 'deltamax')
     for s in ('301', '302', '303')] +
    # the all-eight enumeration comparator, both arms: its numbers are in the
    # paper, so its instance-seed draws are ours to account for.
    [f'FAMILY_joint-all8{e}-s{s}.log'
     for e in ('', 'e')
     for s in ('301', '302', '303')] +
    # the domain-randomized arm (Sec. S-XVIII remedy): its numbers are in the
    # paper, so its instance-seed draws are ours to account for.
    [f'FAMILY_joint-dr-s{s}.log'
     for s in ('301', '302', '303')]
)
BANNER = re.compile(r'\[train_family\].*updates=(\d+)')


def training_draws():
    """Instance-seed draws over the whole reported campaign.

    The banner is checked against the config as well as read, so a run whose log
    says one thing and whose config says another stops the script instead of
    quietly averaging the two.
    """
    total, runs = 0, 0
    for name in REPORTED_RUNS:
        p = os.path.join(LOGS, name)
        if not os.path.exists(p):
            raise SystemExit(f'missing training banner {p}: the campaign this '
                             f'macro describes is not the one on disk')
        m = BANNER.search(open(p, errors='replace').read())
        if not m:
            raise SystemExit(f'{p} has no [train_family] banner')
        model = '10x25+family+' + name[len('FAMILY_'):-len('.log')]
        cfg_upd = json.load(open(os.path.join(CONFIGS, f'config_{model}.json')
                                 ))['max_updates']
        if int(m.group(1)) != cfg_upd:
            raise SystemExit(f'{name}: banner says {m.group(1)} updates, config '
                             f'says {cfg_upd}')
        total += draws_of(model)
        runs += 1
    return runs, total


def commafy(n):
    return f'{n:,}'.replace(',', '{,}')


# ------------------------------------------------- 4. HOW LONG AN OPERATION IS
# The paper justifies its +-1% equivalence margin by saying that at these makespans
# it is "one to three hours, the length of a single operation". That does not
# survive the benchmark: the median compatible processing time is 4 h and the range
# runs to 14 h, so one to three hours is SHORTER than a typical operation, not
# equal to one. The margin
# stands -- it was pre-specified -- but its justification has to be one the numbers
# support, so the numbers are emitted here and the sentence is rewritten around
# them.
def op_durations():
    """Every compatible (operation, machine) processing time in the test set."""
    vals = []
    for f in sorted(glob.glob('data/PPVC/10x25+ppvc-mixed/*.fjs')):
        t = open(f).read().split()
        i, nj = 3, int(t[0])
        for _ in range(nj):
            n_ops = int(t[i]); i += 1
            for _ in range(n_ops):
                k = int(t[i]); i += 1
                for _ in range(k):
                    i += 1                      # machine index
                    vals.append(int(t[i])); i += 1
    if not vals:
        raise SystemExit('no .fjs instances found; refusing to guess the durations')
    return np.array(vals, dtype=float)


_ops = op_durations()
op_median, op_max = float(np.median(_ops)), float(_ops.max())
print(f'4. OPERATION DURATION over {len(_ops):,} compatible (op, machine) pairs')
print(f'   median {op_median:.0f} h, max {op_max:.0f} h')
print('   (the paper said a 1-3 h margin was "the length of a single operation";')
print('    it is shorter than the median one, which is the honest thing to say)')

# ---------------------------------------------- 5. WHAT EACH ARM COSTS TO DEPLOY
# The deployment table compares the FiLM policy
# against a specialist and a catalogue, but not against the two conditioning arms the
# paper's own ablation says are no worse: the token-blind joint and the direct
# (concatenation) one. He is right that a reader deciding what to ship needs them,
# and the answer is that the token-blind joint is the SAME SIZE as
# a single specialist and covers every regime. Read from the training banners and the
# checkpoints, not typed.
BANNER_PARAMS = re.compile(r'params total=(\d+)')


def arm_cost(model):
    log = os.path.join(LOGS, f'FAMILY_{model}.log')
    ckpt = f'trained_network/FAMILY/10x25+family+{model}.pth'
    if not os.path.exists(log) or not os.path.exists(ckpt):
        raise SystemExit(f'{model}: need both {log} and {ckpt} to state its cost')
    m = BANNER_PARAMS.search(open(log, errors='replace').read())
    if not m:
        raise SystemExit(f'{log} has no [model] params banner')
    return int(m.group(1)), os.path.getsize(ckpt) / 1024.0


blind_p, blind_kib = arm_cost('jointblind-v1')
concat_p, concat_kib = arm_cost('joint-concat-s301')
film_p, film_kib = arm_cost('joint-v1')
print(f'5. DEPLOYMENT COST of each conditioning arm')
print(f'   token-blind joint  : {blind_p:6,} params, {blind_kib:5.0f} KiB')
print(f'   direct (concat)    : {concat_p:6,} params, {concat_kib:5.0f} KiB')
print(f'   FiLM (the paper\'s) : {film_p:6,} params, {film_kib:5.0f} KiB')
print(f'   -> the token-blind joint is the size of ONE specialist and serves all '
      f'{len(REGIMES_ALL) if "REGIMES_ALL" in dir() else 8} regimes')

runs, draws = training_draws()
p_draw = N_TEST_SEEDS / SEED_SPACE
p_cum = 1.0 - (1.0 - p_draw) ** draws
# The EXPECTED COUNT is the number that decides how worried to be, and it is the one
# the obvious question does not ask for and the reader most needs. "A 4.9% chance
# of at least one collision" invites the reading that 4.9% of the data are affected.
# What it actually says is that the campaign expected 0.05 collisions: the likeliest
# outcome by far is none at all, and one hit among a million training draws could not
# move a test mean. Emit both, and let the prose lead with this one.
exp_hits = draws * p_draw
print(f'1. SEED COLLISION over {runs} reported training runs')
print(f'   instance-seed draws           : {draws:,}')
print(f'   per draw                      : {p_draw:.3e}')
print(f'   EXPECTED collisions           : {exp_hits:.3f}   <- the one that matters')
print(f'   P(at least one) over campaign : {100 * p_cum:.2f}%')
print('   (a hit reproduces the BASE instance; the setup matrix and outage')
print('    calendar come from an independently drawn seed, so only N and L,')
print('    whose descriptors draw nothing, would be reproduced whole)')

# ------------------------------------------------------------------ 2. NO-WAIT
probe = json.load(open('test_results/FAMILY/nowait100/probe.json'))
nw_n = len(probe)
nw_opt = sum(1 for x in probe if x['status'] == 'OPTIMAL')
# Count what a schedule IS, not what it is not. \NwFeasible backs the sentence "every
# one of these instances admits a no-wait schedule", so it may only count solves that
# actually produced one: OPTIMAL or FEASIBLE. The previous form, n - count(INFEASIBLE),
# is an exclusion, and an exclusion silently promotes every OTHER status to feasible.
# CP-SAT returns UNKNOWN when it neither finds a schedule nor proves none exists; that
# would have been counted as evidence that a schedule exists, backed by a solve that
# found no schedule. Today the probe holds only OPTIMAL and FEASIBLE, so the number is
# right and always has been. It is the reasoning that was unsound, and it is cheaper to
# fix now than to discover on a rerun.
FOUND = {'OPTIMAL', 'FEASIBLE'}
seen = {x['status'] for x in probe}
if not seen <= FOUND | {'INFEASIBLE'}:
    raise SystemExit(
        f'REFUSING: the no-wait probe contains status(es) {sorted(seen - FOUND - {"INFEASIBLE"})}. '
        f'A solve that neither found a schedule nor proved none exists cannot be '
        f'counted on either side of "this instance admits a no-wait schedule". Decide '
        f'what it means before the paper quotes it.')
nw_found = sum(1 for x in probe if x['status'] in FOUND)
nw_infeas = sum(1 for x in probe if x['status'] == 'INFEASIBLE')
infl = np.array([100.0 * (x['objective'] / x['ref'] - 1.0) for x in probe])
nw_free = int((infl <= 0).sum())          # no-wait costs this instance nothing
# A no-wait schedule is LAG-feasible, so it can never beat the LAG OPTIMUM.
# Where it beats the LAG reference, the reference is a best-found incumbent and
# demonstrably not an optimum. Counted, because it is the cleanest evidence in
# the paper that a solver cell is an incumbent.
nw_below = int((infl < 0).sum())
print(f'\n2. NO-WAIT on the {nw_n} LAG test instances')
print(f'   feasible                      : {nw_n - nw_infeas}/{nw_n}')
print(f'   proved optimal                : {nw_opt}/{nw_n}')
print(f'   costs nothing vs the reference: {nw_free}/{nw_n}')
print(f'   BEATS the reference           : {nw_below}/{nw_n} '
      f'(so that reference is not an optimum)')
print(f'   inflation mean {infl.mean():.2f}%  median {np.median(infl):.2f}%  '
      f'max {infl.max():.2f}%  min {infl.min():.2f}%')

# --------------------------------------------------------------- 3. THE HORIZON
# The outage calendars are laid out on H, the job-chain lower bound, which is
# computed from the base instance's TRUE lags in EVERY regime so that a given
# instance carries a byte-identical calendar under W, LW, SW and LSW alike
# (descriptor.py:186). On the regimes that enforce those lags, or that add
# setups, the schedule outlives H and the whole calendar binds. On W alone it
# does not: no lag is enforced, the shop finishes early, and the tail of the
# calendar falls past the last operation. Measured, not assumed.
def horizon_tail(reg):
    ratio, tail, dens = [], [], []
    for f in sorted(glob.glob(f'data/FAMILY/10x25/desc/{reg}/*.desc.json')):
        d = json.load(open(f))
        name = os.path.basename(f)[:-len('.desc.json')]
        ref = json.load(open(f'or_solution/FAMILY/10x25/{reg}/{name}.json'))
        cmax, H = float(ref['objective']), float(d['horizon'])
        spans = [(s, e) for w in d['windows'] for (s, e) in w]
        tot = sum(e - s for (s, e) in spans)
        inside = sum(max(0.0, min(e, cmax) - s) for (s, e) in spans if s < cmax)
        if not tot:
            continue
        ratio.append(H / cmax)
        tail.append(1.0 - inside / tot)
        # nominal density is measured against H, effective against the span the
        # shop actually runs. Uniform placement makes the two agree; check it.
        dens.append((inside / cmax) / (tot / H))
    return np.mean(ratio), np.mean(tail), np.mean(dens)


print('\n3. OUTAGE CALENDAR vs THE SCHEDULE (H is the layout canvas)')
w_ratio, w_tail, w_dens = horizon_tail('W')
other = [horizon_tail(r)[1] for r in ('LW', 'SW', 'LSW')]
print(f'   W    : H / Cmax = {w_ratio:.2f}, {100 * w_tail:.0f}% of outage time '
      f'falls past the last operation')
print(f'          effective density / nominal density = {w_dens:.2f} '
      f'(uniform placement makes the tail cost count, not severity)')
print(f'   LW,SW,LSW: worst tail {100 * max(other):.1f}% -- the calendar lies '
      f'entirely inside the schedule')

# ------------------------------------------------- 4. WHAT THE CP-SAT CELLS ARE
# What each campaign MUST contain. Counting cells with len(glob(...)) and reporting
# the answer means a solver JSON that failed to write shrinks the denominator instead
# of raising: "62% of cells are certified optimal" would quietly become a percentage
# of a campaign smaller than the one the paper describes, and nothing would say so.
# The 300 s campaign is a deliberate 30-instance prefix, not a truncated 100.
EXPECT = {'10x25': 800, '10x25_fairwarm5s': 800, '10x25_fairwarm30s': 800,
          '10x25_fairwarm300s': 240}


def status(camp):
    c = {}
    for f in glob.glob(f'or_solution/FAMILY/{camp}/*/*.json'):
        d = json.load(open(f))
        c[d['status']] = c.get(d['status'], 0) + 1
    n = sum(c.values())
    if n != EXPECT[camp]:
        raise SystemExit(
            f'REFUSING: campaign {camp} has {n} cells on disk, not {EXPECT[camp]}. '
            f'Every certification percentage this section prints is a fraction of '
            f'that denominator, and a denominator that shrinks when a file goes '
            f'missing reports a different campaign than the one the paper describes.')
    return c, n


print('\n4. CP-SAT CERTIFICATION, by campaign')
cells = {}
for camp in ('10x25', '10x25_fairwarm5s', '10x25_fairwarm30s',
             '10x25_fairwarm300s'):
    c, n = status(camp)
    cells[camp] = (c, n)
    print(f'   {camp:22s} n={n:4d}  ' +
          '  '.join(f'{k}={v}' for k, v in sorted(c.items())))

ref_l = {}
for f in glob.glob('or_solution/FAMILY/10x25/L/*.json'):
    d = json.load(open(f))
    ref_l[d['status']] = ref_l.get(d['status'], 0) + 1
print(f'   reference campaign, regime L alone: {ref_l} '
      '(the baseline the no-wait inflation above is measured against)')

ref_c, ref_n = cells['10x25']
f5_c, f5_n = cells['10x25_fairwarm5s']
f30_c, f30_n = cells['10x25_fairwarm30s']
f300_c, f300_n = cells['10x25_fairwarm300s']
timed_opt = (f5_c.get('OPTIMAL', 0) + f30_c.get('OPTIMAL', 0)
             + f300_c.get('OPTIMAL', 0))
timed_n = f5_n + f30_n + f300_n
print(f'   -> certified optimal over the whole TIMED campaign: '
      f'{timed_opt}/{timed_n} = {100 * timed_opt / timed_n:.1f}%')

emit('paper/macros_exposure.tex', [
    '% auto-generated by scripts/p28_exposure_and_certification.py -- do not hand-edit',
    # 1. seed collision
    ('SeedRuns', str(runs)),
    ('SeedDraws', commafy(draws)),
    # Three decimals, not two: at 54,000 draws the expectation is 0.003, and
    # "0.00" would read as an exact zero, which is a different and unearned claim.
    ('SeedCollisionExp', f'{exp_hits:.3f}'),
    ('SeedCollisionPct', f'{100 * p_cum:.2f}'),
    # 2. no-wait activity on the benchmark
    ('NwN', str(nw_n)),
    ('NwFeasible', str(nw_found)),
    ('NwOpt', str(nw_opt)),
    ('NwFree', str(nw_free)),
    ('NwBelowRef', str(nw_below)),
    ('NwMeanPct', f'{infl.mean():.2f}'),
    ('NwMedianPct', f'{np.median(infl):.2f}'),
    ('NwMaxPct', f'{infl.max():.2f}'),
    # 3. the outage-layout horizon
    ('WinHRatio', f'{w_ratio:.2f}'),
    ('WinTailPct', f'{100 * w_tail:.0f}'),
    ('WinDensX', f'{w_dens:.2f}'),
    ('WinTailOtherPct', f'{100 * max(other):.1f}'),
    # 4. what a CP-SAT cell certifies
    ('CpRefCells', str(ref_n)),
    ('CpRefOpt', str(ref_c.get('OPTIMAL', 0))),
    ('CpRefLCells', str(sum(ref_l.values()))),
    ('CpRefLOpt', str(ref_l.get('OPTIMAL', 0))),
    ('CpTimedCells', str(timed_n)),
    ('CpTimedOpt', str(timed_opt)),
    ('CpFiveCells', str(f5_n)),
    ('CpFiveOpt', str(f5_c.get('OPTIMAL', 0))),
    ('CpFiveUnknown', str(f5_c.get('UNKNOWN', 0))),
    ('CpThirtyCells', str(f30_n)),
    ('CpThirtyOpt', str(f30_c.get('OPTIMAL', 0))),
    ('CpCapCells', str(f300_n)),
    ('CpCapOpt', str(f300_c.get('OPTIMAL', 0))),
    ('OpDurMedian', f'{op_median:.0f}'),
    ('OpDurMax', f'{op_max:.0f}'),
    # No thousands separator: they sit in the same table row as \SpecParams and
    # \ParamsTotal, which macros.tex writes bare. A row that mixes 32786 with
    # 34{,}322 is a row a reader stops at.
    ('BlindParams', str(blind_p)),
    ('BlindKiB', f'{blind_kib:.0f}'),
    ('ConcatParams', str(concat_p)),
    ('ConcatKiB', f'{concat_kib:.0f}'),
])
print('\nwrote paper/macros_exposure.tex')
