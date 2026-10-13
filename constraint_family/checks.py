"""Configuration checks on a constraint descriptor, run BEFORE the descriptor
configures FJSPEnvFamily (paper table "configuration checks").

  C1 schema_and_values      code       shapes, types, ranges, bit/parameter consistency
  C2 setup_convention       code       only the attached convention is admitted (Remark 1)
  C3 family_membership      code       every active rule type is in the registry
  C4 magnitude_range        code       WARN when magnitudes leave the benchmark's ranges
  C5 applicability          code       blocking shops (finite downstream buffers) are out
  C6 schedule_validation    code       independent re-check of a produced schedule
  C7 source_reconciliation  procedure  descriptor vs. plant source records (manual)

Each check returns a CheckResult (id, name, status, messages, details) with
status in {PASS, WARN, FAIL, MANUAL}. `run_all` returns the ordered report
C1..C7. Every check is a pure function of its inputs: no randomness, no
clock, no environment state, so statuses are deterministic.

The training ranges used by C4 are DERIVED from the generator code: the
keyword defaults of descriptor.draw_setup_matrix / draw_windows (read with
inspect.signature, so they are the defaults the published benchmark was
drawn with) and the lag ranges of ppvc_instance_generator.ROUTES. No range is
retyped here.
"""
import hashlib
import inspect
import json
from dataclasses import dataclass, field

import numpy as np

from constraint_family.descriptor import (IDLE_CLASS, N_CLASSES,
                                          ConstraintDescriptor,
                                          draw_setup_matrix, draw_windows)
from constraint_family import reference_sim
import schedule_validator
from ppvc_instance_generator import ROUTES

PASS, WARN, FAIL, MANUAL = 'PASS', 'WARN', 'FAIL', 'MANUAL'

# Positional meaning of ConstraintDescriptor.active.
ACTIVATION_ORDER = ('LAG', 'SETUP', 'WINDOW')


@dataclass
class CheckResult:
    id: str
    name: str
    status: str
    messages: list = field(default_factory=list)
    details: dict = field(default_factory=dict)

    @property
    def passed(self):
        """True unless the check failed (WARN and MANUAL do not block)."""
        return self.status != FAIL

    def as_row(self):
        return (self.id, self.status, '; '.join(self.messages))


def _result(cid, name, fails, warns=(), infos=(), details=None, manual=False):
    if manual:
        status = MANUAL
    elif fails:
        status = FAIL
    elif warns:
        status = WARN
    else:
        status = PASS
    return CheckResult(cid, name, status,
                       list(fails) + list(warns) + list(infos),
                       dict(details or {}))


def _is_int_valued(a):
    a = np.asarray(a, dtype=float)
    return bool(np.all(np.isfinite(a)) and np.all(a == np.round(a)))


# ---------------------------------------------------------------------------
# C1 schema_and_values
# ---------------------------------------------------------------------------

def check_schema_and_values(desc, op_pt=None, job_length=None):
    """C1. Shapes, types and value ranges of every descriptor field, and the
    consistency between activation bits and parameters.

    N (number of operations) and M (number of machines) are taken from op_pt
    when it is given; otherwise N = len(op_class) and the per-machine count of
    the window calendar is not checked. `details['setup_all_positive']`
    reports whether every setup entry is > 0, which the strictly-tightening
    envelope claim needs (the admissibility proof needs only >= 0).

    Release dates (optional field `release`, one per job): 1-D, finite,
    non-negative integer hours. Their length must equal J = len(job_length)
    when job_length is given; without it the length is only bounded by N,
    and C6 checks the release dates against the produced schedule.
    """
    fails, infos, det = [], [], {}
    C = N_CLASSES

    # activation bits
    act = getattr(desc, 'active', None)
    if not (isinstance(act, (tuple, list)) and len(act) == 3
            and all(isinstance(b, (bool, np.bool_)) for b in act)):
        fails.append(f'active must be a 3-tuple of bools, got {act!r}')
        return _result('C1', 'schema_and_values', fails, details=det)
    has_lag, has_setup, has_window = (bool(b) for b in act)

    N = M = None
    if op_pt is not None:
        op_pt = np.asarray(op_pt)
        if op_pt.ndim != 2:
            fails.append(f'op_pt must be 2-D [N, M], got shape {op_pt.shape}')
        else:
            N, M = op_pt.shape

    # op_class
    oc = np.asarray(desc.op_class)
    if oc.ndim != 1:
        fails.append(f'op_class must be 1-D, got shape {oc.shape}')
    else:
        if N is None:
            N = oc.shape[0]
        if oc.shape[0] != N:
            fails.append(f'op_class length {oc.shape[0]} != N = {N}')
        if oc.size and (not _is_int_valued(oc) or oc.min() < 0
                        or oc.max() > C - 1):
            fails.append(f'op_class entries must be integers in 0..{C - 1}, '
                         f'got range [{oc.min()}, {oc.max()}]')

    # lag
    lag = np.asarray(desc.lag, dtype=float)
    if lag.ndim != 1 or (N is not None and lag.shape[0] != N):
        fails.append(f'lag must be 1-D of length N = {N}, got shape {lag.shape}')
    elif lag.size:
        if not np.all(np.isfinite(lag)) or lag.min() < 0:
            fails.append(f'lag entries must be finite and >= 0, min = {lag.min()}')
        elif not _is_int_valued(lag):
            fails.append('lag entries must be integer hours '
                         '(to_json() truncates non-integers)')
    if not has_lag and lag.size and np.any(lag != 0):
        fails.append('LAG bit is off but lag has nonzero entries')

    # setup
    S = desc.setup
    if S is not None:
        S = np.asarray(S, dtype=float)
        if S.shape != (C + 1, C):
            fails.append(f'setup must have shape [C+1, C] = [{C + 1}, {C}], '
                         f'got {list(S.shape)}')
        elif not np.all(np.isfinite(S)) or S.min() < 0:
            fails.append(f'setup entries must be finite and >= 0, '
                         f'min = {np.nanmin(S)}')
        elif not _is_int_valued(S):
            fails.append('setup entries must be integer hours '
                         '(to_json() truncates non-integers)')
        else:
            det['setup_all_positive'] = bool(S.min() > 0)
            infos.append('setup entries all > 0 (envelope strictly tightening)'
                         if det['setup_all_positive'] else
                         'setup has zero entries (envelope admissible, '
                         'not strictly tightening)')
    if has_setup and S is None:
        fails.append('SETUP bit is on but setup is None')
    if not has_setup and S is not None and np.asarray(S).size \
            and np.any(np.asarray(S, dtype=float) != 0):
        fails.append('SETUP bit is off but setup has nonzero entries')

    # windows
    W = desc.windows
    if W is not None:
        if not isinstance(W, (list, tuple)):
            fails.append(f'windows must be a list of per-machine lists, '
                         f'got {type(W).__name__}')
        else:
            if M is not None and len(W) != M:
                fails.append(f'windows has {len(W)} machine lists, M = {M}')
            for m, spans in enumerate(W):
                if not isinstance(spans, (list, tuple)):
                    fails.append(f'windows[{m}] is not a list')
                    continue
                prev_s = prev_e = None
                for k, iv in enumerate(spans):
                    if not (isinstance(iv, (list, tuple)) and len(iv) == 2):
                        fails.append(f'windows[{m}][{k}] is not a (start, end) pair')
                        continue
                    s, e = iv
                    num = (int, np.integer, float, np.floating)
                    if not (isinstance(s, num) and isinstance(e, num)
                            and _is_int_valued([s, e])):
                        fails.append(f'windows[{m}][{k}] = {iv!r} is not a pair '
                                     f'of integer hours')
                        continue
                    if s < 0:
                        fails.append(f'windows[{m}][{k}] = [{s}, {e}) starts before 0')
                    if not s < e:
                        fails.append(f'windows[{m}][{k}] = [{s}, {e}) needs start < end')
                    if prev_s is not None:
                        if s < prev_s:
                            fails.append(f'windows[{m}] not sorted at index {k}')
                        elif s < prev_e:
                            fails.append(f'windows[{m}][{k - 1}] = [{prev_s}, {prev_e}) '
                                         f'and [{s}, {e}) overlap')
                    prev_s, prev_e = s, e
    if has_window:
        if W is None:
            fails.append('WINDOW bit is on but windows is None')
        h = desc.horizon
        if h is None or not _is_int_valued([h]) or h <= 0:
            fails.append(f'WINDOW bit is on but horizon = {h!r} is not a '
                         f'positive integer')
    elif W is not None and isinstance(W, (list, tuple)) \
            and any(len(spans) for spans in W if isinstance(spans, (list, tuple))):
        fails.append('WINDOW bit is off but windows lists outages')

    # release dates (candidate fourth rule; active iff the field is present)
    rel = getattr(desc, 'release', None)
    if rel is not None:
        rel = np.asarray(rel, dtype=float)
        if rel.ndim != 1 or rel.size == 0:
            fails.append(f'release must be a non-empty 1-D list of per-job '
                         f'dates, got shape {rel.shape}')
        elif N is not None and rel.shape[0] > N:
            fails.append(f'release has {rel.shape[0]} entries, more than the '
                         f'N = {N} operations')
        elif job_length is not None and rel.shape[0] != len(job_length):
            fails.append(f'release has {rel.shape[0]} entries, '
                         f'J = {len(job_length)} jobs')
        elif not np.all(np.isfinite(rel)) or rel.min() < 0:
            fails.append(f'release entries must be finite and >= 0, '
                         f'min = {np.nanmin(rel)}')
        elif not _is_int_valued(rel):
            fails.append('release entries must be integer hours '
                         '(to_json() truncates non-integers)')
        else:
            infos.append(f'release dates on {rel.shape[0]} jobs, max '
                         f'{int(rel.max())} h')

    return _result('C1', 'schema_and_values', fails, infos=infos, details=det)


# ---------------------------------------------------------------------------
# C2 setup_convention
# ---------------------------------------------------------------------------

ADMITTED_SETUP_CONVENTIONS = ('attached',)
KNOWN_SETUP_CONVENTIONS = ('attached', 'detached', 'hybrid')


def check_setup_convention(desc):
    """C2. The released environment implements attached setups only.

    A detached or hybrid changeover can run during the job's wait, so the
    descriptor-only minimum-setup envelope over-estimates the completion and
    the dense reward is no longer admissible (paper Remark 1). Old records
    without the field are read as 'attached'.
    """
    conv = getattr(desc, 'setup_convention',
                   ConstraintDescriptor.DEFAULT_SETUP_CONVENTION)
    det = {'setup_convention': conv}
    if conv in ADMITTED_SETUP_CONVENTIONS:
        return _result('C2', 'setup_convention', [],
                       infos=[f"declared convention '{conv}'"], details=det)
    if conv in KNOWN_SETUP_CONVENTIONS:
        return _result('C2', 'setup_convention', [
            f"declared setup convention '{conv}' is not implemented: the "
            f"minimum-setup envelope is inadmissible unless setups are attached "
            f"(Remark 1, rem:detached); re-express the changeover as attached "
            f"(internal) setup time or do not use this environment"],
            details=det)
    return _result('C2', 'setup_convention', [
        f"unknown setup convention {conv!r}; expected one of "
        f"{list(KNOWN_SETUP_CONVENTIONS)} and only 'attached' is admitted"],
        details=det)


# ---------------------------------------------------------------------------
# C3 family_membership
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RuleSpec:
    hook: str        # 'job' (post-completion job delay) or 'machine' (start map)
    envelope: str    # 'trivial' (delta = 0) or 'min-setup' (Eq. envelope)
    proof_ref: str   # where the Definition 1 obligations are discharged


VALID_HOOKS = ('job', 'machine')
VALID_ENVELOPES = ('trivial', 'min-setup')

RULE_REGISTRY = {
    'LAG': RuleSpec('job', 'trivial', 'Def. 1(i); trivial envelope (family.tex)'),
    'SETUP': RuleSpec('machine', 'min-setup',
                      'Def. 1(ii); Lemma 1 (attached convention), Eq. envelope'),
    'WINDOW': RuleSpec('machine', 'trivial',
                       'Def. 1(ii), push map inflationary and monotone; '
                       'trivial envelope'),
    # Candidate fourth rule: not an activation bit; active iff the
    # descriptor carries a `release` field (see active_rule_types).
    'RELEASE': RuleSpec('machine', 'trivial',
                        'Def. 1(ii), start map b -> max(b, rho_i) on the '
                        'first operation of job i, inflationary and monotone; '
                        'trivial envelope'),
}


def register_rule(name, hook, envelope, proof_ref, registry=None):
    """Admit a new rule type to the family registry used by C3.

    Registering asserts that the rule meets the proof obligations of
    Definition 1: its effect factors through a schedule-independent job delay
    (hook 'job') or an inflationary, monotone machine-side start map computable
    from the descriptor and the partial schedule (hook 'machine'), and it has
    an admissible delay lower envelope ('trivial' is always admissible;
    'min-setup' needs the attached-convention argument of Lemma 1).
    `proof_ref` names where those obligations are discharged. Registration
    does NOT implement the rule in FJSPEnvFamily; the environment's transition
    and its _bound_min_term must be extended separately.
    """
    reg = RULE_REGISTRY if registry is None else registry
    if not isinstance(name, str) or not name or name != name.upper():
        raise ValueError(f'rule name must be a non-empty upper-case string, got {name!r}')
    if name in reg:
        raise ValueError(f'rule {name!r} is already registered')
    if hook not in VALID_HOOKS:
        raise ValueError(f'hook must be one of {VALID_HOOKS}, got {hook!r}')
    if envelope not in VALID_ENVELOPES:
        raise ValueError(f'envelope must be one of {VALID_ENVELOPES}, got {envelope!r}')
    if not isinstance(proof_ref, str) or not proof_ref.strip():
        raise ValueError('proof_ref must name where the Definition 1 '
                         'obligations are discharged')
    reg[name] = RuleSpec(hook, envelope, proof_ref)
    return reg[name]


def active_rule_types(desc, declared_rules=None):
    """Active rule types: the named activation bits, RELEASE when the
    descriptor carries release dates, plus any rule types the
    caller declares (argument or a `declared_rules` attribute on the
    descriptor), in a deterministic order without duplicates."""
    names = [n for n, b in zip(ACTIVATION_ORDER, desc.active) if b]
    if getattr(desc, 'release', None) is not None:
        names.append('RELEASE')
    extra = list(getattr(desc, 'declared_rules', ()) or ()) + list(declared_rules or ())
    for n in extra:
        if n not in names:
            names.append(n)
    return names


def check_family_membership(desc, declared_rules=None, registry=None):
    """C3. Every active rule type must be registered with a valid hook and
    envelope."""
    reg = RULE_REGISTRY if registry is None else registry
    active = active_rule_types(desc, declared_rules)
    fails, infos = [], []
    for n in active:
        spec = reg.get(n)
        if spec is None:
            fails.append(f'rule type {n!r} is not in the family registry '
                         f'{sorted(reg)}; admit it with register_rule() only '
                         f'after discharging the Definition 1 obligations')
        elif spec.hook not in VALID_HOOKS or spec.envelope not in VALID_ENVELOPES:
            fails.append(f'rule type {n!r} has invalid registry entry {spec}')
        else:
            infos.append(f'{n}: hook={spec.hook}, envelope={spec.envelope}')
    return _result('C3', 'family_membership', fails, infos=infos,
                   details={'active_rules': active})


# ---------------------------------------------------------------------------
# C4 magnitude_range (training ranges derived from the generator code)
# ---------------------------------------------------------------------------

def _kw_default(fn, name):
    return inspect.signature(fn).parameters[name].default


# Fractions of the instance mean compatible processing time p_bar.
SETUP_SAME_RANGE = tuple(_kw_default(draw_setup_matrix, 'same'))
SETUP_CROSS_RANGE = tuple(_kw_default(draw_setup_matrix, 'cross'))
SETUP_SYMMETRIC = bool(_kw_default(draw_setup_matrix, 'symmetric'))
# draw_setup_matrix draws the idle (cold-start) row from the same-class range.
SETUP_COLD_RANGE = SETUP_SAME_RANGE
# Fraction of the horizon H, and hours.
WINDOW_DENSITY_RANGE = tuple(_kw_default(draw_windows, 'density_range'))
WINDOW_DUR_RANGE = tuple(_kw_default(draw_windows, 'dur_range'))
# Nonzero post-operation lag ranges (hours) of the PPVC routes.
LAG_RANGES = tuple(sorted({lag for route in ROUTES.values()
                           for (*_, lag) in route if lag[1] > 0}))
# Tolerance for rounding to integer hours (the generator rounds every draw).
ROUND_TOL_H = 0.5


def training_ranges():
    """The ranges C4 compares against, as derived from the generator."""
    return {'setup_same_frac_pbar': SETUP_SAME_RANGE,
            'setup_cross_frac_pbar': SETUP_CROSS_RANGE,
            'setup_cold_frac_pbar': SETUP_COLD_RANGE,
            'setup_symmetric': SETUP_SYMMETRIC,
            'window_density_frac_H': WINDOW_DENSITY_RANGE,
            'window_duration_h': WINDOW_DUR_RANGE,
            'lag_nonzero_h': LAG_RANGES,
            'rounding_tolerance_h': ROUND_TOL_H}


def _entries_out(vals, lo_h, hi_h):
    """Entries outside [lo_h, hi_h] hours, allowing the rounding tolerance
    and the generator's max(1, .) clamp."""
    lo = min(lo_h - ROUND_TOL_H, 1.0)
    hi = max(hi_h + ROUND_TOL_H, 1.0)
    return [v for v in vals if v < lo or v > hi]


def check_magnitude_range(desc, op_pt):
    """C4. WARN (never FAIL) when a magnitude lies outside the range the
    published benchmark was drawn from, i.e. outside what the policy trained on.

    Quantities: setup entries (same-class, cross-class, cold-start) and the
    mean cross-class setup as fractions of p_bar = mean compatible processing
    time of op_pt; setup asymmetry max|S[a,b]-S[b,a]| (the generator is
    symmetric); per-machine outage density (total outage hours / H) and
    outage durations; nonzero lag magnitudes. Each entry may deviate by the
    0.5 h rounding tolerance. The density upper bound adds one maximum outage
    duration because draw_windows keeps adding outages until the target is
    reached, so it overshoots by at most one outage.
    """
    warns, infos, det = [], [], {}
    C = N_CLASSES
    if desc.has_setup and desc.setup is not None:
        if op_pt is None:
            warns.append('SETUP: op_pt not given, setup magnitudes not checked')
        else:
            op_pt = np.asarray(op_pt, dtype=float)
            pbar = float(op_pt[op_pt > 0].mean())
            S = np.asarray(desc.setup, dtype=float)
            diag = [S[a, a] for a in range(C)]
            cross = [S[a, b] for a in range(C) for b in range(C) if a != b]
            cold = list(S[IDLE_CLASS])
            cross_mean = float(np.mean(cross)) / pbar
            asym = float(np.abs(S[:C] - S[:C].T).max())
            det.update(pbar=pbar, cross_mean_frac=cross_mean, asymmetry_h=asym)
            tol_frac = ROUND_TOL_H / pbar
            lo, hi = SETUP_CROSS_RANGE
            if not (lo - tol_frac <= cross_mean <= hi + tol_frac):
                warns.append(f'SETUP: mean cross-class setup = {cross_mean:.2f} '
                             f'p_bar, outside training range [{lo:.2f}, {hi:.2f}] '
                             f'p_bar (p_bar = {pbar:.2f} h)')
            for label, vals, (lo, hi) in (('same-class', diag, SETUP_SAME_RANGE),
                                          ('cross-class', cross, SETUP_CROSS_RANGE),
                                          ('cold-start', cold, SETUP_COLD_RANGE)):
                out = _entries_out(vals, lo * pbar, hi * pbar)
                if out:
                    warns.append(f'SETUP: {len(out)} of {len(vals)} {label} entries '
                                 f'outside [{lo * pbar:.1f}, {hi * pbar:.1f}] h '
                                 f'(= [{lo:.2f}, {hi:.2f}] p_bar); extreme = '
                                 f'{max(out, key=lambda v: abs(v - (lo + hi) / 2 * pbar)):.0f} h')
            if SETUP_SYMMETRIC and asym > ROUND_TOL_H:
                warns.append(f'SETUP: asymmetry max|S[a,b]-S[b,a]| = {asym:.0f} h; '
                             f'the training generator draws symmetric matrices')
    if desc.has_window and desc.windows is not None:
        H = desc.horizon
        if not H:
            warns.append('WINDOW: horizon missing, outage density not checked')
        else:
            H = float(H)
            dlo, dhi = WINDOW_DENSITY_RANGE
            ulo, uhi = WINDOW_DUR_RANGE
            dens = [sum(e - s for (s, e) in w) / H for w in desc.windows]
            det['window_density_range'] = (min(dens), max(dens)) if dens else None
            lo_ok = dlo - ROUND_TOL_H / H
            hi_ok = dhi + (uhi + ROUND_TOL_H) / H
            bad = [(m, d) for m, d in enumerate(dens) if not lo_ok <= d <= hi_ok]
            if bad:
                m, d = max(bad, key=lambda x: abs(x[1] - (dlo + dhi) / 2))
                warns.append(f'WINDOW: {len(bad)} of {len(dens)} machines have '
                             f'outage density outside [{dlo:.2f}, {dhi:.2f}] H '
                             f'(+ one outage of overshoot); worst machine {m} = '
                             f'{d:.3f} H (H = {H:.0f} h)')
            durs = [e - s for w in desc.windows for (s, e) in w]
            bad_d = [u for u in durs if not ulo - ROUND_TOL_H <= u <= uhi + ROUND_TOL_H]
            if bad_d:
                warns.append(f'WINDOW: {len(bad_d)} of {len(durs)} outages have '
                             f'duration outside [{ulo}, {uhi}] h; extreme = '
                             f'{max(bad_d, key=lambda u: abs(u - (ulo + uhi) / 2))} h')
    if desc.has_lag:
        lag = np.asarray(desc.lag, dtype=float)
        nz = lag[lag > 0]
        det['lag_nonzero_range'] = (float(nz.min()), float(nz.max())) if nz.size else None
        mid = (min(lo for lo, _ in LAG_RANGES) + max(hi for _, hi in LAG_RANGES)) / 2
        bad = [v for v in nz if not any(lo - ROUND_TOL_H <= v <= hi + ROUND_TOL_H
                                        for (lo, hi) in LAG_RANGES)]
        if bad:
            warns.append(f'LAG: {len(bad)} of {nz.size} nonzero lags outside the '
                         f'route ranges {list(LAG_RANGES)} h; extreme = '
                         f'{max(bad, key=lambda v: abs(v - mid)):.0f} h')
    if not warns:
        infos.append('all active magnitudes inside the benchmark training ranges')
    return _result('C4', 'magnitude_range', [], warns=warns, infos=infos,
                   details=det)


# ---------------------------------------------------------------------------
# C5 applicability
# ---------------------------------------------------------------------------

def check_applicability(finite_downstream_buffers):
    """C5. A shop with finite downstream buffers blocks: a machine is released
    only when the job's next operation starts, which violates the fixed
    release f_k <- C_ij of the transition and can deadlock the constructive
    policy (paper, limits of the family: blocking). Such shops are outside the
    family. The flag must be declared explicitly as True or False."""
    det = {'finite_downstream_buffers': finite_downstream_buffers}
    if not isinstance(finite_downstream_buffers, (bool, np.bool_)):
        return _result('C5', 'applicability', [
            f'finite_downstream_buffers must be declared True or False, got '
            f'{finite_downstream_buffers!r}'], details=det)
    if finite_downstream_buffers:
        return _result('C5', 'applicability', [
            'finite downstream buffers make the shop blocking; blocking shops are '
            'outside the monotone-delay family (the machine is not released at '
            'completion and the constructive policy can deadlock)'], details=det)
    return _result('C5', 'applicability', [],
                   infos=['unlimited downstream buffers declared (non-blocking)'],
                   details=det)


# ---------------------------------------------------------------------------
# C6 schedule_validation
# ---------------------------------------------------------------------------

def check_schedule_validation(desc, job_length, op_pt, op_mch, op_ct, op_st=None,
                              tol=1e-6):
    """C6. Re-check a produced schedule with the independent validators.

    schedule_validator.validate_schedule (no repository imports) checks
    compatibility, non-negative starts, job precedence with the active lags,
    machine no-overlap of processing intervals, the job-chain lower bound
    and, when the descriptor carries release dates, that no job's first
    operation starts before its release date.
    reference_sim.validate_schedule adds the descriptor-aware conditions:
    processing durations, attached setup blocks that must not overlap the
    previous operation on the machine, and blocks that must not overlap an
    outage. op_st defaults to op_ct - p (the processing start; the attached
    setup precedes it).
    """
    op_pt = np.asarray(op_pt, dtype=float)
    op_mch = np.asarray(op_mch, dtype=int)
    op_ct = np.asarray(op_ct, dtype=float)
    if op_st is None:
        op_st = schedule_validator.reconstruct_starts(op_pt, op_mch, op_ct)
    op_st = np.asarray(op_st, dtype=float)
    lag = (np.asarray(desc.lag, dtype=float) if desc.has_lag
           else np.zeros(op_pt.shape[0]))
    r1 = schedule_validator.validate_schedule(
        job_length, op_pt, lag, op_mch, op_ct, tol=tol,
        release=getattr(desc, 'release', None))
    v2 = reference_sim.validate_schedule(job_length, op_pt, desc, op_st, op_ct,
                                         op_mch, tol=tol)
    violations = [f'[schedule_validator] {v}' for v in r1['violations']] + \
                 [f'[reference_sim] {v}' for v in v2]
    det = {'makespan': r1['makespan'], 'lower_bound': r1['lower_bound'],
           'violations': violations}
    if violations:
        return _result('C6', 'schedule_validation',
                       [f'{len(violations)} violation(s); first: {violations[0]}'],
                       details=det)
    return _result('C6', 'schedule_validation', [],
                   infos=[f'feasible; makespan = {r1["makespan"]:.0f} h'],
                   details=det)


# ---------------------------------------------------------------------------
# C7 source_reconciliation (procedure)
# ---------------------------------------------------------------------------

def descriptor_fingerprint(desc):
    """SHA-256 of the canonical to_json() record (sorted keys)."""
    blob = json.dumps(desc.to_json(), sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(blob.encode()).hexdigest()


def check_source_reconciliation(desc, descriptor_version=None, source_records=None):
    """C7. Organisational procedure, not decidable in code.

    Procedure: before release, a planner compares each descriptor parameter
    with the plant record it was taken from (lags with the curing / ponding /
    drying specification, setup matrix with the changeover log, outage
    calendar with the maintenance plan, routing classes with the module
    schedule) and signs off. This stub records what was reconciled: the
    descriptor version, its content fingerprint and the source record
    identifiers passed in. It always returns MANUAL.
    """
    rec = {'descriptor_version': descriptor_version,
           'descriptor_sha256': descriptor_fingerprint(desc),
           'source_records': sorted(str(s) for s in (source_records or ()))}
    msg = (f'manual reconciliation required: descriptor version '
           f'{descriptor_version!r}, sha256 {rec["descriptor_sha256"][:12]}, '
           f'{len(rec["source_records"])} source record(s) recorded')
    return _result('C7', 'source_reconciliation', [], infos=[msg], details=rec,
                   manual=True)


# ---------------------------------------------------------------------------
# run_all
# ---------------------------------------------------------------------------

def run_all(descriptor, op_pt, finite_downstream_buffers=False, *,
            declared_rules=None, schedule=None, job_length=None,
            descriptor_version=None, source_records=None):
    """Ordered report C1..C7 as a list of CheckResult.

    C4 is evaluated only when C1 passes (it reads the fields C1 validates);
    otherwise it reports WARN 'not evaluated'. C6 runs only when a schedule
    is given as a dict with keys op_mch, op_ct and optionally op_st (plus
    job_length); without one it is MANUAL (deferred to the produced schedule).
    """
    c1 = check_schema_and_values(descriptor, op_pt, job_length)
    report = [c1,
              check_setup_convention(descriptor),
              check_family_membership(descriptor, declared_rules)]
    if c1.status == FAIL:
        report.append(_result('C4', 'magnitude_range', [], warns=[
            'not evaluated because C1 failed']))
    else:
        report.append(check_magnitude_range(descriptor, op_pt))
    report.append(check_applicability(finite_downstream_buffers))
    if schedule is None:
        report.append(_result('C6', 'schedule_validation', [], infos=[
            'deferred: run on each produced schedule '
            '(check_schedule_validation)'], manual=True))
    elif job_length is None or c1.status == FAIL:
        report.append(_result('C6', 'schedule_validation', [
            'schedule given but job_length missing or C1 failed']))
    else:
        report.append(check_schedule_validation(
            descriptor, job_length, op_pt, schedule['op_mch'],
            schedule['op_ct'], schedule.get('op_st')))
    report.append(check_source_reconciliation(descriptor, descriptor_version,
                                              source_records))
    return report


def format_report(report):
    return '\n'.join(f'{r.id:<3} {r.name:<22} {r.status:<6} {"; ".join(r.messages)}'
                     for r in report)
