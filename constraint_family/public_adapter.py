"""Adapter from public FJSP benchmark files to the arrays the environment uses.

The environment (fjsp_env_same_op_nums.FJSPEnvForSameOpNums and its subclass
constraint_family.fjsp_env_family.FJSPEnvFamily) takes one instance as

    job_length  int array [J]      number of operations of each job
    op_pt       int array [N, M]   processing time of operation i on machine k,
                                   0 when machine k cannot process operation i

with operations numbered job by job from 0. Jobs may have different numbers of
operations and every operation may have its own set of eligible machines.

Standard text format (.fjs), as distributed with the instance library of
Mastrolilli and Gambardella (IDSIA, FJSPLIB TextData.zip):

    line 1   J  M  [average number of eligible machines per operation]
    then J job records. A job record is the number of operations followed,
    for each operation, by k (the number of eligible machines) and k pairs
    (machine, processing time). Machines are numbered from 1.

A job record normally occupies one line. The parser reads the file after the
header as one stream of integers, so a record that wraps over several lines
is read correctly, and it fails loudly when the stream does not end exactly
after the last job record (or after the setup block below).

Setup-time variant (the Fattahi-based instances with sequence-dependent
setup times distributed with the Hexaly FJSP-SDST example): after the J job
records follow M blocks of N rows of N integers. Entry [m][i][j] is the setup
time on machine m when operation j directly follows operation i. A value of
SETUP_SENTINEL (1000000) marks a pair that machine m cannot process. The
files give no initial (first-operation) setup. The example model that ships
with these files imposes start(j) >= end(i) + setup[m][i][j], so the setup can
run while job j is still waiting for its previous operation. That is the
detached convention; FJSPEnvFamily implements attached setups only (see
constraint_family/checks.py, C2).

Base instances can also be drawn from the SD2 distribution of the
dual-attention network release (data_utils.SD2_instance_generator with
data_suffix 'mix'); see sd2_instance().

This module depends on numpy only. It does not import params.py, so it can be
imported from tests and scripts without touching sys.argv.
"""
import os
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

SETUP_SENTINEL = 1000000

_INT = re.compile(r'-?\d+')


class UnrepresentableInstance(ValueError):
    """The file is well formed but the environment cannot hold it, for
    example an eligible processing time of 0 (the environment uses 0 as the
    "machine not eligible" code)."""
_NUM = re.compile(r'-?\d+(?:\.\d+)?')


@dataclass
class PublicInstance:
    """One FJSP instance parsed from a public file.

    Attributes
      name                 file name without extension
      path                 path the instance was read from ('' when parsed
                           from text)
      n_jobs               J
      n_machines           M as declared in the header (machines that no
                           operation can use are kept; see used_machines)
      header_flexibility   third header number when present, else None
      job_length           int [J]
      op_pt                int [N, M], 0 = machine not eligible
      setup                int [M, N, N] or None; SETUP_SENTINEL marks pairs
                           the machine cannot process
    """
    name: str
    path: str
    n_jobs: int
    n_machines: int
    header_flexibility: Optional[float]
    job_length: np.ndarray
    op_pt: np.ndarray
    setup: Optional[np.ndarray] = None
    notes: list = field(default_factory=list)

    # ------------------------------------------------------------------
    @property
    def n_ops(self):
        return int(self.op_pt.shape[0])

    @property
    def op_job(self):
        """[N] index of the job that owns each operation."""
        return np.repeat(np.arange(self.n_jobs), self.job_length)

    @property
    def job_first_op(self):
        return np.concatenate([[0], np.cumsum(self.job_length)[:-1]]).astype(int)

    @property
    def eligible_counts(self):
        """[N] number of eligible machines of each operation."""
        return (self.op_pt > 0).sum(axis=1)

    @property
    def flexibility(self):
        """Mean number of eligible machines per operation."""
        return float(self.eligible_counts.mean())

    @property
    def used_machines(self):
        """Sorted indices of machines that at least one operation can use."""
        return np.flatnonzero((self.op_pt > 0).any(axis=0))

    @property
    def has_incompatible_pair(self):
        """True when some (operation, machine) pair is not eligible.

        FJSPEnvFamily normalizes processing times by (pt - min) / (max - min)
        over the batch and asserts that the minimum is 0, i.e. that the batch
        contains at least one ineligible pair. An instance where every
        operation is eligible on every machine (total flexibility) fails that
        assertion when it is evaluated alone."""
        return bool((self.op_pt == 0).any())

    @property
    def mean_pt(self):
        """Mean processing time over eligible (operation, machine) pairs."""
        return float(self.op_pt[self.op_pt > 0].mean())

    def env_arrays(self):
        """(job_length, op_pt) as fresh int arrays in the environment's format."""
        return self.job_length.astype(int).copy(), self.op_pt.astype(int).copy()

    def without_unused_machines(self):
        """Copy with every machine that no operation can use removed.

        Machine indices are renumbered in their original order; the mapping
        new -> old is recorded in notes. The makespan of any schedule is
        unchanged because the removed machines can process nothing."""
        keep = self.used_machines
        if len(keep) == self.n_machines:
            return self
        setup = None if self.setup is None else self.setup[keep]
        inst = PublicInstance(self.name, self.path, self.n_jobs, len(keep),
                              self.header_flexibility, self.job_length.copy(),
                              self.op_pt[:, keep].copy(), setup,
                              list(self.notes))
        inst.notes.append(f'removed unused machines; new->old machine map '
                          f'{keep.tolist()}')
        return inst

    def summary(self):
        return {'name': self.name, 'jobs': self.n_jobs,
                'machines_declared': self.n_machines,
                'machines_used': int(len(self.used_machines)),
                'operations': self.n_ops,
                'ops_per_job': (int(self.job_length.min()),
                                int(self.job_length.max())),
                'flexibility': round(self.flexibility, 4),
                'max_eligible': int(self.eligible_counts.max()),
                'pt_range': (int(self.op_pt[self.op_pt > 0].min()),
                             int(self.op_pt.max())),
                'has_setup_block': self.setup is not None}


# ----------------------------------------------------------------------
# parsing
# ----------------------------------------------------------------------

def parse_fjs_text(text, name='', path=''):
    """Parse the text of a .fjs file (standard or setup-time variant)."""
    lines = text.splitlines()
    i0 = 0
    while i0 < len(lines) and not lines[i0].strip():
        i0 += 1
    if i0 == len(lines):
        raise ValueError(f'{name or path}: empty file')
    head = _NUM.findall(lines[i0])
    if len(head) < 2:
        raise ValueError(f'{name or path}: header needs J and M, got {lines[i0]!r}')
    J, M = int(float(head[0])), int(float(head[1]))
    flex = float(head[2]) if len(head) >= 3 else None
    if J <= 0 or M <= 0:
        raise ValueError(f'{name or path}: header J={J}, M={M}')
    toks = [int(t) for t in _INT.findall('\n'.join(lines[i0 + 1:]))]

    pos = 0
    job_length = np.zeros(J, dtype=int)
    rows = []
    for j in range(J):
        if pos >= len(toks):
            raise ValueError(f'{name or path}: file ends inside job {j}')
        n_op = toks[pos]
        pos += 1
        if n_op <= 0:
            raise ValueError(f'{name or path}: job {j} has {n_op} operations')
        job_length[j] = n_op
        for o in range(n_op):
            k = toks[pos]
            pos += 1
            if k <= 0:
                raise ValueError(f'{name or path}: job {j} operation {o} has '
                                 f'{k} eligible machines')
            row = np.zeros(M, dtype=int)
            for _ in range(k):
                mch, p = toks[pos], toks[pos + 1]
                pos += 2
                if not 1 <= mch <= M:
                    raise ValueError(f'{name or path}: job {j} operation {o}: '
                                     f'machine {mch} outside 1..{M}')
                if p < 0:
                    raise ValueError(f'{name or path}: job {j} operation {o}: '
                                     f'negative processing time {p}')
                if p == 0:
                    # 0 is the environment's "not eligible" code, so a zero
                    # processing time cannot be represented
                    raise UnrepresentableInstance(
                        f'{name or path}: job {j} operation {o}: processing '
                        f'time 0 on machine {mch}')
                if row[mch - 1]:
                    raise ValueError(f'{name or path}: job {j} operation {o}: '
                                     f'machine {mch} listed twice')
                row[mch - 1] = p
            rows.append(row)
    op_pt = np.array(rows, dtype=int)
    N = op_pt.shape[0]

    rest = toks[pos:]
    setup = None
    if rest:
        if len(rest) != M * N * N:
            raise ValueError(f'{name or path}: {len(rest)} integers after the '
                             f'job records; a setup block needs M*N*N = '
                             f'{M * N * N}')
        setup = np.array(rest, dtype=np.int64).reshape(M, N, N)
    return PublicInstance(name=name, path=path, n_jobs=J, n_machines=M,
                          header_flexibility=flex, job_length=job_length,
                          op_pt=op_pt, setup=setup)


def read_fjs(path):
    """Parse one .fjs file."""
    with open(path, encoding='ascii', errors='strict') as f:
        text = f.read()
    name = os.path.splitext(os.path.basename(path))[0]
    return parse_fjs_text(text, name=name, path=path)


def natural_key(s):
    """Sort key that orders 'la2' before 'la10' and 'orb2' before 'orb10'."""
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r'(\d+)', os.path.basename(s))]


def list_instances(set_dir, ext='.fjs'):
    """Instance files of a set directory in natural order."""
    files = [os.path.join(set_dir, f) for f in os.listdir(set_dir)
             if f.endswith(ext)]
    return sorted(files, key=natural_key)


def load_set(set_dir, skip_unrepresentable=False):
    """All instances of a set directory, in natural order.

    With skip_unrepresentable=True, files that raise UnrepresentableInstance
    are left out and listed in the second return value; otherwise the error
    propagates and only the list of instances is returned."""
    if not skip_unrepresentable:
        return [read_fjs(p) for p in list_instances(set_dir)]
    out, skipped = [], []
    for p in list_instances(set_dir):
        try:
            out.append(read_fjs(p))
        except UnrepresentableInstance as e:
            skipped.append((os.path.basename(p), str(e)))
    return out, skipped


def load_base_instance(stem):
    """(job_length, op_pt, meta) of a base instance named by its path stem.

    A factory instance has a <stem>.meta.json side-car and is read with
    ppvc_instance_generator.load_instance, unchanged. A public instance has
    only <stem>.fjs; its meta carries op_type and mch_type None (no type
    embedding is possible), routing_class None and an all-zero time_lag (the
    lags of the public family live in the descriptor). Callers that pass the
    type lists to the environment must pass None, not [None]."""
    if os.path.exists(stem + '.meta.json'):
        from ppvc_instance_generator import load_instance
        return load_instance(stem)
    inst = read_fjs(stem + '.fjs')
    jl, pt = inst.env_arrays()
    meta = {'op_type': None, 'mch_type': None, 'routing_class': None,
            'time_lag': np.zeros(pt.shape[0], dtype=int), 'base_format': 'fjs',
            'name': inst.name}
    return jl, pt, meta


def to_fjs_text(job_length, op_pt, flexibility=None):
    """Standard .fjs text of an instance (no setup block)."""
    job_length = np.asarray(job_length, dtype=int)
    op_pt = np.asarray(op_pt, dtype=int)
    if flexibility is None:
        flexibility = float((op_pt > 0).sum(axis=1).mean())
    lines = [f'{len(job_length)}\t{op_pt.shape[1]}\t{flexibility:.2f}']
    i = 0
    for L in job_length:
        parts = [str(int(L))]
        for _ in range(L):
            elig = np.flatnonzero(op_pt[i])
            parts.append(str(len(elig)))
            for k in elig:
                parts += [str(k + 1), str(int(op_pt[i, k]))]
            i += 1
        lines.append(' '.join(parts))
    return '\n'.join(lines) + '\n'


# ----------------------------------------------------------------------
# SD2 base instances
# ----------------------------------------------------------------------

def sd2_instance(n_j, n_m, rng, low=1, high=99, op_per_job=None,
                 eligible_range=None):
    """One base instance from the SD2 distribution, drawn with `rng`.

    The distribution is the one of data_utils.SD2_instance_generator with
    data_suffix 'mix' (the setting of the published SD2 test sets
    '10x5+mix' and '20x10+mix'): every job has op_per_job operations (default
    n_m), every operation draws its number of eligible machines uniformly from
    eligible_range (default 1..n_m) and the eligible machines uniformly
    without replacement, and every eligible processing time is drawn
    independently and uniformly from low..high. The random stream differs from
    the original generator, which draws from numpy's global state; this
    function takes a numpy Generator so that training draws never touch the
    global state.

    :return: (job_length [J], op_pt [N, M]) int arrays
    """
    op_per_job = n_m if op_per_job is None else int(op_per_job)
    lo_k, hi_k = (1, n_m) if eligible_range is None else eligible_range
    if not (1 <= lo_k <= hi_k <= n_m):
        raise ValueError(f'eligible_range {(lo_k, hi_k)} with n_m={n_m}')
    n_op = n_j * op_per_job
    job_length = np.full(n_j, op_per_job, dtype=int)
    k = rng.integers(lo_k, hi_k + 1, size=n_op)
    op_pt = rng.integers(low, high + 1, size=(n_op, n_m))
    for i in range(n_op):
        if k[i] < n_m:
            off = rng.choice(n_m, n_m - k[i], replace=False)
            op_pt[i, off] = 0
    return job_length, op_pt.astype(int)


# ----------------------------------------------------------------------
# structure of a setup-time block
# ----------------------------------------------------------------------

def setup_dont_care(inst):
    """[M, N, N] bool: setup entries that no schedule can use.

    An entry (m, i, j) is never used when i == j, when machine m cannot
    process i or j, or when j precedes i in the same job (i is processed
    after j, so i cannot be followed directly by j on any machine)."""
    N, M = inst.n_ops, inst.n_machines
    elig = inst.op_pt.T > 0                                   # [M, N]
    dc = ~(elig[:, :, None] & elig[:, None, :])
    dc |= np.eye(N, dtype=bool)[None]
    job = inst.op_job
    order = np.arange(N)
    same_job = job[:, None] == job[None, :]
    backward = same_job & (order[None, :] < order[:, None])   # j before i
    dc |= backward[None]
    return dc


def setup_structure(inst, source_convention='detached'):
    """How the setup block of an instance depends on machines, jobs and
    operations, and whether the project's descriptor can hold it exactly.

    The descriptor holds one [C+1, C] matrix between C job classes (C = 4)
    plus a cold-start row, shared by all machines, with attached semantics.
    An exact representation therefore needs (a) setups that do not depend on
    the machine, (b) a partition of the operations into at most C classes
    such that the setup depends only on the pair of classes, and (c) the
    attached convention. The class count reported here is the number of
    distinct (row, column) patterns over the used entries, an upper bound on
    the smallest exact partition.
    """
    if inst.setup is None:
        return {'has_setup': False}
    S = inst.setup.astype(np.int64)
    dc = setup_dont_care(inst)
    used = ~dc
    sentinel = S >= SETUP_SENTINEL
    elig = inst.op_pt.T > 0
    pair_ok = elig[:, :, None] & elig[:, None, :]
    out = {'has_setup': True}
    out['sentinel_matches_eligibility'] = bool(np.array_equal(
        sentinel, ~pair_ok))
    vals = S[used]
    out['used_entries'] = int(used.sum())
    out['setup_min'] = int(vals.min()) if vals.size else None
    out['setup_max'] = int(vals.max()) if vals.size else None
    out['setup_mean'] = float(vals.mean()) if vals.size else None
    out['setup_mean_over_mean_pt'] = (float(vals.mean()) / inst.mean_pt
                                      if vals.size else None)

    # (a) machine independence on entries used on two machines
    M = inst.n_machines
    diffs = 0
    for a in range(M):
        for b in range(a + 1, M):
            both = used[a] & used[b]
            diffs += int((S[a][both] != S[b][both]).sum())
    out['machine_independent'] = diffs == 0
    out['machine_mismatch_entries'] = diffs

    # (b) dependence only on the pair of jobs, per machine
    job = inst.op_job
    job_only = True
    for m in range(M):
        seen = {}
        idx = np.argwhere(used[m])
        for i, j in idx:
            key = (job[i], job[j])
            v = S[m, i, j]
            if seen.setdefault(key, v) != v:
                job_only = False
                break
        if not job_only:
            break
    out['job_pair_only'] = job_only

    # distinct (row, column) patterns per machine, don't-care entries masked
    Smask = np.where(used, S, -1)
    n_cls = []
    for m in range(M):
        ops = np.flatnonzero(elig[m])
        pats = {tuple(Smask[m, i, ops]) + ('|',) + tuple(Smask[m, ops, i])
                for i in ops}
        n_cls.append(len(pats))
    out['classes_needed_upper_bound_per_machine'] = (min(n_cls), max(n_cls))

    # triangle inequality s(i,k) <= s(i,j) + s(j,k) on used entries
    tri = 0
    for m in range(M):
        ops = np.flatnonzero(elig[m])
        sub = np.where(used[m][np.ix_(ops, ops)],
                       S[m][np.ix_(ops, ops)], np.int64(SETUP_SENTINEL))
        via = sub[:, :, None] + sub[None, :, :]                # i, j, k
        direct = sub[:, None, :]
        ok = used[m][np.ix_(ops, ops)]
        tri += int(((direct > via) & ok[:, None, :]
                    & ok[:, :, None] & ok[None, :, :]).sum())
    out['triangle_violations'] = tri
    out['initial_setup_given'] = False
    out['convention_of_source_model'] = source_convention
    reasons = []
    if source_convention != 'attached':
        reasons.append(f'the source model uses {source_convention} setups; '
                       f'the environment implements attached setups only')
    if not out['machine_independent']:
        reasons.append('setups depend on the machine; the descriptor holds '
                       'one matrix for all machines')
    if not job_only:
        reasons.append('setups depend on the operation, not only on the job')
    if max(n_cls) > 4:
        reasons.append(f'up to {max(n_cls)} distinct operation patterns on a '
                       f'machine; the descriptor has 4 classes')
    out['reasons_not_exact'] = reasons
    out['exact_in_descriptor'] = not reasons
    # Without an initial setup in the file, an exact descriptor would set the
    # idle (cold-start) row to 0; check C1 admits zero entries.
    out['idle_row_needed'] = 'zeros'
    return out
