"""Dense integer arrays of one family instance.

The decoder and the search work on integer hours only (the descriptor format
stores every duration, lag, setup, outage and release date in integer hours),
so every array here is int64 and every comparison is exact.

Operation indexing follows the instance file: operations are numbered job by
job, operation k of job j has the global index first[j] + k.
"""
import numpy as np

from constraint_family.descriptor import IDLE_CLASS, N_CLASSES


def _as_int(a, name):
    a = np.asarray(a, dtype=float)
    if a.size and not np.all(a == np.round(a)):
        raise ValueError(f'{name} must hold integer hours')
    return np.ascontiguousarray(np.round(a).astype(np.int64))


class Instance:
    """Arrays of one (job_length, op_pt, descriptor) triple.

    Attributes (N operations, M machines, J jobs, C routing classes)
      pt        [N, M] processing time, 0 = incompatible pair
      job_of    [N] job of each operation
      jprev     [N] job predecessor operation, -1 for a first operation
      jnext     [N] job successor operation, -1 for a last operation
      first     [J] index of each job's first operation
      lag       [N] minimum time lag after the operation (0 when LAG is off)
      rel_op    [N] release date of the job on its first operation, else 0
      cls       [N] routing class of the operation's job
      S         [C+1, C] setup matrix, row C = cold start (all 0 when SETUP is off)
      nw        [M] number of outages per machine (0 when WINDOW is off)
      ws, we    [M, K] outage starts and ends, sorted, padded
      cptr      [N+1] offsets into cmach (compatible machines of each operation)
      cmach     compatible machines, operation by operation
    """

    def __init__(self, job_length, op_pt, desc):
        if getattr(desc, 'setup_mult', None) is not None:
            # the per-machine setup multiplier of the sensitivity study; the
            # decoders and the move evaluation here charge the full setup,
            # so such an instance would be solved under the wrong rule
            raise ValueError(
                'descriptor carries setup_mult (per-machine setup multiplier); '
                'the memetic baseline does not implement it and refuses the '
                'instance instead of ignoring the field')
        self.desc = desc
        self.job_length = np.asarray(job_length, dtype=np.int64)
        self.pt = _as_int(op_pt, 'op_pt')
        N, M = self.pt.shape
        J = len(self.job_length)
        self.N, self.M, self.J = N, M, J
        first = np.concatenate([[0], np.cumsum(self.job_length)[:-1]]).astype(np.int64)
        self.first = first
        self.job_of = np.repeat(np.arange(J, dtype=np.int64), self.job_length)
        self.jprev = np.arange(N, dtype=np.int64) - 1
        self.jprev[first] = -1
        last = first + self.job_length - 1
        self.jnext = np.arange(N, dtype=np.int64) + 1
        self.jnext[last] = -1

        self.has_lag = bool(desc.has_lag)
        self.has_setup = bool(desc.has_setup)
        self.has_window = bool(desc.has_window and desc.windows is not None)
        rel = getattr(desc, 'release', None)
        self.has_release = rel is not None

        self.lag = (_as_int(desc.lag, 'lag') if self.has_lag
                    else np.zeros(N, dtype=np.int64))
        self.cls = _as_int(desc.op_class, 'op_class')
        if self.has_setup:
            self.S = _as_int(desc.setup, 'setup')
        else:
            self.S = np.zeros((N_CLASSES + 1, N_CLASSES), dtype=np.int64)
        self.idle = IDLE_CLASS

        self.rel_op = np.zeros(N, dtype=np.int64)
        self.release = np.zeros(J, dtype=np.int64)
        if self.has_release:
            self.release = _as_int(rel, 'release')
            if self.release.shape != (J,):
                raise ValueError(f'release must have length J = {J}')
            self.rel_op[first] = self.release

        K = 1
        if self.has_window:
            K = max(1, max(len(w) for w in desc.windows))
        self.nw = np.zeros(M, dtype=np.int64)
        self.ws = np.zeros((M, K), dtype=np.int64)
        self.we = np.zeros((M, K), dtype=np.int64)
        if self.has_window:
            for m, spans in enumerate(desc.windows):
                spans = sorted((int(s), int(e)) for (s, e) in spans)
                self.nw[m] = len(spans)
                for k, (s, e) in enumerate(spans):
                    self.ws[m, k] = s
                    self.we[m, k] = e

        cm = [np.nonzero(self.pt[i] > 0)[0] for i in range(N)]
        if any(len(c) == 0 for c in cm):
            raise ValueError('an operation has no compatible machine')
        self.compat = cm
        self.cptr = np.concatenate([[0], np.cumsum([len(c) for c in cm])]).astype(np.int64)
        self.cmach = np.concatenate(cm).astype(np.int64)
        # machine of minimum processing time per operation (first on ties)
        self.min_pt_mach = np.array([c[np.argmin(self.pt[i, c])] for i, c in enumerate(cm)],
                                    dtype=np.int64)
        self.base_os = np.repeat(np.arange(J, dtype=np.int64), self.job_length)

    def kernel_args(self):
        """Static arrays in the order the numba kernels take them."""
        return (self.pt, self.jprev, self.jnext, self.lag, self.rel_op, self.cls,
                self.S, np.int64(self.idle), self.nw, self.ws, self.we)
