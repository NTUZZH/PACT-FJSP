"""Schedule builders and schedule evaluation for the FJSP timing-rule family.

Timing semantics (identical to constraint_family/reference_sim.simulate and
FJSPEnvFamily): operation o of job j on machine m with machine predecessor a
(the operation processed on m immediately before o, or none) occupies m over
the contiguous block [b, b + s + p], where
    s = S[class(a), class(o)]   (S[idle, class(o)] when o is first on m),
    r = release date of j       (first operation of j),
      = end(jprev) + lag(jprev) (any later operation),
    b >= max(r, end(a)),        (attached setup: the job must be present)
and [b, b + s + p] overlaps no outage of m (no preemption). The processing
start is b + s and the end is b + s + p.

Three kernels build or evaluate schedules, all in integer hours:

  decode_append   chromosome (job-repetition sequence, machine assignment)
                  -> schedule; each operation is appended to its machine at the
                  earliest time; equals reference_sim.simulate exactly.
  decode_insert   the same chromosome -> schedule, but each operation is placed
                  into the earliest idle interval of its machine in which it
                  fits: the job is present, the setup from the operation before
                  the interval and the new setup into the operation after it
                  both fit, the block avoids every outage, and the operation
                  after the interval keeps its processing start.
  eval_seq        machine sequences (linked lists) + assignment -> the earliest
                  (semi-active) schedule that respects those sequences,
                  computed in a topological order of the precedence graph;
                  returns -1 when the sequences contain a cycle.

Machine sequences are doubly linked lists: mpred[o], msucc[o] (-1 at the
ends), mhead[m], mtail[m] (-1 for an empty machine).
"""
import numpy as np
from numba import njit


@njit(cache=True)
def push_block(b, dur, ws_m, we_m, k):
    """Earliest t >= b such that [t, t + dur] overlaps no outage of the
    machine. Outages are sorted and disjoint, so one ascending pass is exact
    (same rule as reference_sim.push_past_outages_scalar)."""
    for i in range(k):
        if b < we_m[i] and b + dur > ws_m[i]:
            b = we_m[i]
    return b


@njit(cache=True)
def _overlaps_outage(b, e, ws_m, we_m, k):
    for i in range(k):
        if b < we_m[i] and e > ws_m[i]:
            return True
    return False


@njit(cache=True)
def decode_append(os_, ma, first, pt, jprev, jnext, lag, rel_op, cls, S, idle,
                  nw, ws, we, blk, st, ct, rdy, mpred, msucc, mhead, mtail):
    """Append decoder. Fills blk/st/ct/rdy and the machine linked lists and
    returns the makespan."""
    N = os_.shape[0]
    M = pt.shape[1]
    J = first.shape[0]
    nxt = first.copy()
    for m in range(M):
        mhead[m] = -1
        mtail[m] = -1
    ms = 0
    for t in range(N):
        j = os_[t]
        o = nxt[j]
        nxt[j] += 1
        m = ma[o]
        p = pt[o, m]
        jp = jprev[o]
        if jp < 0:
            r = rel_op[o]
        else:
            r = ct[jp] + lag[jp]
        a = mtail[m]
        if a >= 0:
            f = ct[a]
            s = S[cls[a], cls[o]]
        else:
            f = 0
            s = S[idle, cls[o]]
        b = r if r > f else f
        if nw[m] > 0:
            b = push_block(b, s + p, ws[m], we[m], nw[m])
        blk[o] = b
        st[o] = b + s
        ct[o] = b + s + p
        rdy[o] = r
        mpred[o] = a
        msucc[o] = -1
        if a >= 0:
            msucc[a] = o
        else:
            mhead[m] = o
        mtail[m] = o
        if ct[o] > ms:
            ms = ct[o]
    return ms


@njit(cache=True)
def decode_insert(os_, ma, first, pt, jprev, jnext, lag, rel_op, cls, S, idle,
                  nw, ws, we, blk, st, ct, rdy, mpred, msucc, mhead, mtail):
    """Gap-insertion decoder. Each operation, in chromosome order, goes into
    the first idle interval of its machine where it fits (see module doc);
    operations already placed keep their processing starts, and the one
    after the interval gets the setup from the inserted operation's class.
    Returns the makespan."""
    N = os_.shape[0]
    M = pt.shape[1]
    nxt = first.copy()
    for m in range(M):
        mhead[m] = -1
        mtail[m] = -1
    ms = 0
    for t in range(N):
        j = os_[t]
        o = nxt[j]
        nxt[j] += 1
        m = ma[o]
        p = pt[o, m]
        co = cls[o]
        jp = jprev[o]
        if jp < 0:
            r = rel_op[o]
        else:
            r = ct[jp] + lag[jp]
        rdy[o] = r
        # c = first operation on m whose processing starts after r; earlier
        # operations cannot follow o (o would end after their start)
        c = mtail[m]
        if c >= 0 and st[c] > r:
            while mpred[c] >= 0 and st[mpred[c]] > r:
                c = mpred[c]
        else:
            c = -1
        placed = False
        while c >= 0:
            a = mpred[c]
            if a >= 0:
                f = ct[a]
                s_in = S[cls[a], co]
            else:
                f = 0
                s_in = S[idle, co]
            b = r if r > f else f
            if nw[m] > 0:
                b = push_block(b, s_in + p, ws[m], we[m], nw[m])
            e = b + s_in + p
            s_c = S[co, cls[c]]
            bc = st[c] - s_c
            ok = e <= bc and bc >= rdy[c]
            if ok and nw[m] > 0 and bc < blk[c]:
                ok = not _overlaps_outage(bc, ct[c], ws[m], we[m], nw[m])
            if ok:
                blk[o] = b
                st[o] = b + s_in
                ct[o] = e
                blk[c] = bc
                mpred[o] = a
                msucc[o] = c
                mpred[c] = o
                if a >= 0:
                    msucc[a] = o
                else:
                    mhead[m] = o
                placed = True
                break
            c = msucc[c]
        if not placed:
            a = mtail[m]
            if a >= 0:
                f = ct[a]
                s_in = S[cls[a], co]
            else:
                f = 0
                s_in = S[idle, co]
            b = r if r > f else f
            if nw[m] > 0:
                b = push_block(b, s_in + p, ws[m], we[m], nw[m])
            blk[o] = b
            st[o] = b + s_in
            ct[o] = b + s_in + p
            mpred[o] = a
            msucc[o] = -1
            if a >= 0:
                msucc[a] = o
            else:
                mhead[m] = o
            mtail[m] = o
        if ct[o] > ms:
            ms = ct[o]
    return ms


@njit(cache=True)
def eval_seq(ma, mpred, msucc, pt, jprev, jnext, lag, rel_op, cls, S, idle,
             nw, ws, we, blk, st, ct, indeg, order, tail):
    """Earliest schedule for fixed machine sequences and assignment.

    Kahn's algorithm over the precedence graph (job arcs with lags, machine
    arcs with setups); start times are the earliest ones given the arcs and
    the outages. A backward pass over the same topological order then gives
    tail[o], the longest chain of successor blocks after the end of o (job
    arcs carry their lag; outage pushes are not included).

    Returns (makespan, number of operations with end + tail = makespan),
    the second value counting the operations on longest chains, or (-1, -1)
    when the machine sequences and the job order form a cycle."""
    N = ma.shape[0]
    top = 0
    for o in range(N):
        d = 0
        if jprev[o] >= 0:
            d += 1
        if mpred[o] >= 0:
            d += 1
        indeg[o] = d
    # order[] doubles as the stack of ready operations: entries [0, head)
    # are processed, [head, top) are ready
    for o in range(N):
        if indeg[o] == 0:
            order[top] = o
            top += 1
    head = 0
    ms = 0
    while head < top:
        o = order[head]
        head += 1
        m = ma[o]
        p = pt[o, m]
        jp = jprev[o]
        if jp < 0:
            r = rel_op[o]
        else:
            r = ct[jp] + lag[jp]
        a = mpred[o]
        if a >= 0:
            f = ct[a]
            s = S[cls[a], cls[o]]
        else:
            f = 0
            s = S[idle, cls[o]]
        b = r if r > f else f
        if nw[m] > 0:
            b = push_block(b, s + p, ws[m], we[m], nw[m])
        blk[o] = b
        st[o] = b + s
        e = b + s + p
        ct[o] = e
        if e > ms:
            ms = e
        jn = jnext[o]
        if jn >= 0:
            indeg[jn] -= 1
            if indeg[jn] == 0:
                order[top] = jn
                top += 1
        mn = msucc[o]
        if mn >= 0:
            indeg[mn] -= 1
            if indeg[mn] == 0:
                order[top] = mn
                top += 1
    if head < N:
        return -1, -1
    ncrit = 0
    for t in range(N - 1, -1, -1):
        o = order[t]
        T = 0
        jn = jnext[o]
        if jn >= 0:
            x = lag[o] + (ct[jn] - blk[jn]) + tail[jn]
            if x > T:
                T = x
        mn = msucc[o]
        if mn >= 0:
            x = (ct[mn] - blk[mn]) + tail[mn]
            if x > T:
                T = x
        tail[o] = T
        if ct[o] + T == ms:
            ncrit += 1
    return ms, ncrit


# ---------------------------------------------------------------------------
# moves on machine sequences
# ---------------------------------------------------------------------------

@njit(cache=True)
def _unlink(x, ma, mpred, msucc, mhead, mtail):
    m = ma[x]
    p = mpred[x]
    s = msucc[x]
    if p >= 0:
        msucc[p] = s
    else:
        mhead[m] = s
    if s >= 0:
        mpred[s] = p
    else:
        mtail[m] = p


@njit(cache=True)
def _link_after(x, y, m, ma, mpred, msucc, mhead, mtail):
    """Insert x on machine m right after y (y = -1: at the head)."""
    ma[x] = m
    if y >= 0:
        s = msucc[y]
        msucc[y] = x
    else:
        s = mhead[m]
        mhead[m] = x
    mpred[x] = y
    msucc[x] = s
    if s >= 0:
        mpred[s] = x
    else:
        mtail[m] = x


@njit(cache=True)
def apply_move(x, y, m, ma, mpred, msucc, mhead, mtail):
    """Move operation x to machine m, right after y. Returns (old machine,
    old predecessor) for undo_move."""
    om = ma[x]
    op = mpred[x]
    _unlink(x, ma, mpred, msucc, mhead, mtail)
    _link_after(x, y, m, ma, mpred, msucc, mhead, mtail)
    return om, op


@njit(cache=True)
def undo_move(x, om, op, ma, mpred, msucc, mhead, mtail):
    _unlink(x, ma, mpred, msucc, mhead, mtail)
    _link_after(x, op, om, ma, mpred, msucc, mhead, mtail)


@njit(cache=True)
def eval_moves(moves, n, ma, mpred, msucc, mhead, mtail, pt, jprev, jnext, lag,
               rel_op, cls, S, idle, nw, ws, we, blk, st, ct, indeg, order,
               tail, out_ms, out_nc):
    """Exact evaluation of each move (x, y, m) by apply, eval_seq, undo.
    out_ms[i] = -1 marks a move that creates a cycle; out_nc[i] is the
    number of operations on longest chains. The sequence state is unchanged
    on return; blk/st/ct/tail hold the last neighbour's values."""
    for i in range(n):
        x = moves[i, 0]
        y = moves[i, 1]
        m = moves[i, 2]
        om, op = apply_move(x, y, m, ma, mpred, msucc, mhead, mtail)
        a, b = eval_seq(ma, mpred, msucc, pt, jprev, jnext, lag, rel_op, cls,
                        S, idle, nw, ws, we, blk, st, ct, indeg, order, tail)
        out_ms[i] = a
        out_nc[i] = b
        undo_move(x, om, op, ma, mpred, msucc, mhead, mtail)


@njit(cache=True)
def critical_path(ma, mpred, blk, ct, jprev, lag, rel_op, path, via_m, pick):
    """One critical path of the current schedule, chronological in
    path[0:L]. It starts from an operation that ends at the makespan (the
    pick-th one, modulo their number) and follows tight arcs backwards: the
    machine arc when end(mpred) = block start, else the job arc when
    end(jprev) + lag = block start. When an outage pushed the block past
    both arcs, the path continues through the later of the two (an earlier
    end there can let the block fit before the outage); it stops at a
    release date or at time 0. via_m[i] = 1 when path[i] follows path[i-1]
    through a tight machine arc (the arcs that form critical blocks)."""
    N = ma.shape[0]
    ms = 0
    for o in range(N):
        if ct[o] > ms:
            ms = ct[o]
    cnt = 0
    for o in range(N):
        if ct[o] == ms:
            cnt += 1
    k = pick % cnt
    v = -1
    for o in range(N):
        if ct[o] == ms:
            if k == 0:
                v = o
                break
            k -= 1
    L = 0
    while True:
        path[L] = v
        L += 1
        a = mpred[v]
        jp = jprev[v]
        f = ct[a] if a >= 0 else 0
        r = ct[jp] + lag[jp] if jp >= 0 else rel_op[v]
        via_m[L - 1] = 0
        if a >= 0 and f == blk[v]:
            # a machine arc from the job predecessor is a job arc for the
            # block structure (the two operations cannot be swapped)
            if a != jp:
                via_m[L - 1] = 1
            v = a
        elif jp >= 0 and r == blk[v]:
            v = jp
        elif blk[v] > f and blk[v] > r:
            # pushed by an outage: continue through the later arc
            if a >= 0 and (f >= r or jp < 0):
                if f >= r:
                    v = a
                else:
                    break
            elif jp >= 0:
                v = jp
            else:
                break
        else:
            break
    # In backward order via_m[i] is the arc into the i-th traced operation
    # from the (i+1)-th; reversing both arrays gives the chronological path
    # with via_m[i] = arc from path[i-1] into path[i] (via_m[0] = 0).
    for i in range(L // 2):
        t = path[i]
        path[i] = path[L - 1 - i]
        path[L - 1 - i] = t
        t = via_m[i]
        via_m[i] = via_m[L - 1 - i]
        via_m[L - 1 - i] = t
    return L


@njit(cache=True)
def tails(ma, msucc, jnext, lag, blk, ct, tail):
    """tail[o] = length of the longest chain of successor blocks after the
    end of o (job arcs carry their lag), on the current machine sequences.
    Outage pushes are not included, so it is a lower bound on the time the
    successors of o still need."""
    order = np.argsort(-blk)
    for t in range(order.shape[0]):
        o = order[t]
        T = 0
        jn = jnext[o]
        if jn >= 0:
            x = lag[o] + (ct[jn] - blk[jn]) + tail[jn]
            if x > T:
                T = x
        mn = msucc[o]
        if mn >= 0:
            x = (ct[mn] - blk[mn]) + tail[mn]
            if x > T:
                T = x
        tail[o] = T


@njit(cache=True)
def gen_moves(ma, mpred, msucc, mhead, blk, ct, pt, jprev, jnext, lag, rel_op,
              cls, S, idle, nw, ws, we, cptr, cmach, tail, path, via_m, L,
              use_n7, out, est):
    """Critical-path neighbourhood of the current schedule, written to out
    as rows (x, y, m): move operation x to machine m right after y
    (y = -1: at the head of m).

    Sequence moves inside each critical block (maximal run of path
    operations joined by tight machine arcs): swap of the first two and of
    the last two operations (Nowicki and Smutnicki), and, with use_n7, the
    moves of Zhang et al. (2007): an interior operation to the front or the
    back of the block, the first operation behind an interior one, the last
    operation in front of an interior one. Swaps get est = -1, N7 moves
    est = -2.

    Insertion moves: for every critical operation v and every compatible
    machine m' (its own included), the position on m' with the smallest
    estimate est = block end of v at that position + longest tail through v
    (job successor with its lag, or the operation after v with the new
    setup), computed from the current heads and tails with the setups of
    the new neighbours and the outages of m'. On another machine only
    positions after an operation that starts before v ends and before an
    operation that ends after v starts are considered, which excludes every
    position that would put v after one of its successors or before one of
    its predecessors (Mastrolilli and Gambardella's insertion rule)."""
    n = 0
    i = 0
    while i < L:
        k = i + 1
        while k < L and via_m[k] == 1:
            k += 1
        bl = k - i
        if bl >= 2:
            m = ma[path[i]]
            b1 = path[i]
            bk = path[k - 1]
            out[n, 0] = path[i + 1]
            out[n, 1] = mpred[b1]
            out[n, 2] = m
            est[n] = -1
            n += 1
            if bl >= 3:
                out[n, 0] = bk
                out[n, 1] = mpred[path[k - 2]]
                out[n, 2] = m
                est[n] = -1
                n += 1
            if use_n7 and bl >= 3:
                # (moves equivalent to the two swaps above are skipped)
                for q in range(i + 2, k - 1):
                    out[n, 0] = path[q]
                    out[n, 1] = mpred[b1]
                    out[n, 2] = m
                    est[n] = -2
                    n += 1
                for q in range(i + 1, k - 2):
                    out[n, 0] = path[q]
                    out[n, 1] = bk
                    out[n, 2] = m
                    est[n] = -2
                    n += 1
                for q in range(i + 2, k):
                    out[n, 0] = b1
                    out[n, 1] = path[q]
                    out[n, 2] = m
                    est[n] = -2
                    n += 1
                for q in range(i, k - 2):
                    out[n, 0] = bk
                    out[n, 1] = mpred[path[q]]
                    out[n, 2] = m
                    est[n] = -2
                    n += 1
        i = k
    for q in range(L):
        v = path[q]
        jp = jprev[v]
        if jp < 0:
            r = rel_op[v]
        else:
            r = ct[jp] + lag[jp]
        jn = jnext[v]
        tj = 0
        if jn >= 0:
            tj = lag[v] + (ct[jn] - blk[jn]) + tail[jn]
        cv = cls[v]
        for ci in range(cptr[v], cptr[v + 1]):
            m2 = cmach[ci]
            own = m2 == ma[v]
            p = pt[v, m2]
            best_e = -1
            best_a = -2
            a = -1
            c = mhead[m2]
            while True:
                if c == v:
                    c = msucc[c]
                    continue
                if not (own and a == mpred[v]):
                    ok = True
                    if not own:
                        if a >= 0 and blk[a] >= ct[v]:
                            ok = False
                        if c >= 0 and ct[c] <= blk[v]:
                            ok = False
                    if ok:
                        if a >= 0:
                            f = ct[a]
                            sa = S[cls[a], cv]
                        else:
                            f = 0
                            sa = S[idle, cv]
                        b = r if r > f else f
                        if nw[m2] > 0:
                            b = push_block(b, sa + p, ws[m2], we[m2], nw[m2])
                        e = b + sa + p
                        tc = 0
                        if c >= 0:
                            tc = S[cv, cls[c]] + pt[c, ma[c]] + tail[c]
                        val = e + (tj if tj > tc else tc)
                        if best_a == -2 or val < best_e:
                            best_e = val
                            best_a = a
                    elif not own and a >= 0 and blk[a] >= ct[v]:
                        break
                if c < 0:
                    break
                a = c
                c = msucc[c]
            if best_a != -2:
                out[n, 0] = v
                out[n, 1] = best_a
                out[n, 2] = m2
                est[n] = best_e
                n += 1
    return n


@njit(cache=True)
def seq_from_schedule(ma, blk, order, mpred, msucc, mhead, mtail):
    """Machine linked lists from a schedule: operations taken in `order`
    (ascending block start) are appended to their machines."""
    M = mhead.shape[0]
    for m in range(M):
        mhead[m] = -1
        mtail[m] = -1
    for t in range(order.shape[0]):
        o = order[t]
        m = ma[o]
        a = mtail[m]
        mpred[o] = a
        msucc[o] = -1
        if a >= 0:
            msucc[a] = o
        else:
            mhead[m] = o
        mtail[m] = o
