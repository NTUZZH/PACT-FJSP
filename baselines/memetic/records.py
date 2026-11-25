"""Reading the anytime record of a run.

Each entry of rec['anytime'] (eval_memetic_family.py) is

  [search_s, wall_s, cpu_s, makespan]

  search_s   seconds of the budget clock (rec['budget_clock']: wall or CPU)
             since the start of the search, i.e. since the timed window opened;
  wall_s     wall seconds since the start of the process that solved the
             instance (a child forked for this run);
  cpu_s      CPU seconds of that process since its start;
  makespan   the incumbent from that moment on.

The span between process start and search start holds instance loading,
reading or rolling out the seed schedules, their validation and the numba
warm-up; the record stores it as search_start_wall_s and
search_start_cpu_s, and the warm-up alone as jit_time_s. Records of the
first version of the runner have two columns [search_s, makespan].
"""

CLOCKS = ('wall', 'cpu')
ORIGINS = ('process', 'search')


def incumbent_at(rec, t, clock='wall', origin='process'):
    """Best makespan known at time t (seconds, inclusive), or None when the
    run had no incumbent yet.

    clock:  'wall' or 'cpu'
    origin: 'process' (from the start of the solving process) or 'search'
            (from the start of the timed window)"""
    if clock not in CLOCKS or origin not in ORIGINS:
        raise ValueError(f'clock in {CLOCKS}, origin in {ORIGINS}')
    rows = rec['anytime']
    if rows and len(rows[0]) == 2:
        # first-version record: budget clock from the search start only
        if origin != 'search' or clock != rec.get('budget_clock', 'wall'):
            raise ValueError('this record holds only the budget clock from the '
                             'search start')
        times = [r[0] for r in rows]
    else:
        col = 1 if clock == 'wall' else 2
        times = [r[col] for r in rows]
        if origin == 'search':
            off = rec['search_start_wall_s' if clock == 'wall' else 'search_start_cpu_s']
            times = [x - off for x in times]
    best = None
    for x, r in zip(times, rows):
        if x <= t + 1e-9:
            best = r[-1]
        else:
            break
    return best


def incumbents_at(rec, times, clock='wall', origin='process'):
    """incumbent_at for several times; returns a dict {t: makespan or None}."""
    return {t: incumbent_at(rec, t, clock, origin) for t in times}
