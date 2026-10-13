#!/bin/bash
# Re-run the fair-warm CP-SAT campaign with the REPAIRED warm-start hints as the
# solver's true AddHint, for the 5 s and 30 s tiers at 10x25 (100 instances x 8
# regimes each). Everything except the hint SOURCE is identical to the original
# campaign (scripts/run_cpsat_fair.sh, scripts/p12_cpsat_fairwarm.py): same
# solver, 8 search workers, 2 parallel jobs, min(hint, search) crediting, same
# validation. Only fw.WARM_CACHE is repointed at the repaired cache built by
# scripts/p41_repaired_warm_cache.py.
#
# Outputs go to NEW directories and never touch the originals:
#   or_solution/FAMILY/10x25_fairwarmR5s/{regime}/{instance}.json
#   or_solution/FAMILY/10x25_fairwarmR30s/{regime}/{instance}.json
#
# PREPARE-ONLY. This script refuses to run unless RUN=1 is set AND no CP-SAT /
# crossover campaign is on the cores, because it needs cores 0-15 to itself: a
# wall-clock budget measured under contention is not a measurement.
#
#   Launch later with:  RUN=1 bash scripts/run_fairwarm_repaired.sh
#
set -u
cd "$(dirname "$0")/.."
PY=$HOME/miniconda3/envs/fjsp/bin/python
REP=or_solution/FAMILY/warmstart_cache_repaired
CORES=0-15
export OMP_NUM_THREADS=1        # OR-Tools manages its own search workers
export MKL_NUM_THREADS=1

# 1. Explicit opt-in. Without it, this is documentation, not a launch.
if [ "${RUN:-0}" != "1" ]; then
  echo "REFUSING: set RUN=1 to launch (this starts CP-SAT solves on cores $CORES)."
  echo "  RUN=1 bash scripts/run_fairwarm_repaired.sh"
  exit 1
fi

# 2. Refuse if a CP-SAT / crossover campaign is still holding the cores. A second
#    solve campaign sharing cores 0-15 would double every wall-clock and bias the
#    5 s / 30 s cells this script exists to measure.
if pgrep -af 'p12_cpsat_fairwarm|p44_cpsat_scale|cpsat.*crossover|crossover' \
      | grep -v -- 'run_fairwarm_repaired' >/dev/null; then
  echo "REFUSING: a CP-SAT / crossover campaign is still running:"
  pgrep -af 'p12_cpsat_fairwarm|p44_cpsat_scale|crossover' | grep -v 'run_fairwarm_repaired'
  echo "Wait for it to finish; these solves need cores $CORES to themselves."
  exit 1
fi

# 3. Refuse unless the repaired 10x25 cache is complete (800 hints). A missing
#    hint would make p12._warm_start silently REBUILD it from the ORIGINAL rules,
#    which is exactly the thing this rerun replaces.
n=$(ls $REP/{N,L,S,W,LS,LW,SW,LSW}/*.npz 2>/dev/null | wc -l)
if [ "$n" -ne 800 ]; then
  echo "REFUSING: repaired 10x25 cache has $n/800 hints. Build it first:"
  echo "  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 taskset -c 20-23 \\"
  echo "    $PY scripts/p41_repaired_warm_cache.py --sizes 10x25"
  exit 1
fi

mkdir -p train_log
echo "=== repaired-hint CP-SAT (5s, 30s) starting $(date '+%m-%d %H:%M') ==="
taskset -c $CORES $PY - <<'PYEOF' 2>&1 | tee -a train_log/cpsat_fairwarm_repaired.log
import glob, multiprocessing as mp, os, sys, time
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, 'scripts'); sys.path.insert(0, '.')
sys.argv = [sys.argv[0]]                       # clean argv for params import
import p12_cpsat_fairwarm as fw

REP = 'or_solution/FAMILY/warmstart_cache_repaired'
DATA, BASE = 'data/FAMILY/10x25', 'data/PPVC/10x25+ppvc-mixed'
REGIMES = ['S', 'LS', 'SW', 'LSW', 'N', 'L', 'W', 'LW']   # order as run_cpsat_fair
WORKERS, JOBS = 8, 2                            # identical to the original campaign
fw.WARM_CACHE = REP                             # the ONLY change: repaired hints

for BUDGET in (5, 30):
    out_tag = f'_fairwarmR{BUDGET}s'
    tasks = []
    for r in REGIMES:
        out_dir = f'or_solution/FAMILY/10x25{out_tag}/{r}'
        os.makedirs(out_dir, exist_ok=True)
        for dp in sorted(glob.glob(f'{DATA}/desc/{r}/*.desc.json')):
            name = os.path.basename(dp).replace('.desc.json', '')
            cp = f'{REP}/{r}/{name}.npz'
            if not os.path.exists(cp):
                raise SystemExit(f'REFUSING: no repaired hint at {cp}')
            out_path = os.path.join(out_dir, f'{name}.json')
            if os.path.exists(out_path):
                continue                        # idempotent
            tasks.append((os.path.join(BASE, name), dp, out_path, BUDGET, WORKERS))
    print(f'[{BUDGET}s] {len(tasks)} repaired-hint solves '
          f'({WORKERS} workers x {JOBS} jobs)', flush=True)
    ctx = mp.get_context('fork')                # workers inherit fw.WARM_CACHE=REP
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=JOBS, mp_context=ctx) as ex:
        for k, out in enumerate(ex.map(fw.solve_one, tasks), 1):
            inst, reg, st, obj, rule, wms = out
            print(f'[{BUDGET}s {k}/{len(tasks)}] {reg:4s} {inst} {st:9s} '
                  f'obj={obj} warm={rule}({wms}) '
                  f'({(time.time()-t0)/60:.1f}m)', flush=True)
    print(f'[{BUDGET}s] complete -> or_solution/FAMILY/10x25{out_tag}', flush=True)
print('repaired-hint rerun complete', flush=True)
PYEOF
echo "=== repaired-hint CP-SAT done $(date '+%m-%d %H:%M') ==="
