#!/bin/bash
# Randomized dispatching rules within the time budget, on one slot
# (scripts/p135_randomized_rules.py run). Method, cells, records, validity, the
# plan and the analysis: the docstring of that script.
#
# PRE-RUN RECORD (the launch note of every start repeats it with the processes
# on the slot's cores).
#
# 1. PURPOSE. The matched dispatching-rule baseline of the one-minute row of the
#    time-budget comparison: the four rules of the best-of-four start schedule,
#    randomized (restricted candidate list, alpha drawn from 0.05, 0.1, 0.2,
#    0.3) and sampled on the eight single-thread workers of the slot for 60 s,
#    as the policy's sampled schedules are (scripts/p97_policy_anytime.py). The
#    numbers land in results/randomized_rules.json and
#    manuscript/macros_randomized_rules.tex (p135 infer): the randomized rules
#    against the policy's greedy and sampled schedules, the best deterministic
#    rule, the eight memetic runs and CP-SAT, at 10, 30 and 60 s.
# 2. EXPECTED RESULT. At or below the best deterministic rule on every cell (the
#    first schedule of workers 0 to 3 is one of the four deterministic rules, so
#    the best of four is held after one rollout). If the policy's sampled
#    schedules stay shorter, the policy's lead at equal effort holds against the
#    matched rule baseline; if the randomized rules are shorter, the decision
#    table reports it. Either outcome is reported.
# 3. CONTAMINATION. The runner writes each record atomically and never
#    overwrites one; an invalid record moves to <combination>/invalid/ and its
#    cell runs again at the next start (held after two). Instances 000-002
#    selected the search configurations and are not in the plan. Before the
#    start the slot's cores are checked as the timed campaign's prelaunch checks
#    them (a process above 5% of a core on the slot refuses the start; the
#    launch note lists every process above 5% with the core it last ran on),
#    and again before every cell (the core sample of p97); every record stores
#    each worker's core and CPU share (valid at 0.9 or more), the mask and the
#    load average. The results folder test_results/FAMILY/rndrules_p135 is read
#    by no existing analysis.
# 4. DATA ACCURACY. data/FAMILY/<size>/desc/<combination>/instance_<i> and the
#    base instance each names, loaded by p97's rule-arm loader; the four
#    deterministic makespans of every cell must equal the rule cache
#    or_solution/FAMILY/warmstart_cache_xmask_<size>, or the record is invalid;
#    every schedule is validated with the CP-SAT runner's validators.
#
# The launcher and its plan call run on CONTROL_CORE; only the runner (the cells
# and their workers) runs on the slot.
#
# Usage (repository root):
#   SLOT=8-15 SHARE=1/2 scripts/run_randomized_rules.sh
#   SLOT=16-23 SHARE=2/2 scripts/run_randomized_rules.sh
#       checks, then runs the cells of its share in the foreground, appending to
#       train_log/randomized_rules_slot<cores>.log; exit 1 when a start is
#       refused or when a cell of the share has no valid record at the end.
#       One launcher per slot: the lock train_log/randomized_rules_slot<cores>.lock
#       is held by the launcher and its runner for their lifetime (flock; released
#       when they end, also when they are killed), and a second start for the same
#       slot is refused while it is held or while the pid file names a live runner.
#       Every refusal is written to the log as well.
#   DRY=1 SLOT=8-15 SHARE=1/2 scripts/run_randomized_rules.sh
#       the plan: cells, states, expected wall-clock, commands; starts nothing
#       and writes nothing
#   Detached: setsid -f nohup env SLOT=8-15 SHARE=1/2 \
#       scripts/run_randomized_rules.sh > /dev/null 2>&1 < /dev/null
# Stopping:
#   touch train_log/RNDRULE_HOLD   no new cell; the running cell ends and is classified
#   kill -TERM $(cat train_log/randomized_rules_slot8-15.pid)
#       stops at once: the running cell is killed and writes no record, so it
#       runs again at the next start
# Options (environment):
#   SLOT (required, a core list such as 8-15), SHARE (1/1; i/n splits the cells
#   between n slots), BUDGET (60), WORKERS (8), SEED (1), SIZES, COMBOS,
#   INSTANCES (the runner's defaults when unset), CONTROL_CORE (3), PY,
#   RESULT_ROOT (smoke runs only: --development --root RESULT_ROOT, logs under
#   RESULT_ROOT/train_log; such records are not measurements)
set -u
SELF=$(readlink -f "$0")
cd "$(dirname "$SELF")/.."
PY=${PY:-$HOME/miniconda3/envs/fjsp/bin/python}
CONTROL_CORE=${CONTROL_CORE:-3}
if ! taskset -pc "$CONTROL_CORE" $$ > /dev/null; then
  echo "REFUSING: could not pin the launcher to core $CONTROL_CORE"; exit 1
fi
SLOT=${SLOT:-}
if [ -z "$SLOT" ]; then
  echo "REFUSING: SLOT is required (SLOT=8-15 or SLOT=16-23)"; exit 1
fi
DRY=${DRY:-0}
SHARE=${SHARE:-1/1}
BUDGET=${BUDGET:-60}
WORKERS=${WORKERS:-8}
SEED=${SEED:-1}
SIZES=${SIZES:-}
COMBOS=${COMBOS:-}
INSTANCES=${INSTANCES:-}
RESULT_ROOT=${RESULT_ROOT:-}
if [ -n "$RESULT_ROOT" ]; then LOGDIR=$RESULT_ROOT/train_log; else LOGDIR=train_log; fi
TAG=${SLOT//,/_}
LOG=$LOGDIR/randomized_rules_slot$TAG.log
PIDFILE=$LOGDIR/randomized_rules_slot$TAG.pid
HOLD=$LOGDIR/RNDRULE_HOLD

ARGS=(--share "$SHARE" --budget "$BUDGET" --workers "$WORKERS" --seed "$SEED")
[ -n "$SIZES" ] && ARGS+=(--sizes $SIZES)
[ -n "$COMBOS" ] && ARGS+=(--combos $COMBOS)
[ -n "$INSTANCES" ] && ARGS+=(--instances "$INSTANCES")
[ -n "$RESULT_ROOT" ] && ARGS+=(--development --root "$RESULT_ROOT")
ENVS=(env PACT_ENV=exact PACT_SEEDS=301-305 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
      OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=)
SCRIPT=(-B scripts/p135_randomized_rules.py)
PLAN=("${ENVS[@]}" "$PY" "${SCRIPT[@]}" plan "${ARGS[@]}" --slot "$SLOT")
RUNNER=(taskset -c "$SLOT" "${ENVS[@]}" "$PY" "${SCRIPT[@]}" run "${ARGS[@]}")
say() { echo "[rndrules $(date '+%m-%d %H:%M:%S')] $*"; }

if [ "$DRY" = 1 ]; then
  "${PLAN[@]}"; rc=$?
  echo "command of this slot: ${RUNNER[*]}"
  echo "DRY=1: nothing started, nothing written"
  exit $rc
fi

mkdir -p "$LOGDIR"
LOCK=$LOGDIR/randomized_rules_slot$TAG.lock
exec 9>> "$LOCK"
if ! flock -n 9; then
  say "REFUSING: a launcher of slot $SLOT runs already (lock $LOCK held)" | tee -a "$LOG"; exit 1
fi
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2> /dev/null; then
  say "REFUSING: a runner of slot $SLOT runs already (pid $(cat "$PIDFILE"))" | tee -a "$LOG"; exit 1
fi
if [ -f "$HOLD" ]; then
  say "held by $HOLD; remove it to start" | tee -a "$LOG"; exit 0
fi
out=$("${PLAN[@]}" --strict 2>&1); rc=$?
echo "$out" >> "$LOG"
if [ $rc -ne 0 ]; then
  echo "$out" | grep '^REFUSAL'
  say "REFUSING to start (plan in $LOG)" | tee -a "$LOG"; exit 1
fi
say "launcher pid $$ on core $CONTROL_CORE; slot $SLOT; share $SHARE; budget $BUDGET s; log $LOG" | tee -a "$LOG"
"${RUNNER[@]}" >> "$LOG" 2>&1 &
child=$!
echo "$child" > "$PIDFILE"
on_signal() {
  trap - TERM INT HUP
  kill -TERM "$child" 2> /dev/null
  wait "$child"
  rm -f "$PIDFILE"
  say "stopped by a signal: the running cell wrote no record and runs again at the next start" | tee -a "$LOG"
  exit 143
}
trap on_signal TERM INT HUP
wait "$child"; rc=$?
rm -f "$PIDFILE"
say "runner exit $rc" | tee -a "$LOG"
grep -E '^(share [0-9]+/[0-9]+: [0-9]+ of|REFUS|STOP|held|without a valid record)' "$LOG" | tail -n 8
exit $rc
