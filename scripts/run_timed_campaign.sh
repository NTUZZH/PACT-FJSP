#!/bin/bash
# Timed evidence campaign (tracker row C4): the memetic algorithm and the
# CP-SAT comparator, each started from the dispatching-rule schedule and from
# the policy's schedule, on the evidence instances 003-029 of the 30-instance
# production sets (and 000-029 of the training size 10x25 with SIZES=10x25),
# on the two exclusive E-core slots. Cells, batches, the
# memory pairing of the two slots, validity, launch notes and the reader of
# the trajectories: scripts/timed_campaign_plan.py.
#
# PRE-RUN RECORD (each batch's launch note states it for its arm).
#
# 1. PURPOSE. One long run per cell with the incumbent's timestamps, so that
#    the incumbent at 1, 10, 60, 300, 600 and 1800 s is read from one
#    trajectory; the numbers land in the decision table by time budget and in
#    the paired comparisons policy-seeded against rule-seeded (C4).
# 2. IDEAL RESULT. Every cell gets a valid record. A batch whose records are all
#    invalid stops the campaign: something was on the slot's cores.
# 3. CONTAMINATION. The runners write each record atomically (temporary file,
#    then rename) and skip an existing record; nothing is ever overwritten. An
#    invalid record moves to <combination>/invalid/ and its cell runs again; a
#    record that does not match the plan (budget, seed, hint, workers) stops the
#    start. Instances 000-002 of the production sets selected the
#    configurations and are left out by default; no selection was made at 10x25.
#    Before every batch the slot's cores are checked for any process above 5% of
#    a core (refusal stops the campaign), and a launch note goes to
#    train_log/timed_campaign_launch_notes.txt. The two slots never run batches
#    whose memory needs sum above MEM_BUDGET_GB (other processes of the host
#    hold up to about 20 GB of the 62).
# 4. DATA ACCURACY. data/FAMILY/<size>/desc/<combination>/instance_<i>, rule
#    schedules from or_solution/FAMILY/warmstart_cache_xmask_<size> (best of
#    the four masked rules), policy schedules from
#    or_solution/FAMILY/pactwarm_cache_xmask_s<seed>_<size>; both runners
#    validate the start schedule before the timed window and the result before
#    the record. The policy arms are refused while a schedule file is missing.
#
# The launcher, its planner calls and its waiting all run on CONTROL_CORE; only
# the timed commands run on the slots. A memetic batch is one single-thread
# process per core of its slot (eight on a full slot); a CP-SAT batch is one
# 8-worker solve and runs only on a slot with eight cores. A slot takes the
# first arm of ARMS it can run that has cells left, so a narrow slot B runs
# the memetic arms while slot A runs CP-SAT.
#
# Usage:
#   DRY=1 ARMS="memetic_rule cpsat_rule" scripts/run_timed_campaign.sh
#       the plan: cells, commands, done counts, slot-hours, simulated pairing
#       and wall clock; starts nothing and writes nothing
#   ARMS="memetic_rule cpsat_rule" CPSAT_ARM=<pilot choice> scripts/run_timed_campaign.sh
#       checks, then detaches (log train_log/timed_campaign.log, pid file
#       train_log/timed_campaign.pid)
# Stopping:
#   touch train_log/TIMED_HOLD   no new batch; running batches end and are classified
#   kill -TERM $(cat train_log/timed_campaign.pid)
#       stops at once: the running cells are killed with the launcher's process
#       group; a killed run writes no record (only a stray *.tmp file at most),
#       so its cell runs again at the next start
# Options (environment):
#   ARMS            arm tokens name[@budget[:instances]] (memetic_rule, memetic_policy,
#                   cpsat_rule, cpsat_policy), run in this order
#   BUDGET (1800), INSTANCES   defaults of the tokens; INSTANCES unset: each
#                   size's evidence instances (3-29 at 50x25n30 and 80x25n30,
#                   0-29 at 10x25); set, it applies to every size
#   SIZES ("50x25n30 80x25n30"; 10x25 is the training size), COMBOS (all eight)
#   SLOT_A (8-15), SLOT_B (16-23)   core lists, e.g. SLOT_B=16-19 or SLOT_B=16,17;
#                   SLOT_B= (empty) runs one slot, named A in the notes whatever
#                   its cores (cores 16-23 alone: SLOT_A=16-23 SLOT_B=)
#   PEER_STATE      the other project's memory state file, passed to the CP-SAT
#                   commands as PACT_PEER_MEMORY_STATE (an 80-module full model
#                   with setups waits while its last line starts with LOADING);
#                   required when such cells are planned; PEER_STATE=none: no hold
#   OWN_STATE       this project's state file, PACT_MEMORY_STATE of the CP-SAT
#                   commands (default train_log/PACT_MEMORY_STATE.txt)
#   CPSAT_ARM       the pilot's choice (ref_fullhint until the pilot ends)
#   POLICY_SEED     the policy seed of the policy-seeded arms
#   RULE_CACHE_TPL, POLICY_CACHE_TPL   folders ({size}, {seed} replaced)
#   MEM_BUDGET_GB (40), MEM_MEMETIC_GB (per memetic process)
#   BARS            frozen CPU-share bars of the CP-SAT cell types
#   MAX_INVALID (2) invalid records after which a cell is held
#   FAIL_STOP (3)   consecutive batches without any record after which it stops
#   CONTROL_CORE (3), ALL_COMMANDS=1 (DRY: print every command)
#   LAYOUTS ("8-15/16-23 8-15/16-19")   DRY: the same cells on these slot layouts
#   RESULT_ROOT, CPSAT_WORKERS, LOGDIR, HOLD   smoke runs only (other folders,
#                   fewer workers; such records are not measurements)
#   FORCE=1         start although a slot is not idle; accepted only with
#                   RESULT_ROOT, so a forced record never reaches the result folders
set -u
SELF=$(readlink -f "$0")
cd "$(dirname "$SELF")/.."
PY=${PY:-$HOME/miniconda3/envs/fjsp/bin/python}
CONTROL_CORE=${CONTROL_CORE:-3}
if ! taskset -pc "$CONTROL_CORE" $$ > /dev/null; then
  echo "REFUSING: could not pin the launcher to core $CONTROL_CORE"; exit 1
fi
DRY=${DRY:-0}
ARMS=${ARMS:-"memetic_rule cpsat_rule"}
BUDGET=${BUDGET:-1800}
INSTANCES=${INSTANCES:-}
SIZES=${SIZES:-"50x25n30 80x25n30"}
COMBOS=${COMBOS:-"N L S W LS LW SW LSW"}
SLOT_A=${SLOT_A-8-15}
SLOT_B=${SLOT_B-16-23}
CPSAT_ARM=${CPSAT_ARM:-ref_fullhint}
POLICY_SEED=${POLICY_SEED:-}
RULE_CACHE_TPL=${RULE_CACHE_TPL:-"or_solution/FAMILY/warmstart_cache_xmask_{size}"}
POLICY_CACHE_TPL=${POLICY_CACHE_TPL:-"or_solution/FAMILY/pactwarm_cache_xmask_s{seed}_{size}"}
MEM_BUDGET_GB=${MEM_BUDGET_GB:-40}
MEM_MEMETIC_GB=${MEM_MEMETIC_GB:-0.75}
BARS=${BARS:-test_results/FAMILY/timed_campaign_share_bars.json}
MAX_INVALID=${MAX_INVALID:-2}
FAIL_STOP=${FAIL_STOP:-3}
CPSAT_WORKERS=${CPSAT_WORKERS:-8}
RESULT_ROOT=${RESULT_ROOT:-}
FORCE=${FORCE:-}
[ "$FORCE" = 0 ] && FORCE=
if [ -n "$FORCE" ] && [ -z "$RESULT_ROOT" ]; then
  echo "REFUSING: FORCE=1 only with RESULT_ROOT (records of a forced start are not measurements)"; exit 1
fi
if [ -n "$RESULT_ROOT" ]; then LOGDIR=${LOGDIR:-$RESULT_ROOT/train_log}; else LOGDIR=${LOGDIR:-train_log}; fi
HOLD=${HOLD:-$LOGDIR/TIMED_HOLD}
OWN_STATE=${OWN_STATE:-$LOGDIR/PACT_MEMORY_STATE.txt}
PEER_STATE=${PEER_STATE:-}
LAYOUTS=${LAYOUTS:-"8-15/16-23 8-15/16-19"}
NOTES=$LOGDIR/timed_campaign_launch_notes.txt
STATE=$LOGDIR/timed_campaign_state
PIDFILE=$LOGDIR/timed_campaign.pid
LOG=$LOGDIR/timed_campaign.log
NAMES=(A); SLOTS=("$SLOT_A")
if [ -n "$SLOT_B" ]; then NAMES+=(B); SLOTS+=("$SLOT_B"); fi

ARGS=(--arms "$ARMS" --budget "$BUDGET" --sizes $SIZES
      --combos $COMBOS --slots "${SLOTS[@]}" --cpsat-arm "$CPSAT_ARM"
      --rule-cache "$RULE_CACHE_TPL" --policy-cache "$POLICY_CACHE_TPL"
      --workers "$CPSAT_WORKERS" --mem-budget "$MEM_BUDGET_GB" --mem-memetic "$MEM_MEMETIC_GB"
      --bars "$BARS" --max-invalid "$MAX_INVALID" --py "$PY" --logdir "$LOGDIR" --state "$STATE"
      --own-state "$OWN_STATE" --peer-state "$PEER_STATE" --control-core "$CONTROL_CORE")
[ -n "$INSTANCES" ] && ARGS+=(--instances "$INSTANCES")
[ -n "$POLICY_SEED" ] && ARGS+=(--policy-seed "$POLICY_SEED")
[ -n "$RESULT_ROOT" ] && ARGS+=(--result-root "$RESULT_ROOT")
PLAN=("$PY" scripts/timed_campaign_plan.py)
say() { echo "[timed $(date '+%m-%d %H:%M:%S')] $*"; }

if [ "$DRY" = 1 ]; then
  LAY=(); for l in $LAYOUTS; do LAY+=("${l//\// }"); done
  "${PLAN[@]}" plan "${ARGS[@]}" ${ALL_COMMANDS:+--all-commands} --layouts "${LAY[@]}"; rc=$?
  echo "DRY=1: nothing started, nothing written"
  exit $rc
fi

if [ "${TIMED_DETACHED:-0}" != 1 ]; then
  mkdir -p "$LOGDIR"
  if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2> /dev/null; then
    echo "REFUSING: a campaign launcher runs already (pid $(cat "$PIDFILE"))"; exit 1
  fi
  out=$("${PLAN[@]}" plan "${ARGS[@]}" --strict --layouts 2>&1); rc=$?
  echo "$out" | tee -a "$LOG"
  if [ $rc -ne 0 ]; then echo "REFUSING to start (see above)"; exit 1; fi
  rm -f "$PIDFILE"
  TIMED_DETACHED=1 setsid -f nohup "$SELF" >> "$LOG" 2>&1 < /dev/null
  for _ in 1 2 3 4 5 6 7 8 9 10; do [ -s "$PIDFILE" ] && break; sleep 1; done
  echo "started: pid $(cat "$PIDFILE" 2> /dev/null), log $LOG, launch notes $NOTES"
  echo "stop after the running batches: touch $HOLD; stop at once: kill -TERM \$(cat $PIDFILE)"
  exit 0
fi

# ---- detached launcher: session leader, so its process group holds every
# batch shell and every timed process
echo $$ > "$PIDFILE"
mkdir -p "$STATE"
rm -f "$STATE"/running_*.tsv
: > "$STATE/skip.txt"
on_signal() {
  trap - TERM INT HUP
  say "stopped by a signal: running cells killed with the process group; they wrote no record and run again at the next start"
  rm -f "$PIDFILE" "$STATE"/running_*.tsv
  kill -TERM -- -$$ 2> /dev/null
  exit 143
}
trap on_signal TERM INT HUP
say "launcher pid $$ on core $CONTROL_CORE; slots ${SLOTS[*]}; arms $ARMS; CP-SAT arm $CPSAT_ARM"

run_batch() {  # batch file: memetic cells at once (one per core), CP-SAT cells in turn
  local b=$1 runner cores id rec log cmd pids=()
  runner=$(sed -n '1s/.*runner=\([a-z]*\).*/\1/p' "$b")
  while IFS=$'\t' read -r cores id rec log cmd; do
    if [ "$runner" = memetic ]; then
      eval "$cmd" >> "$log" 2>&1 &
      pids+=($!)
    else
      eval "$cmd" >> "$log" 2>&1
    fi
  done < <(grep -v '^#' "$b")
  [ ${#pids[@]} -gt 0 ] && wait "${pids[@]}"
  return 0
}

declare -A PID_OF=() SLOT_OF=()
STOP=0; FAILS=0; HOLD_SAID=0
start_batch() {  # slot name, cores; sets LAST to started, done, wait or error
  local s=$1 b="$STATE/batch_$1.tsv" rc
  "${PLAN[@]}" next "${ARGS[@]}" --slot "$s" > "$b" 2> "$STATE/next_$s.err"; rc=$?
  [ -s "$STATE/next_$s.err" ] && cat "$STATE/next_$s.err"
  case $rc in
    0) ;;
    2) LAST=done; return 0 ;;
    5) LAST=wait; return 0 ;;
    *) say "planner refused slot $s:"; cat "$b"; LAST=error; STOP=1; return 1 ;;
  esac
  if ! "${PLAN[@]}" prelaunch "${ARGS[@]}" --batch "$b" --notes "$NOTES" ${FORCE:+--force}; then
    say "REFUSING slot $s: a process is on its cores (launch note in $NOTES); campaign stops after the running batches"
    LAST=error; STOP=1; return 1
  fi
  cp "$b" "$STATE/running_$s.tsv"
  run_batch "$STATE/running_$s.tsv" &
  PID_OF[$s]=$!; SLOT_OF[$!]=$s
  LAST=started
  say "slot $s: $(head -1 "$b" | cut -c3-)"
}

while :; do
  any_wait=0
  for k in "${!NAMES[@]}"; do
    s=${NAMES[$k]}
    [ -n "${PID_OF[$s]:-}" ] && continue
    [ "$STOP" = 1 ] && continue
    if [ -f "$HOLD" ]; then
      [ "$HOLD_SAID" = 0 ] && say "hold file $HOLD: no new batch"; HOLD_SAID=1; continue
    fi
    LAST=; start_batch "$s"
    [ "$LAST" = wait ] && any_wait=1
  done
  if [ ${#PID_OF[@]} -eq 0 ]; then
    rm -f "$PIDFILE"
    if [ "$STOP" = 1 ]; then say "campaign stopped (see above)"; exit 1; fi
    if [ -f "$HOLD" ]; then say "campaign held by $HOLD; remove it and start again to continue"; exit 0; fi
    if [ "$any_wait" = 1 ]; then say "ERROR: a slot waits while nothing runs"; exit 1; fi
    say "campaign finished: no cell left to run (held cells and cells without a record are listed by DRY=1)"
    exit 0
  fi
  FIN=
  wait -n -p FIN "${PID_OF[@]}"
  [ -z "$FIN" ] && continue
  s=${SLOT_OF[$FIN]}
  unset "PID_OF[$s]" "SLOT_OF[$FIN]"
  "${PLAN[@]}" classify "${ARGS[@]}" --batch "$STATE/running_$s.tsv" --notes "$NOTES"; crc=$?
  rm -f "$STATE/running_$s.tsv"
  case $crc in
    0) FAILS=0 ;;
    3) STOP=1; say "STOP: a whole batch on slot $s is invalid; check cores ${SLOTS[$([ "$s" = A ] && echo 0 || echo 1)]} before starting again" ;;
    4) FAILS=$((FAILS + 1))
       if [ "$FAILS" -ge "$FAIL_STOP" ]; then STOP=1; say "STOP: $FAILS batches in a row wrote no record (see the cell logs under $LOGDIR/timed_campaign/)"; fi ;;
    *) STOP=1; say "STOP: classification failed on slot $s" ;;
  esac
done
