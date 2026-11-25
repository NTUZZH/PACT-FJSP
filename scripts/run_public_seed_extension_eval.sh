#!/bin/bash
# Evaluation of the second five training seeds (306 to 310) of the benchmark from public
# instances under the active mask, on pub_SD2_test_10x5+mix, for the comparison of the policy
# with the specialists with ten seeds (scripts/pub_seed_extension.py,
# scripts/p115_public_seed_extension.py). AN ADDITION AFTER THE FIVE-SEED RESULT WAS KNOWN: the
# five-seed record of scripts/p89_public_family.py --family active stays the reading of record.
#
# Jobs: one per run of the second five seeds, 'scripts/pub_seed_extension.py evaljobs': the job
# of the evaluation plan (scripts/eval_newseeds_plan.py, group pubactfam,
# pub_SD2_test_10x5+mix) of the same recipe at seed 305 with the run name exchanged. The policy
# is evaluated greedy on the eight combinations, a specialist on its own. The command of a job
# is built by build_cmd of scripts/run_eval_newseeds.sh, read from that file at start, so it is
# the command of the plan's job with only the run name exchanged
# (tests/test_public_seed_extension.py compares them for seed 305). eval_family.py restores
# mask_mode from each run's stored configuration, so a run is rolled out under the active mask.
#
# PRE-RUN RECORD.
# 1. PURPOSE. The arrays of the second five seeds that the comparison with ten seeds reads,
#    in the folder of the first seeds (test_results/PUBFAMILY/pub_SD2_test_10x5+mix_xmask). The
#    numbers land in the ten-seed rows of p115, beside the five-seed record.
# 2. IDEAL RESULT. The pre-flight reproduces the stored arrays, every job finishes and
#    'scripts/pub_seed_extension.py check' reports 45 complete runs.
# 3. CONTAMINATION. New run names only; the registry refuses their seeds, and no glob or
#    pattern of the readers of the first five seeds takes their files. A job runs into its own
#    staging folder and its files move into place only when all exist and every array holds
#    one row per test instance; a job whose files partly exist in place is refused, a finished
#    job skipped. No wall-clock of this launcher is a measurement.
# 4. DATA ACCURACY. Every job: CPU, OMP and MKL threads 1, --device cpu --device_id "",
#    --torch_threads 1, --first_mask exact --mask_ties exact, --tag _xmask, the plan's test set,
#    base records and best-known table. Pre-flight: the stored policy run pubx-act-s305
#    re-evaluated with its job's own options on the first five instances of N and LSW must give
#    the stored arrays bit for bit (the evaluation path of the first five seeds). A run is
#    evaluated only once it is trained ('scripts/pub_seed_extension.py trained').
#
# Usage:
#   DRY=1 scripts/run_public_seed_extension_eval.sh   print the state of the runs, the
#                                                     pre-flight and every command; nothing
#                                                     is pinned, run or written
#   CORES=16-19 scripts/run_public_seed_extension_eval.sh       once every run is trained
#   CORES=16-19 LOOP=1 setsid nohup scripts/run_public_seed_extension_eval.sh \
#       > train_log/public_seed_extension_eval.log 2>&1 &   while runs are still in training
# Options:
#   CORES=<list>     cores, one worker per core, one job per core at a time (required
#                    outside DRY)
#   WORKERS=<n>      the first n cores of CORES (default all)
#   LOOP=1           repeat passes every LOOP_SLEEP s (600) while runs are still in training;
#                    without it the launcher refuses while a selected run is not trained
#   ONLY=<regex>     only jobs whose id matches
#   HOLD=<file>      while it exists no new job starts (default train_log/EVAL_HOLD)
#   STAGE_ROOT, LOCKDIR, LOGDIR, PUB_RESULTS    folders (tests only)
# Exit status: 0 when no job failed, 1 when a job or the pre-flight failed, 2 when a run is not
# trained (without LOOP=1) or a setting refuses.
set -u
cd "$(dirname "$0")/.."
PY=$HOME/miniconda3/envs/fjsp/bin/python
DRY=${DRY:-0}
ONLY=${ONLY:-.}
CORES=${CORES:-}
LOOP=${LOOP:-0}
LOOP_SLEEP=${LOOP_SLEEP:-600}
HOLD=${HOLD:-train_log/EVAL_HOLD}
STAGE_ROOT=${STAGE_ROOT:-test_results/.eval_staging}
LOCKDIR=${LOCKDIR:-train_log/public_seed_extension_eval_locks}
LOGDIR=${LOGDIR:-train_log/public_seed_extension_eval}
export PUB_RESULTS=${PUB_RESULTS:-test_results/PUBFAMILY}
unset PACT_FIRST_MASK PACT_MASK_TOL
PSE="$PY -B scripts/pub_seed_extension.py"
PREFIX=10x5+pubfamily+
say() { echo "[pubseedext-eval $(date '+%m-%d %H:%M:%S')] $*"; }

# the command builder, the printer and the mask options of the plan's queue, from its file
QUEUE=scripts/run_eval_newseeds.sh
eval "$(sed -n '/^build_cmd() {/,/^}/p; /^show() {.*}$/p; /^MASK_OPTS=/p' "$QUEUE")"
if ! declare -F build_cmd > /dev/null || ! declare -F show > /dev/null || [ -z "${MASK_OPTS:-}" ]; then
  echo "REFUSING: could not read build_cmd, show and MASK_OPTS from $QUEUE"; exit 2
fi

expand_cores() {
  local out=() part a b c
  IFS=',' read -ra parts <<< "$1"
  for part in "${parts[@]}"; do
    if [[ $part == *-* ]]; then a=${part%-*}; b=${part#*-}; for ((c=a; c<=b; c++)); do out+=("$c"); done
    else out+=("$part"); fi
  done
  echo "${out[@]}"
}
if [ "$DRY" = 1 ]; then
  CORE_LIST=("<core>")
else
  [[ "$CORES" =~ ^[0-9,-]+$ ]] || { echo "REFUSING: CORES=<list> is required (e.g. CORES=16-19)"; exit 2; }
  read -ra CORE_LIST <<< "$(expand_cores "$CORES")"
  # the launcher shell, and so every worker, wait and helper it starts, stays on CORES
  if ! taskset -pc "$(IFS=,; echo "${CORE_LIST[*]}")" $$ > /dev/null; then
    echo "REFUSING: could not pin the launcher to CORES=$CORES"; exit 2
  fi
fi
WORKERS=${WORKERS:-${#CORE_LIST[@]}}
if [ "$WORKERS" -gt "${#CORE_LIST[@]}" ]; then
  echo "REFUSING: WORKERS=$WORKERS but CORES=$CORES names ${#CORE_LIST[@]} cores (one worker per core)"; exit 2
fi
say "second five seeds of the benchmark from public instances under the active mask: an addition after the five-seed result was known"

JOBFILE=$(mktemp)
trap 'rm -f "$JOBFILE"' EXIT
$PSE evaljobs > "$JOBFILE" || { say "REFUSING: scripts/pub_seed_extension.py evaljobs failed: $(tail -1 "$JOBFILE")"; exit 2; }
say "$(wc -l < "$JOBFILE") jobs of the second five seeds; results under $PUB_RESULTS"

# ---- the gate: every selected run is trained, unless LOOP=1 waits for it
SELECTED=()
while IFS= read -r line; do
  IFS='|' read -r id _ _ _ _ _ _ _ model _ <<< "$line"
  [[ "$id" =~ $ONLY ]] && SELECTED+=("${model#"$PREFIX"}")
done < "$JOBFILE"
if [ ${#SELECTED[@]} -eq 0 ]; then say "REFUSING: no job matches ONLY=$ONLY"; exit 2; fi
UNTRAINED=$($PSE trained "${SELECTED[@]}" | grep -v ': trained$' | cut -d: -f1 | sort -u)
N_UNTRAINED=$(grep -c . <<< "$UNTRAINED")
if [ "$N_UNTRAINED" -gt 0 ]; then
  if [ "$DRY" = 1 ]; then
    say "gate (not enforced in a dry run): $N_UNTRAINED of ${#SELECTED[@]} selected runs not trained"
  elif [ "$LOOP" != 1 ]; then
    say "REFUSING: $N_UNTRAINED of ${#SELECTED[@]} selected runs are not trained (first: $(head -1 <<< "$UNTRAINED")); see 'scripts/pub_seed_extension.py check', or start with LOOP=1 to wait for them"; exit 2
  else
    say "$N_UNTRAINED of ${#SELECTED[@]} selected runs not trained yet; LOOP=1 waits for them"
  fi
else
  say "gate: all ${#SELECTED[@]} selected runs are trained"
fi

# ---- pre-flight: the stored policy run of seed 305 reproduced with its job's options
preflight() {
  local line id group camp envn device omp tt extra model methods data base tag limit regimes
  local token blind bk sched oroot folder n ckpt log pre rset files root rc
  line=$($PY -B -c "
import sys; sys.path.insert(0, 'scripts')
import pub_seed_extension as S
j = S.plan_jobs(305)[('released', None)]
print(S.job_line(j))") || { say "PREFLIGHT FAILED: no plan job of seed 305"; return 1; }
  IFS='|' read -r id group camp envn device omp tt extra model methods data base tag limit regimes token blind bk sched oroot folder n ckpt log pre rset files <<< "$line"
  root="$STAGE_ROOT/pubseedext_preflight"
  build_cmd "${CORE_LIST[0]}" "$device" "$omp" "$tt" "$extra" "$model" "$methods" "$data" "$base" \
    "$tag" 5 "N LSW" "$token" "$blind" "$bk" - exact "$root" "$rset"
  if [ "$DRY" = 1 ]; then
    echo "PREFLIGHT: $(show)"
    echo "  then: $PY scripts/eval_newseeds_plan.py compare $root/$folder $oroot/$folder greedy+$model \"N LSW\" 5"
    return 0
  fi
  rm -rf "$root"; mkdir -p "$root"
  say "PREFLIGHT: $(show)"
  "${CMD[@]}" > "$LOGDIR/preflight.log" 2>&1; rc=$?
  [ $rc -eq 0 ] && { $PY -B scripts/eval_newseeds_plan.py compare "$root/$folder" "$oroot/$folder" \
                       "greedy+$model" "N LSW" 5 >> "$LOGDIR/preflight.log" 2>&1; rc=$?; }
  say "PREFLIGHT $([ $rc -eq 0 ] && echo passed || echo FAILED): $(tail -1 "$LOGDIR/preflight.log")"
  [ $rc -eq 0 ] && rm -rf "$root"
  return $rc
}

# ---- one pass of one worker over the job list
one_pass() {  # worker core -> sets RAN PENDING
  local w=$1 core=$2 line id group camp envn device omp tt extra model methods data base tag
  local limit regimes token blind bk sched oroot folder n ckpt log pre rset files have total f
  local lk stage staged ok rc arrays
  RAN=0; PENDING=0
  while IFS= read -r line; do
    IFS='|' read -r id group camp envn device omp tt extra model methods data base tag limit regimes token blind bk sched oroot folder n ckpt log pre rset files <<< "$line"
    [[ "$id" =~ $ONLY ]] || continue
    have=0; total=0
    for f in $files; do total=$((total + 1)); [ -e "$f" ] && have=$((have + 1)); done
    [ $have -eq $total ] && { [ "$DRY" = 1 ] && echo "SKIP $id (finished)"; continue; }
    if [ $have -gt 0 ]; then
      [ "$w" = 0 ] && { say "REFUSING $id: $have of $total files exist in place; inspect and remove them by hand"; echo "$id partial" >> "$FAILED"; }
      continue
    fi
    if [ "$sched" != "-" ] || [ "$envn" != exact ]; then
      [ "$w" = 0 ] && { say "REFUSING $id: not an exact job without schedule export"; echo "$id layout" >> "$FAILED"; }
      continue
    fi
    if ! $PSE trained "${model#"$PREFIX"}" > /dev/null 2>&1; then
      PENDING=$((PENDING + 1))
      [ "$DRY" = 1 ] || continue
    fi
    stage="$STAGE_ROOT/$id"
    build_cmd "$core" "$device" "$omp" "$tt" "$extra" "$model" "$methods" "$data" "$base" "$tag" \
      "$limit" "$regimes" "$token" "$blind" "$bk" - exact "$stage/out" "$rset"
    if [ "$DRY" = 1 ]; then
      echo "JOB $id [$group] ($total files, $n instances x $(wc -w <<< "$regimes") combinations x $(wc -w <<< "$methods") methods; pre-flight $pre): $(show)"
      continue
    fi
    lk="$LOCKDIR/$id"
    mkdir "$lk" 2> /dev/null || continue                   # another worker has it
    echo "$BASHPID $core $(date +%s)" > "$lk/pid"
    have=0; for f in $files; do [ -e "$f" ] && have=$((have + 1)); done
    if [ $have -gt 0 ]; then rm -rf "$lk"; continue; fi     # finished meanwhile
    while [ -f "$HOLD" ]; do sleep 60; done
    rm -rf "$stage"; mkdir -p "$stage/out"
    say "worker $w core $core RUN $id: $(show)"
    "${CMD[@]}" > "$LOGDIR/jobs/${id//\//_}.log" 2>&1; rc=$?
    ok=1; arrays=()
    if [ $rc -eq 0 ]; then
      for f in $files; do
        staged="$stage/out/${f#"$oroot"/}"
        [ -e "$staged" ] || { ok=0; say "FAILED $id: $staged was not written"; break; }
        [[ "$f" == *.npy ]] && arrays+=("$staged")
      done
    else ok=0; fi
    if [ $ok = 1 ] && ! $PSE rows "$n" "${arrays[@]}" >> "$LOGDIR/jobs/${id//\//_}.log" 2>&1; then
      ok=0; say "FAILED $id: an array does not hold $n rows (see $LOGDIR/jobs/${id//\//_}.log)"
    fi
    if [ $ok = 1 ]; then
      for f in $files; do
        staged="$stage/out/${f#"$oroot"/}"
        mkdir -p "$(dirname "$f")"
        mv -n "$staged" "$f"
        [ -e "$staged" ] && { ok=0; say "FAILED $id: $f appeared meanwhile; the staged copy stays in $stage"; }
      done
    fi
    if [ $ok = 1 ]; then
      rm -rf "$stage"; RAN=$((RAN + 1)); say "worker $w done $id ($total files)"
    else
      echo "$id" >> "$FAILED"; say "FAILED $id (exit $rc); see $LOGDIR/jobs/${id//\//_}.log and $stage"
    fi
    rm -rf "$lk"
  done < "$JOBFILE"
}

worker() {  # index core
  local w=$1 core=$2
  while :; do
    one_pass "$w" "$core"
    say "worker $w pass: $RAN run, $PENDING awaiting training"
    [ "$LOOP" = 1 ] && [ "$PENDING" -gt 0 ] || break
    [ "$RAN" -gt 0 ] || sleep "$LOOP_SLEEP"
  done
}

if [ "$DRY" = 1 ]; then
  FAILED=/dev/null
  preflight
  one_pass 0 "<core>"
  echo "runs not yet trained (listed above, would wait): $PENDING"
  $PSE check
  exit 0
fi
mkdir -p "$LOGDIR/jobs" "$LOCKDIR" "$STAGE_ROOT"
FAILED="$LOGDIR/failed_$$.txt"
: > "$FAILED"
preflight || { say "the pre-flight failed; no job started"; exit 1; }
for ((w=0; w<WORKERS; w++)); do
  worker "$w" "${CORE_LIST[$w]}" >> "$LOGDIR/worker_$w.log" 2>&1 &
done
wait
say "workers finished:"
$PSE check
if [ -s "$FAILED" ]; then
  say "$(wc -l < "$FAILED") job(s) failed or were refused: $(tr '\n' ' ' < "$FAILED")"; exit 1
fi
rm -f "$FAILED"
say "launcher finished"
exit 0
