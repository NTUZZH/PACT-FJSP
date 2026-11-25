#!/bin/bash
# Evaluation of one of two further test sets in the environment of the paper (exact
# non-delay mask): SET=27x25, the per-floor batch (27 modules on the default factory of 25
# stations), or SET=10x25_asymsetup, asymmetric setup matrices at the trained magnitude.
#
# Jobs: 'scripts/further_sets_plan.py jobs <set>', each the job of the evaluation plan
# (scripts/eval_newseeds_plan.py) on the model set with the set exchanged: 20x25 for 27x25,
# 10x25_oodsetup for 10x25_asymsetup. The policy xmask-v1-s<seed> on the eight combinations,
# each specialist xmask-solo-<R>-s<seed> on its own combination, and one job of the four
# dispatching rules FIFO, MOR, SPT and MWKR on the eight combinations under the exact mask
# (--rule_set masked); the plan evaluates the single-rule control on 10x25 only, so no control
# job. The command of a job is built by build_cmd of scripts/run_eval_newseeds.sh, read from
# that file at start.
#
# PRE-RUN RECORD.
# 1. PURPOSE. The arrays of the set in arm_registry.res_dir(<set>) of the exact environment
#    (test_results/FAMILY/<set>_xmask). 27x25: the policy against the specialists and the
#    dispatching rules at the per-floor batch (scripts/p118_per_floor.py, macros Pfl).
#    10x25_asymsetup: the out-of-range statistics of 10x25_oodsetup on a set that differs from
#    it in the setup magnitude range alone (scripts/p119_asym_setups.py, macros Asy).
# 2. IDEAL RESULT. The pre-flight reproduces the stored arrays of the model set, every job
#    finishes and 'scripts/further_sets_plan.py check <set>' reports no missing file. The
#    expected 27x25 result lies between the results at 20 and 30 modules; whatever comes is
#    reported.
# 3. CONTAMINATION. New folders only (data/FAMILY/<set>, <set>_xmask); no array of another
#    folder is written or read for writing. A job runs into its own staging folder and its
#    files move into the set's folder only when all exist and every array holds one row per
#    test instance; a finished job (every file present, every array of the right length) is
#    skipped; a job whose files partly exist, or exist with another length, is refused. Rollouts
#    are deterministic at one thread; no wall-clock of this runner is a measurement.
# 4. DATA ACCURACY. Every job: CPU, OMP and MKL threads 1, --device cpu --device_id "",
#    --torch_threads 1, --first_mask exact --mask_ties exact, --tag _xmask, the set's
#    descriptors and the base folder its dataset_meta.json names. Pre-flight: the seed-301
#    policy on the model set, first three instances of S and LSW, with the job's own options,
#    must equal the stored arrays bit for bit. A run is evaluated only once it is trained
#    ('scripts/further_sets_plan.py trained <model>').
#
# Usage:
#   SET=27x25 LIST_ONLY=1 scripts/run_further_sets_eval.sh   print the pre-flight and every
#                                                           command; nothing is run or written
#   SET=27x25 CORE=16-19 setsid nohup scripts/run_further_sets_eval.sh \
#       > train_log/further_27x25_eval.log 2>&1 &
# Options:
#   SET=<set>        27x25 or 10x25_asymsetup (required)
#   CORE=<list>      cores, one worker per core, one job per core at a time (required outside
#                    LIST_ONLY), e.g. 16 or 16-19
#   ONLY=<regex>     only jobs whose id matches
#   LIST_ONLY=1      list the pre-flight, the jobs (finished ones as SKIP) and the estimate
#   SEEDS=<list>     training seeds (default 301 302 303 304 305; 306 to 310 accepted, each run
#                    only once it is trained)
#   LOOP=1           repeat passes every LOOP_SLEEP s (600) while runs are still in training
#   HOLD=<file>      while it exists no new job starts (default train_log/EVAL_HOLD)
#   STAGE_ROOT, LOCKDIR, LOGDIR, RESULTS (PACT_RESULTS follows it)   folders (tests only)
# Exit status: 0 when no job failed, 1 when a job or the pre-flight failed, 2 when a setting
# refuses.
set -u
cd "$(dirname "$0")/.."
PY=$HOME/miniconda3/envs/fjsp/bin/python
SET=${SET:-}
LIST_ONLY=${LIST_ONLY:-0}
ONLY=${ONLY:-.}
CORE=${CORE:-}
LOOP=${LOOP:-0}
LOOP_SLEEP=${LOOP_SLEEP:-600}
HOLD=${HOLD:-train_log/EVAL_HOLD}
case "$SET" in
  27x25|10x25_asymsetup) ;;
  *) echo "REFUSING: SET=<27x25|10x25_asymsetup> is required (got '$SET')"; exit 2 ;;
esac
STAGE_ROOT=${STAGE_ROOT:-test_results/.eval_staging/further_$SET}
LOCKDIR=${LOCKDIR:-train_log/further_sets_eval_locks/$SET}
LOGDIR=${LOGDIR:-train_log/further_sets_eval/$SET}
export RESULTS=${RESULTS:-test_results/FAMILY}
export PACT_RESULTS=${PACT_RESULTS:-$RESULTS}
export PACT_ENV=exact PACT_SEEDS=301-305
export SEEDS=${SEEDS:-301 302 303 304 305}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
unset PACT_FIRST_MASK PACT_MASK_TOL
FP="$PY -B scripts/further_sets_plan.py"
say() { echo "[further-eval $SET $(date '+%m-%d %H:%M:%S')] $*"; }

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
if [ "$LIST_ONLY" = 1 ]; then
  CORE_LIST=("<core>")
else
  [[ "$CORE" =~ ^[0-9,-]+$ ]] || { echo "REFUSING: CORE=<list> is required (e.g. CORE=16-19)"; exit 2; }
  read -ra CORE_LIST <<< "$(expand_cores "$CORE")"
  # the launcher shell, and so every worker, wait and helper it starts, stays on CORE
  if ! taskset -pc "$(IFS=,; echo "${CORE_LIST[*]}")" $$ > /dev/null; then
    echo "REFUSING: could not pin the launcher to CORE=$CORE"; exit 2
  fi
fi
WORKERS=${#CORE_LIST[@]}

JOBFILE=$(mktemp "${TMPDIR:-/tmp}/further_${SET}_jobs.XXXXXX")
trap 'rm -f "$JOBFILE"' EXIT
$FP jobs "$SET" > "$JOBFILE" || { say "REFUSING: scripts/further_sets_plan.py jobs $SET failed: $(tail -1 "$JOBFILE")"; exit 2; }
FOLDER_OF_SET="$RESULTS/$(head -1 "$JOBFILE" | cut -d'|' -f21)"
say "$(wc -l < "$JOBFILE") jobs; results in $FOLDER_OF_SET; seeds $SEEDS"

# ---- pre-flight: the seed-301 policy on the model set reproduces the stored arrays
preflight() {
  local out line stored id group camp envn device omp tt extra model methods data base tag limit regimes
  local token blind bk sched oroot folder n ckpt log pre rset files root rc
  out=$($FP preflight "$SET") || { say "PREFLIGHT FAILED: $out"; return 1; }
  line=$(sed -n 1p <<< "$out"); stored=$(sed -n 2p <<< "$out")
  IFS='|' read -r id group camp envn device omp tt extra model methods data base tag limit regimes token blind bk sched oroot folder n ckpt log pre rset files <<< "$line"
  root="$STAGE_ROOT/preflight"
  build_cmd "${CORE_LIST[0]}" "$device" "$omp" "$tt" "$extra" "$model" "$methods" "$data" "$base" \
    "$tag" "$limit" "$regimes" "$token" "$blind" "$bk" - exact "$root" "$rset"
  if [ "$LIST_ONLY" = 1 ]; then
    echo "PREFLIGHT: $(show)"
    echo "  then: $PY scripts/eval_newseeds_plan.py compare $root/$folder $stored greedy+$model \"$regimes\" $limit"
    return 0
  fi
  rm -rf "$root"; mkdir -p "$root"
  say "PREFLIGHT: $(show)"
  "${CMD[@]}" > "$LOGDIR/preflight.log" 2>&1; rc=$?
  [ $rc -eq 0 ] && { $PY -B scripts/eval_newseeds_plan.py compare "$root/$folder" "$stored" \
                       "greedy+$model" "$regimes" "$limit" >> "$LOGDIR/preflight.log" 2>&1; rc=$?; }
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
    for f in $files; do
      [[ "$f" == "$FOLDER_OF_SET/"* ]] || { [ "$w" = 0 ] && { say "REFUSING $id: $f lies outside $FOLDER_OF_SET"; echo "$id outside" >> "$FAILED"; }; continue 2; }
    done
    have=0; total=0
    for f in $files; do total=$((total + 1)); [ -e "$f" ] && have=$((have + 1)); done
    if [ $have -eq $total ]; then
      if $FP rows "$n" $files > /dev/null 2>&1; then
        [ "$LIST_ONLY" = 1 ] && echo "SKIP $id (finished)"; continue
      fi
      [ "$w" = 0 ] && { say "REFUSING $id: its files exist but an array does not hold $n rows ($($FP rows "$n" $files | head -1))"; echo "$id length" >> "$FAILED"; }
      continue
    fi
    if [ $have -gt 0 ]; then
      [ "$w" = 0 ] && { say "REFUSING $id: $have of $total files exist in place; inspect and remove them by hand"; echo "$id partial" >> "$FAILED"; }
      continue
    fi
    if [ "$sched" != "-" ] || [ "$envn" != exact ]; then
      [ "$w" = 0 ] && { say "REFUSING $id: not an exact job without schedule export"; echo "$id layout" >> "$FAILED"; }
      continue
    fi
    if [ "$model" != rules ] && ! $FP trained "$model" > /dev/null 2>&1; then
      PENDING=$((PENDING + 1))
      [ "$LIST_ONLY" = 1 ] && echo "NOT TRAINED $id: $($FP trained "$model" | head -1)"
      continue
    fi
    stage="$STAGE_ROOT/$id"
    build_cmd "$core" "$device" "$omp" "$tt" "$extra" "$model" "$methods" "$data" "$base" "$tag" \
      "$limit" "$regimes" "$token" "$blind" "$bk" - exact "$stage/out" "$rset"
    if [ "$LIST_ONLY" = 1 ]; then
      echo "JOB $id ($total files, $n instances x $(wc -w <<< "$regimes") combinations x $(wc -w <<< "$methods") methods): $(show)"
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
    if [ $ok = 1 ] && ! $FP rows "$n" "${arrays[@]}" >> "$LOGDIR/jobs/${id//\//_}.log" 2>&1; then
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

if [ "$LIST_ONLY" = 1 ]; then
  FAILED=/dev/null
  preflight
  one_pass 0 "<core>"
  echo "runs not yet trained (listed above, would wait): $PENDING"
  $FP estimate "$SET"
  $FP check "$SET"
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
$FP check "$SET"
if [ -s "$FAILED" ]; then
  say "$(wc -l < "$FAILED") job(s) failed or were refused: $(tr '\n' ' ' < "$FAILED")"; exit 1
fi
rm -f "$FAILED"
say "launcher finished"
exit 0
