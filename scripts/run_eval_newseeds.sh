#!/bin/bash
# Evaluation campaign of the exact non-delay mask (the environment of the paper):
# every principal arm (seeds 301-305), the further exact-mask arms (seeds
# 301-303), the 30-instance production sets with schedule export, the
# dispatching rules, and, when enabled, the public-instance family, the legacy
# seeds 304-305 and the HGNN port of Song et al. (group hgnn). The plan (names,
# folders, files) is scripts/eval_newseeds_plan.py on scripts/arm_registry.py.
#
# PRE-RUN RECORD.
#
# 1. PURPOSE. Produce every array the analysis scripts read in the exact-mask
#    environment (PACT_ENV=exact PACT_SEEDS=301-305), mirroring the stored legacy
#    campaigns of the seed-303 runs and of each further arm's legacy counterpart,
#    plus the 30-instance production sets of the paper. The numbers land in
#    scripts/p88_family_inference.py and every converted p-script under the
#    exact mask.
# 2. IDEAL RESULT. All pre-flights pass, every job finishes, and the closing
#    check reports no missing file and no consistency failure. A failed pre-flight
#    blocks the jobs that depend on it: an evaluation path that does not
#    reproduce, or one that is not deterministic, would make the new arrays
#    unrepeatable, and the cause is found first.
# 3. CONTAMINATION. Exact-mask results only go to <folder>_xmask (and
#    test_results/PUBFAMILY/<set>_xmask); new run names only. A job runs into its
#    own staging folder (STAGE_ROOT) and its files move into place only when all of
#    them exist, so a killed job leaves nothing behind in the result folders and a
#    restart redoes it. A job whose files partly exist in place is refused, never
#    overwritten; a finished job is skipped. Workers claim jobs through lock
#    folders (mkdir is atomic), so two workers never take the same job. Rollouts
#    are deterministic; no wall-clock from this queue is a measurement (p19 only
#    warns on the spread of the time column).
# 4. DATA ACCURACY. Every exact job runs on the CPU with one thread: OMP and MKL
#    threads 1, --device cpu --device_id "", --torch_threads 1, --first_mask exact
#    --mask_ties exact, --tag _xmask. One thread is fixed because the thread count
#    changes the arrays: the stored seed-303 policy at 10x25 gives 711 of 800
#    makespans identical between one and four torch threads (20x25: 15 of 16,
#    50x25: 5 of 6 on the first instances), while four threads reproduce the
#    stored CPU campaign 10x25_cpuref 800 of 800. Instances, base sets and
#    combinations are those of the stored campaigns; the 30-instance sets hold the
#    20 stored instances at indices 0-19, and the check requires their makespans
#    to equal the 20-instance exact results of the same run.
#    The hgnn group runs backbones/hgnn/eval_family_hgnn.py (job field methods =
#    hgnn) with the same settings: OMP and MKL threads 1, CUDA_VISIBLE_DEVICES
#    empty, --device cpu --threads 1, --tag _xmask; the evaluator reads the mask and
#    the reward from the run's configuration and the base folder from the data
#    set's dataset_meta.json. It writes Result_<run>_<R>.npy and no summary file.
#
# PRE-FLIGHTS (run once per code version; a marker in PREFLIGHT_DIR records a
# fingerprint of eval_family.py, params.py, common_utils.py, constraint_family/
# and model/):
#   xdet           one exact job (the first trained exact released run, else the
#                  stored legacy seed-303 checkpoint under the exact options; 10x25,
#                  S and LSW, 5 instances) evaluated twice: identical arrays.
#   xcross_policy  the stored CPU campaign 10x25_cpuref of the seed-303 released
#                  run re-evaluated in full with its own settings (legacy mask,
#                  four torch threads; 8 combinations x 100 instances): 800 of 800
#                  makespans bit for bit equal to the stored files. It runs on the
#                  first four cores of CORES (fewer if fewer workers), before any
#                  worker starts.
#   xcross_rules   the stored legacy rules at 10x25 (one thread): bit for bit.
#   xdet_rules     one masked rules job of the exact environment (--rule_set
#                  masked; 10x25, S and LSW, 10 instances) evaluated twice:
#                  identical arrays. Every exact rules job uses --rule_set masked.
#   pubdet        xdet on pub_Brandimarte with the first trained public policy.
#   legacy_<c>     LEGACY45 only: the stored seed-303 run of campaign c with c's
#                  device and threads, bit for bit (round-one pre-flight).
#   hgnn_cross,    HGNN=1 only: the HGNN evaluator on the CPU with one thread
#   hgnn_cross_pub reproduces the stored earlier-mask evaluation (written on the
#                  GPU, one thread) of hgnn-certified-joint-s301 and of
#                  hgnn-published-joint-s301 on the first two instances of S and
#                  LSW, bit for bit. Their markers record a fingerprint that adds
#                  the HGNN code (backbones/hgnn) to the one above; the hgnn jobs
#                  need both and none of the other pre-flights.
# Several launchers may run at once on the same folders (PART=0/2 and PART=1/2,
# or disjoint CORES): one launcher at a time runs the pre-flights (a lock folder
# in PREFLIGHT_DIR) and the others reuse the markers; jobs are claimed per job.
# Logs: LOGDIR/worker_<part>_<w>.log (appended), LOGDIR/jobs/<id>.log, and
# LOGDIR/preflight_<name>.log.
#
# Usage:
#   DRY=1 scripts/run_eval_newseeds.sh                 list every command, run nothing
#   CORES=10-13 setsid nohup scripts/run_eval_newseeds.sh > train_log/eval_newseeds_queue.log 2>&1 &
# Options:
#   CORES=<list>     cores, one worker per core (e.g. 10-13,16); default 10
#   WORKERS=<n>      use the first n cores of CORES (default: all of them)
#   PART=<i>/<n>     only jobs whose index is i modulo n (independent invocations)
#   EVAL_GROUPS=<list>  default 'principal secondary n30 rules'
#   PUBLIC=1         add the public-instance family (off by default)
#   PUBEQ=1          add the equal-experience arm of the public family (pubx-eq-s<seed>,
#                    off by default); EVAL_GROUPS=pubeq evaluates that arm alone. Its jobs
#                    are the pubx-v1 jobs with the run name changed and wait for the
#                    pubdet pre-flight like the public group.
#   PUBACT=1         add the active-rule arm of the public family (pubx-act-s<seed>,
#                    seeds 301-303, SD2 10x5 and SD2 20x10; off by default);
#                    EVAL_GROUPS=pubact evaluates that arm alone. Its jobs are the
#                    pubx-v1 jobs with the run name changed and wait for the pubdet
#                    pre-flight; eval_family.py restores the arm's mask_mode from its
#                    stored configuration.
#   PUBACTFAM=1, PUBACTSENS=1, PUBACTEQ=1
#                    add the public family under the active mask (groups pubactfam: policy,
#                    control and specialists, seeds 301-305, the six public test sets, without
#                    the jobs of group pubact; pubactsens: the same runs on the sensitivity
#                    sets; pubacteq: pubx-eqact-s<seed> on the sets of group pubeq; off by
#                    default). Each job is the job of the non-delay counterpart with the run
#                    name changed and waits for the pubdet pre-flight and for its run to
#                    finish; eval_family.py restores mask_mode from the run's configuration.
#                    The rules under the active mask come from scripts/p109_public_active_rules.py.
#   LEGACY45=1       add legacy seeds 304-305 (off by default; stored devices,
#                    GPU jobs wait for train_log/GPU_HOLD and for training runs)
#   HGNN=1           add the HGNN runs of scripts/run_hgnn_exact_queue.sh (off by
#                    default); EVAL_GROUPS=hgnn evaluates them alone. HGNN_LR=<rate>
#                    names the learning rate of the runs (default: the selection of
#                    scripts/hgnn_select_lr.py; the planner fails without either);
#                    HGNN_SOLO=auto|1|0 the specialists (auto: once one is trained).
#   ONLY=<regex>     only jobs whose id (env:campaign:run) matches
#   LOOP=1           repeat passes every LOOP_SLEEP s (600) while jobs await training
#   EVAL_HOLD file   train_log/EVAL_HOLD: while it exists, no new job starts
#   PREFLIGHT_DIR, STAGE_ROOT, LOCKDIR, LOGDIR, RESULTS, PUB_RESULTS, STORED,
#   CACHE_ROOT, HOLD   folders and the hold file (tests only)
set -u
cd "$(dirname "$0")/.."
PY=$HOME/miniconda3/envs/fjsp/bin/python
DRY=${DRY:-0}
ONLY=${ONLY:-.}
CORES=${CORES:-10}
PART=${PART:-0/1}
LOOP=${LOOP:-0}
LOOP_SLEEP=${LOOP_SLEEP:-600}
HOLD=${HOLD:-train_log/EVAL_HOLD}
GPU_HOLD=train_log/GPU_HOLD
PREFLIGHT_DIR=${PREFLIGHT_DIR:-${TMPDIR:-/tmp}/pact_eval_preflight}
STAGE_ROOT=${STAGE_ROOT:-test_results/.eval_staging}
LOCKDIR=${LOCKDIR:-train_log/eval_newseeds_locks}
LOGDIR=${LOGDIR:-train_log/eval_newseeds}
export EVAL_GROUPS=${EVAL_GROUPS:-principal secondary n30 rules} PUBLIC=${PUBLIC:-0} LEGACY45=${LEGACY45:-0}
export RESULTS=${RESULTS:-test_results/FAMILY} STORED=${STORED:-test_results/FAMILY}
export PUB_RESULTS=${PUB_RESULTS:-test_results/PUBFAMILY} CACHE_ROOT=${CACHE_ROOT:-}
export EXACT_SEEDS=${EXACT_SEEDS:-301 302 303 304 305} LEGACY_SEEDS=${LEGACY_SEEDS:-304 305}
export HGNN=${HGNN:-0} HGNN_LR=${HGNN_LR:-} HGNN_SOLO=${HGNN_SOLO:-auto}
MASK_OPTS="--first_mask exact --mask_ties exact"
PLAN="$PY scripts/eval_newseeds_plan.py"
say() { echo "[evalnew $(date '+%m-%d %H:%M:%S')] $*"; }

expand_cores() {
  local out=() part a b c
  IFS=',' read -ra parts <<< "$1"
  for part in "${parts[@]}"; do
    if [[ $part == *-* ]]; then a=${part%-*}; b=${part#*-}; for ((c=a; c<=b; c++)); do out+=("$c"); done
    else out+=("$part"); fi
  done
  echo "${out[@]}"
}
read -ra CORE_LIST <<< "$(expand_cores "$CORES")"
WORKERS=${WORKERS:-${#CORE_LIST[@]}}
if [ "$WORKERS" -gt "${#CORE_LIST[@]}" ]; then
  echo "REFUSING: WORKERS=$WORKERS but CORES=$CORES names ${#CORE_LIST[@]} cores (one worker per core)"; exit 1
fi
PART_I=${PART%/*}; PART_N=${PART#*/}
# The launcher shell itself, and so every worker, sleep and planner call it
# starts, stays on CORES: its waiting loops never land on other cores.
if ! taskset -pc "$(IFS=,; echo "${CORE_LIST[*]}")" $$ > /dev/null; then
  echo "REFUSING: could not pin the launcher to CORES=$CORES"; exit 1
fi

# eval_family.py command of a job or pre-flight; sets the array CMD
build_cmd() {  # core device omp tt extra model methods data base tag limit regimes token blind bk sched mask out_root rule_set
  local core=$1 device=$2 omp=$3 tt=$4 extra=$5 model=$6 methods=$7 data=$8 base=$9
  local tag=${10} limit=${11} regimes=${12} token=${13} blind=${14} bk=${15} sched=${16} mask=${17} root=${18}
  local rule_set=${19:-classical}
  if [ "$methods" = hgnn ]; then
    # the HGNN port: its own evaluator, always on the CPU; the data folder gives the
    # size and its parent the data root; mask and reward come from the run's configuration
    CMD=(env -u PACT_FIRST_MASK -u PACT_MASK_TOL OMP_NUM_THREADS=$omp MKL_NUM_THREADS=$omp
         CUDA_VISIBLE_DEVICES= taskset -c "$core" "$PY" backbones/hgnn/eval_family_hgnn.py
         --family_model "$model" --data_root "$(dirname "$data")" --sizes "$(basename "$data")"
         --regimes "${regimes// /,}" --device cpu --threads "$tt" --out_root "$root")
    [ "$tag" != "-" ] && CMD+=(--tag "$tag")
    [ "$limit" != 0 ] && CMD+=(--limit "$limit")
    return 0
  fi
  CMD=(env -u PACT_FIRST_MASK -u PACT_MASK_TOL OMP_NUM_THREADS=$omp MKL_NUM_THREADS=$omp)
  [ "$extra" != "-" ] && CMD+=($extra)
  if [ "$device" = gpu ]; then CMD+=(CUDA_VISIBLE_DEVICES=0); else CMD+=(CUDA_VISIBLE_DEVICES=); fi
  CMD+=(taskset -c "$core" "$PY" eval_family.py --family_model "$model"
        --data "$data" --base "$base" --regimes $regimes --methods $methods --out_root "$root")
  # the device and its id always travel together (eval_family.py sets
  # CUDA_VISIBLE_DEVICES from --device_id when it loads a policy)
  if [ "$device" = gpu ]; then CMD+=(--device cuda --device_id 0); else CMD+=(--device cpu --device_id ""); fi
  [ "$tt" != 0 ] && CMD+=(--torch_threads "$tt")
  [ "$tag" != "-" ] && CMD+=(--tag "$tag")
  [ "$limit" != 0 ] && CMD+=(--limit "$limit")
  [ "$token" != true ] && CMD+=(--token_mode "$token")
  [ "$blind" != "-" ] && CMD+=(--blind_channels "$blind")
  [ "$bk" != "-" ] && CMD+=(--best_known "$bk")
  [ "$sched" != "-" ] && CMD+=(--save_schedules "$sched")
  [ "$mask" = exact ] && CMD+=($MASK_OPTS)
  [ "$rule_set" != classical ] && CMD+=(--rule_set "$rule_set")
  return 0
}
show() { printf '%q ' "${CMD[@]}"; }

wait_turn() {  # device
  [ "$DRY" = 1 ] && return 0
  local said=0
  while [ -f "$HOLD" ] || { [ "$1" = gpu ] && { [ -f "$GPU_HOLD" ] || pgrep -f "[t]rain_family(_hgnn)?\.py" > /dev/null; }; }; do
    [ $said = 0 ] && { say "waiting: $HOLD present${1:+ ($1 job)}"; said=1; }
    sleep 60
  done
}

FINGERPRINT=$(cat eval_family.py params.py common_utils.py constraint_family/*.py model/*.py 2>/dev/null | sha256sum | cut -c1-16)
# the HGNN pre-flights also depend on the HGNN code
HGNN_FINGERPRINT=$({ echo "$FINGERPRINT"; cat backbones/hgnn/*.py backbones/hgnn/*.json backbones/hgnn/env/*.py \
  backbones/hgnn/graph/*.py backbones/hgnn/utils/*.py 2>/dev/null; } | sha256sum | cut -c1-16)
fp_of() {  # pre-flight name -> fingerprint its marker records
  case "$1" in hgnn_*) echo "$HGNN_FINGERPRINT" ;; *) echo "$FINGERPRINT" ;; esac
}

# ---- pre-flights (launcher only)
run_preflights() {
  local line name kind device omp tt extra model methods data base tag limit regimes mask folder stored cmp_tag rset mk root rc lp pc fp
  if [ "$DRY" != 1 ]; then   # one launcher at a time runs pre-flights; the next one finds the markers
    until mkdir "$PREFLIGHT_DIR/.lock" 2> /dev/null; do
      lp=$(cat "$PREFLIGHT_DIR/.lock/pid" 2> /dev/null)
      if [ -n "$lp" ] && ! kill -0 "$lp" 2> /dev/null; then
        say "removing the stale pre-flight lock of pid $lp"; rm -rf "$PREFLIGHT_DIR/.lock"; continue
      fi
      sleep 10
    done
    echo "$BASHPID" > "$PREFLIGHT_DIR/.lock/pid"
  fi
  while IFS= read -r line; do
    IFS='|' read -r name kind device omp tt extra model methods data base tag limit regimes mask folder stored cmp_tag rset <<< "$line"
    mk="$PREFLIGHT_DIR/$name.ok"
    fp=$(fp_of "$name")
    if [ "$DRY" != 1 ] && [ -f "$mk" ] && [ "$(cat "$mk")" = "$fp $line" ]; then
      say "PREFLIGHT $name already passed for this code ($fp)"; continue
    fi
    root="$PREFLIGHT_DIR/$name"
    # a pre-flight with k threads runs on the first min(k, WORKERS) cores of the queue
    # (no worker runs yet); the arrays depend on the thread count, not on the cores
    pc=$(IFS=,; echo "${CORE_LIST[*]:0:$(( omp < WORKERS ? omp : WORKERS ))}")
    build_cmd "$pc" "$device" "$omp" "$tt" "$extra" "$model" "$methods" "$data" "$base" \
      "$tag" "$limit" "$regimes" true - - - "$mask" "$root/a" "$rset"
    if [ "$DRY" = 1 ]; then
      echo "PREFLIGHT $name ($kind): $(show)"
      [ "$kind" = twice ] && echo "  run again into $root/b, then: $PLAN compare $root/a/$folder $root/b/$folder $cmp_tag \"$regimes\" $limit"
      [ "$kind" = stored ] && echo "  then: $PLAN compare $root/a/$folder $STORED/$stored $cmp_tag \"$regimes\" $limit"
      continue
    fi
    if [ "$device" = gpu ] && [ "${HAVE_GPU_JOBS:-0}" = 0 ]; then continue; fi
    wait_turn "$device"
    rm -rf "$root"; mkdir -p "$root"; rm -f "$mk"
    say "PREFLIGHT $name: $(show)"
    "${CMD[@]}" > "$LOGDIR/preflight_$name.log" 2>&1; rc=$?
    if [ $rc -eq 0 ] && [ "$kind" = twice ]; then
      build_cmd "$pc" "$device" "$omp" "$tt" "$extra" "$model" "$methods" "$data" "$base" \
        "$tag" "$limit" "$regimes" true - - - "$mask" "$root/b" "$rset"
      "${CMD[@]}" >> "$LOGDIR/preflight_$name.log" 2>&1; rc=$?
      [ $rc -eq 0 ] && { $PLAN compare "$root/a/$folder" "$root/b/$folder" "$cmp_tag" "$regimes" "$limit" >> "$LOGDIR/preflight_$name.log" 2>&1; rc=$?; }
    elif [ $rc -eq 0 ]; then
      $PLAN compare "$root/a/$folder" "$STORED/$stored" "$cmp_tag" "$regimes" "$limit" >> "$LOGDIR/preflight_$name.log" 2>&1; rc=$?
    fi
    if [ $rc -eq 0 ]; then
      echo "$fp $line" > "$mk"; say "PREFLIGHT $name passed: $(tail -1 "$LOGDIR/preflight_$name.log")"
    else
      say "PREFLIGHT $name FAILED (see $LOGDIR/preflight_$name.log): $(tail -1 "$LOGDIR/preflight_$name.log")"
    fi
  done < <($PLAN preflight)
  [ "$DRY" != 1 ] && rm -rf "$PREFLIGHT_DIR/.lock"
  return 0
}

pre_ok() {  # pre-flight requirement of a job: exact | pubdet | hgnn | legacy_<c>
  local n
  case "$1" in
    exact) for n in xdet xcross_policy xcross_rules xdet_rules; do [ -f "$PREFLIGHT_DIR/$n.ok" ] && grep -q "^$FINGERPRINT " "$PREFLIGHT_DIR/$n.ok" || return 1; done ;;
    pubdet) pre_ok exact || return 1; [ -f "$PREFLIGHT_DIR/pubdet.ok" ] && grep -q "^$FINGERPRINT " "$PREFLIGHT_DIR/pubdet.ok" || return 1 ;;
    hgnn) for n in hgnn_cross hgnn_cross_pub; do [ -f "$PREFLIGHT_DIR/$n.ok" ] && grep -q "^$HGNN_FINGERPRINT " "$PREFLIGHT_DIR/$n.ok" || return 1; done ;;
    *) [ -f "$PREFLIGHT_DIR/$1.ok" ] && grep -q "^$FINGERPRINT " "$PREFLIGHT_DIR/$1.ok" || return 1 ;;
  esac
  return 0
}

# ---- one pass of one worker over the job list
one_pass() {  # worker core -> sets RAN PENDING
  local w=$1 core=$2 idx=-1 line id group camp envn device omp tt extra model methods data base tag
  local limit regimes token blind bk sched oroot folder n ckpt log pre rset files have total f lk stage staged ok rc
  RAN=0; PENDING=0
  while IFS= read -r line; do
    idx=$((idx + 1))
    [ $((idx % PART_N)) -eq "$PART_I" ] || continue
    IFS='|' read -r id group camp envn device omp tt extra model methods data base tag limit regimes token blind bk sched oroot folder n ckpt log pre rset files <<< "$line"
    [[ "$id" =~ $ONLY ]] || continue
    have=0; total=0
    for f in $files; do total=$((total + 1)); [ -e "$f" ] && have=$((have + 1)); done
    [ $have -eq $total ] && { [ "$DRY" = 1 ] && echo "SKIP $id (finished)"; continue; }
    if [ $have -gt 0 ]; then
      [ "$w" = 0 ] && say "REFUSING $id: $have of $total files exist in place; inspect and remove them by hand"
      continue
    fi
    if [ "$ckpt" != "-" ] && { [ ! -f "$ckpt" ] || ! grep -q "done in" "$log" 2>/dev/null; }; then
      PENDING=$((PENDING + 1))
      [ "$DRY" = 1 ] || continue
    fi
    stage="$STAGE_ROOT/$id"; local ssched=-
    [ "$sched" != "-" ] && ssched="$stage/sched"
    local mask=legacy; [ "$envn" = exact ] && mask=exact
    build_cmd "$core" "$device" "$omp" "$tt" "$extra" "$model" "$methods" "$data" "$base" "$tag" \
      "$limit" "$regimes" "$token" "$blind" "$bk" "$ssched" "$mask" "$stage/out" "$rset"
    if [ "$DRY" = 1 ]; then
      echo "JOB $id [$group] ($total files, $n instances x $(wc -w <<< "$regimes") combinations x $(wc -w <<< "$methods") methods; pre-flight $pre): $(show)"
      continue
    fi
    pre_ok "$pre" || { [ "$w" = 0 ] && say "BLOCKED $id: pre-flight $pre has not passed"; continue; }
    lk="$LOCKDIR/$id"
    mkdir "$lk" 2> /dev/null || continue                   # another worker has it
    echo "$BASHPID $core $(date +%s)" > "$lk/pid"          # the worker's own pid, alive while it runs
    have=0; for f in $files; do [ -e "$f" ] && have=$((have + 1)); done
    if [ $have -gt 0 ]; then rm -rf "$lk"; continue; fi     # finished meanwhile
    if [ "$device" = gpu ]; then
      until mkdir "$LOCKDIR/GPU" 2> /dev/null; do
        gp=$(cat "$LOCKDIR/GPU/pid" 2> /dev/null)
        if [ -n "$gp" ] && ! kill -0 "$gp" 2> /dev/null; then
          say "removing the stale GPU lock of pid $gp"; rm -rf "$LOCKDIR/GPU"; continue
        fi
        sleep 30
      done
      echo "$BASHPID" > "$LOCKDIR/GPU/pid"
    fi
    wait_turn "$device"
    rm -rf "$stage"; mkdir -p "$stage/out"
    say "worker $w core $core RUN $id: $(show)"
    "${CMD[@]}" > "$LOGDIR/jobs/${id//\//_}.log" 2>&1; rc=$?
    [ "$device" = gpu ] && rm -rf "$LOCKDIR/GPU"
    ok=1
    if [ $rc -eq 0 ]; then
      for f in $files; do
        if [[ "$f" == "$oroot/"* ]]; then staged="$stage/out/${f#"$oroot"/}"; else staged="$stage/sched/${f#"$sched"/}"; fi
        [ -e "$staged" ] || { ok=0; say "FAILED $id: $staged was not written"; break; }
      done
    else ok=0; fi
    if [ $ok = 1 ]; then
      for f in $files; do
        if [[ "$f" == "$oroot/"* ]]; then staged="$stage/out/${f#"$oroot"/}"; else staged="$stage/sched/${f#"$sched"/}"; fi
        mkdir -p "$(dirname "$f")"
        mv -n "$staged" "$f"
        [ -e "$staged" ] && { ok=0; say "FAILED $id: $f appeared meanwhile; the staged copy stays in $stage"; }
      done
    fi
    if [ $ok = 1 ]; then
      rm -rf "$stage"; RAN=$((RAN + 1)); say "worker $w done $id ($total files)"
    else
      say "FAILED $id (exit $rc); see $LOGDIR/jobs/${id//\//_}.log and $stage"
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

# ---- launcher
JOBFILE=${JOBFILE:-$LOGDIR/jobs_$$.txt}   # one list per launcher; another launcher's settings cannot replace it
if [ "$DRY" = 1 ]; then
  JOBFILE=$(mktemp); $PLAN jobs > "$JOBFILE"
  run_preflights
  CORE_LIST=("<core>"); one_pass 0 "<core>"
  echo "jobs awaiting a checkpoint (listed above, would wait): $PENDING"
  $PLAN counts
  say "completeness check (read only):"; $PLAN check; rc=$?
  rm -f "$JOBFILE"; exit $rc
fi
mkdir -p "$LOGDIR/jobs" "$LOCKDIR" "$STAGE_ROOT" "$PREFLIGHT_DIR"
$PLAN jobs > "$JOBFILE" || { say "the planner failed"; exit 2; }
HAVE_GPU_JOBS=$(cut -d'|' -f5 "$JOBFILE" | grep -c '^gpu$')
# stale locks of workers that no longer run (a lock without a live pid, older than a minute)
for lk in "$LOCKDIR"/*; do
  [ -d "$lk" ] || continue
  pid=$(cut -d' ' -f1 "$lk/pid" 2> /dev/null); t=$(cut -d' ' -f3 "$lk/pid" 2> /dev/null)
  if { [ -n "$pid" ] && ! kill -0 "$pid" 2> /dev/null; } || { [ -z "$pid" ] && [ $(( $(date +%s) - $(stat -c %Y "$lk") )) -gt 60 ]; }; then
    say "removing the stale lock $lk"; rm -rf "$lk"
  fi
done
say "plan: $(wc -l < "$JOBFILE") jobs (groups: $EVAL_GROUPS$([ "$PUBLIC" = 1 ] && echo ' +public')$([ "$LEGACY45" = 1 ] && echo ' +legacy45')$([ "$HGNN" = 1 ] && echo ' +hgnn')); part $PART; $WORKERS workers on cores ${CORE_LIST[*]:0:$WORKERS}"
run_preflights
for ((w=0; w<WORKERS; w++)); do
  worker "$w" "${CORE_LIST[$w]}" >> "$LOGDIR/worker_${PART_I}_$w.log" 2>&1 &   # kept across restarts
done
wait
say "workers finished; completeness check against the stored legacy files:"
$PLAN check
rc=$?
rm -f "$JOBFILE"
say "queue finished (check exit $rc)"
exit $rc
