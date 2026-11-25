#!/bin/bash
# Secondary training arms under the exact mask (--first_mask exact
# --mask_ties exact), training seeds 301 to 303. Runs after the principal arms
# of scripts/run_exactmask_queue.sh.
#
# Purpose. Every arm that the main text compares with the released policy is
#   trained in the environment of the paper:
#     xmask-eq      released recipe with 6000 updates (equal experience per
#                   combination as the specialists): Section 8.1, two budgets
#     xmask-wthird  W specialist with 333 updates (the released policy's share
#                   of W experience): the W shortfall
#     xmask-sparse  released recipe, reward paid once at the end of the episode
#     xmask-dzero   released recipe, estimate without the minimum-setup term
#     xmask-dmax    released recipe, estimate with the maximum-setup term (not
#                   a lower bound): the three reward arms, Section 8.4
#     xmask-film    variant with descriptor input (71838 parameters)
#     xmask-dr      released recipe trained on randomized rule ranges
#     xmask-all8    released recipe trained on all eight combinations, 2000
#                   updates, the reference without a held-out combination. The
#                   validation score that keeps the checkpoint then averages
#                   eight combinations, and each combination gets 2000 x 20 / 8
#                   = 5000 episodes in expectation (6667 in the released recipe)
#     xmask-pair    released recipe with --pair_extra sw, which adds two
#                   channels per operation-machine pair, the exact setup the
#                   assignment would incur now and the delay an unavailability
#                   window adds to its start (pair feature width 10, 32914
#                   parameters, 128 more weights in the first layer of the actor)
#     xmask-active  released recipe with --mask_mode active, the Giffler-Thompson
#                   condition for active schedules: a pair stays legal while its
#                   processing start is strictly earlier than the earliest
#                   completion among the compatible pairs, so the policy may hold
#                   a machine idle for a later job. The non-delay set lies inside
#                   it, each step still schedules one operation, and episode
#                   length, reward and network are those of the released recipe
#                   (32786 parameters). With --first_mask exact the first decision
#                   is built by the same code as every later one, so the condition
#                   holds at every decision. Whether the policy produces shorter
#                   schedules when the mask is widened from non-delay to active
#                   schedules at the same training budget: the supplement section
#                   on the controls for the mask and one sentence of the main text
#                   (scripts/p50_actmask_analysis.py)
# Ideal result. Each arm reproduces the direction its legacy counterpart
#   showed (equal experience closes the W gap; the reward arms differ little
#   except, possibly, the sparse one; descriptor input brings no gain). A
#   different outcome is reported as found: these arms are controls, and the
#   text follows their result. xmask-all8 sets the policy's results on SW and
#   LSW against a policy that trained on them. xmask-pair shows whether the
#   changeover of each operation-machine pair, which the policy does not read,
#   changes its schedules. xmask-active is expected to reproduce the direction
#   of its legacy counterpart jointact: longer schedules than the policy under
#   every combination. If it gives shorter ones, the paper reports by how much
#   the non-delay mask costs the policy at this budget. The two outcomes lead
#   to different text, so the arm is trained.
# Contamination. New names only (prefix xmask-, checked against the analysis
#   scripts' patterns); the script refuses to overwrite a finished run and
#   starts no evaluation. One job on the GPU at a time: it waits while the
#   hold file exists, while the queue of the principal arms is alive, and
#   while another factory training process of this project runs (with
#   NO_GATE=1 only while the hold file exists).
# Data. Same sampler and validation set as the released recipe. Each arm is
#   built once on the CPU and must show the expected parameter count and a
#   resolved configuration equal to the stored legacy configuration
#   jointblind-v1-s303 except for name, seed, device, max_updates, the two mask
#   options (exact) and the arm's own option, whose value is checked (for
#   xmask-active: mask_mode active, network and reward unchanged).
#
# Gates: train_log/GPU_HOLD (in every mode), DEADLINE=<epoch s>, CORES (default
#   10-13; the launcher pins its own shell to them at start and stops if it
#   cannot), THREADS (default 4: OpenMP/MKL threads; a lane on one core is
#   started with CORES=<one core> THREADS=1), ONLY=<regex>, PROBE_ONLY=1, and
#   LANE=<i>/<n>   run only the tasks whose index in the full task list (arms in
#                  the order of ARMS, seeds 301-303 within each arm) is i modulo
#                  n (default 0/1); the index is taken before ONLY, so the n
#                  lanes are disjoint whatever ONLY is. With n > 1 the probe's
#                  model suffix and probe log carry the lane number
#                  (_xeprobe_<i>, xmask_extra_probe_<i>.log).
#   NO_GATE=1      before a run, wait only for the hold file, not for the queue
#                  of the principal arms or another factory training (default 0).
#   LIST_ONLY=1    print the tasks this invocation would train, one line
#                  "TASK <name>" each, after LANE, ONLY and the finished runs,
#                  and exit 0 before the probes; LIST_ONLY=all lists the
#                  finished runs too. Neither starts Python or writes a file;
#                  neither looks at the gates.
# Usage: setsid nohup scripts/run_exactmask_extra_queue.sh > train_log/exactmask_extra_queue.log 2>&1 &
# Lanes, one core each, one queue log per lane:
#   for i in 0 1 2; do LANE=$i/3 CORES=$((4 + i)) THREADS=1 NO_GATE=1 setsid nohup \
#       scripts/run_exactmask_extra_queue.sh > train_log/exactmask_extra_queue_lane$i.log 2>&1 & done
set -u
cd "$(dirname "$0")/.."
PY=$HOME/miniconda3/envs/fjsp/bin/python
THREADS=${THREADS:-4}
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset PACT_FIRST_MASK PACT_MASK_TOL
CORES=${CORES:-10-13}
DEADLINE=${DEADLINE:-0}
ONLY=${ONLY:-.}
PROBE_ONLY=${PROBE_ONLY:-0}
LANE=${LANE:-0/1}
NO_GATE=${NO_GATE:-0}
LIST_ONLY=${LIST_ONLY:-0}
HOLD=train_log/GPU_HOLD
MASK_OPTS="--first_mask exact --mask_ties exact"
SIX="N,L,S,W,LS,LW"
ALL8="N,L,S,W,LS,LW,SW,LSW"
DR_SHIFT='{"cross": [0.5, 2.0], "symmetric": false, "density_range": [0.05, 0.35]}'
LANE_SFX=""   # _<i> when the queue is split into lanes (probe file names)
say() { echo "[xmask-extra${LANE_SFX:+ lane $LANE} $(date '+%m-%d %H:%M')] $*"; }

if ! [[ "$LANE" =~ ^(0|[1-9][0-9]*)/[1-9][0-9]*$ ]] || [ "${LANE%/*}" -ge "${LANE#*/}" ]; then
  say "ABORT: LANE must be <i>/<n> with 0 <= i < n, got '$LANE'"; exit 1
fi
lane_i=${LANE%/*}; lane_n=${LANE#*/}
if [ "$lane_n" -gt 1 ]; then LANE_SFX="_${lane_i}"; fi
case "$NO_GATE" in 0|1) ;; *) say "ABORT: NO_GATE must be 0 or 1, got '$NO_GATE'"; exit 1 ;; esac
case "$LIST_ONLY" in 0|1|all) ;; *) say "ABORT: LIST_ONLY must be 0, 1 or all, got '$LIST_ONLY'"; exit 1 ;; esac
# The launcher shell itself, and so every probe, sleep and pgrep it starts,
# stays on CORES.
if ! taskset -pc "$CORES" $$ > /dev/null; then say "ABORT: could not pin the launcher to CORES=$CORES"; exit 1; fi
ncores=$(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc)
if [ "$THREADS" -gt "$ncores" ]; then say "WARNING: THREADS=$THREADS threads on the $ncores core(s) of CORES=$CORES"; fi

# arm | regimes | updates | parameters | minutes alone on the GPU | own key=value (JSON) | flags
ARMS=(
  "xmask-eq|$SIX|6000|32786|290|{}|"
  "xmask-sparse|$SIX|2000|32786|100|{\"reward_mode\": \"sparse\"}|--reward_mode sparse"
  "xmask-dzero|$SIX|2000|32786|100|{\"delta_mode\": \"zero\"}|--delta_mode zero"
  "xmask-dmax|$SIX|2000|32786|100|{\"delta_mode\": \"max\"}|--delta_mode max"
  "xmask-wthird|W|333|32786|20|{\"family_regimes\": \"W\"}|"
  "xmask-film|$SIX|2000|71838|130|{\"use_film\": true}|--use_film True"
  "xmask-dr|$SIX|2000|32786|100|{\"dr_shift\": \"__DR__\"}|--dr_shift __DR__"
  # appended last, so the lane of every earlier task keeps its index
  "xmask-all8|$ALL8|2000|32786|110|{\"family_regimes\": \"$ALL8\"}|"
  "xmask-pair|$SIX|2000|32914|100|{\"pair_extra\": \"sw\"}|--pair_extra sw"
  "xmask-active|$SIX|2000|32786|100|{\"mask_mode\": \"active\"}|--mask_mode active"
)

train_cmd() {  # $1 regimes $2 updates $3 seed $4 name $5 flags $6... extra (device)
  local regs=$1 upd=$2 seed=$3 name=$4 flags=$5; shift 5
  local -a cmd=($PY train_family.py --data_source FAMILY --n_j 10 --use_type_embedding True
                --validate_timestep 25 --family_regimes "$regs" --max_updates "$upd"
                --seed_train "$seed" $MASK_OPTS --model_suffix "$name" "$@")
  case "$flags" in
    *--use_film*) cmd+=(--use_film True) ;;
    *) cmd+=(--use_film False) ;;
  esac
  case "$flags" in
    *--reward_mode*) cmd+=(--reward_mode sparse) ;;
    *--delta_mode\ zero*) cmd+=(--delta_mode zero) ;;
    *--delta_mode\ max*) cmd+=(--delta_mode max) ;;
    *--dr_shift*) cmd+=(--dr_shift "$DR_SHIFT") ;;
    *--pair_extra*) cmd+=(--pair_extra sw) ;;
    *--mask_mode\ active*) cmd+=(--mask_mode active) ;;
  esac
  "${cmd[@]}"
}

probe() {  # one ARMS entry
  IFS='|' read -r arm regs upd params est own flags <<< "$1"
  local suf="_xeprobe${LANE_SFX}" plog="train_log/xmask_extra_probe${LANE_SFX}.log"
  timeout 900 taskset -c $CORES env PYTORCH_NVML_BASED_CUDA_CHECK=1 \
      bash -c "$(declare -f train_cmd); PY=$PY MASK_OPTS='$MASK_OPTS' DR_SHIFT='$DR_SHIFT' train_cmd '$regs' 1 301 '$suf' '$flags' --device cpu" > "$plog" 2>&1
  local np
  np=$(sed -nE 's/.*\[model\] params total=([0-9]+).*/\1/p' "$plog" | head -1)
  if [ "$np" != "$params" ]; then say "ABORT: ${arm} has $np parameters, want $params (see $plog)"; return 1; fi
  if ! grep -q "^\[train_family\] MASK: first_mask=exact mask_ties=exact" "$plog"; then
    say "ABORT: the probe of ${arm} did not train under the exact mask (see $plog)"; return 1
  fi
  ARM="$arm" OWN="$own" DR_SHIFT="$DR_SHIFT" WANT_REG="$regs" SUF="$suf" $PY - << 'PYEOF' || return 1
import json, os
ALLOWED = {'model_suffix', 'seed_train', 'device', 'max_updates', 'first_mask', 'mask_ties'}
INERT_NEW = {'dr_shift': ('',), 'delta_mode': (None,), 'mask_mode': (None, 'nondelay'),
             'pair_extra': (None, 'none'), 'reward_mode': ('dense',),
             'family_base': ('ppvc',), 'pub_vali_dir': ('',), 'pub_type_source': ('none',)}
arm = os.environ['ARM']
own = {k: (os.environ['DR_SHIFT'] if v == '__DR__' else v)
       for k, v in json.loads(os.environ['OWN']).items()}
ref = json.load(open('train_log/FAMILY/config_10x25+family+jointblind-v1-s303.json'))
probe = json.load(open(f"train_log/FAMILY/config_10x25+family+{os.environ['SUF']}.json"))
assert probe['family_regimes'] == os.environ['WANT_REG'], probe['family_regimes']
for k in ('first_mask', 'mask_ties'):
    if probe.get(k) != 'exact':
        raise SystemExit(f'ABORT: {arm}: the probe records {k}={probe.get(k)!r}, want exact')
for k, v in own.items():
    if probe.get(k) != v:
        raise SystemExit(f'ABORT: {arm}: the probe records {k}={probe.get(k)!r}, want {v!r}')
diffs = set()
for k in set(ref) | set(probe):
    rv, pv = ref.get(k, '<absent>'), probe.get(k, '<absent>')
    if rv == pv:
        continue
    if k in INERT_NEW and rv == '<absent>' and pv in INERT_NEW[k]:
        continue
    diffs.add(k)
bad = diffs - ALLOWED - set(own)
if bad:
    raise SystemExit(f"ABORT: {arm}: configuration drift against jointblind-v1-s303: "
                     + str({k: (ref.get(k), probe.get(k)) for k in sorted(bad)}))
print(f"[xmask-extra] {arm}: configuration OK, differs from the released recipe only in {sorted(diffs)}")
PYEOF
  # the probe's own files only (the trainer writes no other .txt), so a probe
  # never removes the files of another lane's probe
  rm -f "trained_network/FAMILY/10x25+family+${suf}.pth" \
        "train_log/FAMILY/config_10x25+family+${suf}.json" \
        "train_log/FAMILY/valiquality_10x25+family+${suf}.txt" \
        "train_log/FAMILY/reward_10x25+family+${suf}.txt" "$plog" 2>/dev/null
  return 0
}

# task: name | seed | index of its arm in ARMS (arms in order, seeds 301-303 within each)
TASKS=()
for ai in "${!ARMS[@]}"; do
  for seed in 301 302 303; do TASKS+=("${ARMS[$ai]%%|*}-s${seed}|${seed}|${ai}"); done
done

finished() {  # name -> true when its checkpoint exists and its log has the completion line
  [ -f "trained_network/FAMILY/10x25+family+$1.pth" ] && grep -q "done in" "train_log/FAMILY_$1.log" 2>/dev/null
}

# one GPU job at a time: true while the hold file exists (in every mode) and,
# unless NO_GATE=1, while the queue of the principal arms is alive (it must
# finish first) or any factory training run is on
must_wait() {
  [ -f "$HOLD" ] && return 0
  [ "$NO_GATE" = "1" ] && return 1
  pgrep -f "[r]un_exactmask_queue.sh" > /dev/null \
    || pgrep -f "[t]rain_family.py --data_source FAMILY --n_j 10 " > /dev/null
}

# The tasks of this lane: index i modulo n over the full list, then ONLY.
MINE=()
idx=-1
for t in "${TASKS[@]}"; do
  idx=$(( idx + 1 ))
  [ $(( idx % lane_n )) -eq "$lane_i" ] || continue
  [[ "${t%%|*}" =~ $ONLY ]] || continue
  MINE+=("$t")
done
if [ "$LIST_ONLY" != "0" ]; then
  for t in "${MINE[@]}"; do
    name=${t%%|*}
    if [ "$LIST_ONLY" = "1" ] && finished "$name"; then say "SKIP ${name} (finished)" >&2; continue; fi
    echo "TASK ${name}"
  done
  exit 0
fi

say "verifying the arms on the CPU (cores $CORES) ..."
for a in "${ARMS[@]}"; do probe "$a" || exit 1; done
say "all arms verified"
if [ "$PROBE_ONLY" = "1" ]; then say "PROBE_ONLY=1: not starting the queue"; exit 0; fi

for t in "${MINE[@]}"; do
  IFS='|' read -r name seed ai <<< "$t"
  IFS='|' read -r arm regs upd params est own flags <<< "${ARMS[$ai]}"
  log="train_log/FAMILY_${name}.log"
  pth="trained_network/FAMILY/10x25+family+${name}.pth"
  cfg="train_log/FAMILY/config_10x25+family+${name}.json"
  if finished "$name"; then say "SKIP ${name} (finished)"; continue; fi
  while must_wait; do sleep 120; done
  tmo=""
  if [ "$DEADLINE" -gt 0 ]; then
    now=$(date +%s); left=$(( DEADLINE - now ))
    if [ $left -lt $(( est * 60 )) ]; then say "STOP before ${name}: ${left}s to the deadline, the run needs about $(( est * 60 ))s"; break; fi
    tmo="timeout -s TERM ${left}"
  fi
  say "TRAIN ${name} (regimes ${regs}, ${upd} updates, seed ${seed}, exact mask, ${flags:-released recipe})"
  $tmo taskset -c $CORES bash -c "$(declare -f train_cmd); PY=$PY MASK_OPTS='$MASK_OPTS' DR_SHIFT='$DR_SHIFT' train_cmd '$regs' '$upd' '$seed' '$name' '$flags'" > "$log" 2>&1
  rc=$?
  np=$(sed -nE 's/.*\[model\] params total=([0-9]+).*/\1/p' "$log" | head -1)
  if [ $rc -ne 0 ] || ! grep -q "done in" "$log"; then
    say "FAILED ${name} (exit $rc); removing its partial files"
    rm -f "$pth" "$cfg"
    mv "$log" "${log}.failed" 2>/dev/null
    [ $rc -eq 124 ] && { say "the deadline ended the run; stopping"; break; }
    continue
  fi
  if [ "$np" != "$params" ]; then say "ABORT ${name}: $np parameters, want $params"; exit 1; fi
  if ! grep -q '"first_mask": "exact"' "$cfg" || ! grep -q '"mask_ties": "exact"' "$cfg"; then
    say "ABORT ${name}: its configuration does not record the exact mask"; exit 1
  fi
  say "trained ${name}: $(grep 'done in' "$log" | tail -1)"
done
say "queue finished"
