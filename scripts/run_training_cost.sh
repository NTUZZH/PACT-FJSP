#!/bin/bash
# Training time of one run with the GPU and its processor cores to itself, with evidence.
#
# Purpose. The paper's table of what a rule change costs (label tab:rule-change) reports
#   the training time of the policy and of the specialists. The campaign trained up to
#   four runs on the GPU at once, so its "done in X h" lines are contended and are never
#   reported. One invocation of this script repeats ONE finished campaign run alone on
#   the device and writes one evidence record, which scripts/p99_rule_change_cost.py
#   infer reads.
# Ideal result. The record is valid: no other process computed on the GPU or used the
#   reserved cores during the run, and the resolved configuration equals the campaign
#   run's except the name. An invalid record is kept as evidence and never used; the run
#   is repeated with REP=<k+1> once the cause is removed.
# Contamination. New names only, stem xcost- (xcost-v1-s<seed>-r<k>,
#   xcost-solo-<R>-s<seed>-r<k>); no glob or regular expression of the analysis, plan,
#   queue or registry code matches them (see the report). The log is written to
#   train_log/training_cost/, outside the train_log/FAMILY_*.log glob, and after the run
#   the trainer's checkpoint, configuration, validation and reward files are moved from
#   trained_network/FAMILY and train_log/FAMILY into train_log/training_cost/<name>/.
#   The script refuses to reuse a name and refuses while train_log/GPU_HOLD exists.
# Data. The command is the launch line of scripts/run_exactmask_queue.sh with the
#   regimes, updates and seed read from the campaign run's stored configuration. The
#   resolved configuration is compared with that file as soon as the trainer writes it
#   (a drift stops the trainer within one sample interval) and again at the end; the
#   log must report 32786 parameters and the exact mask.
#
# Inputs (environment variables):
#   ARM=released|specialist   the campaign arm to repeat (no default)
#   COMBO=<R>                 the combination of a specialist (N L S W LS LW SW LSW)
#   SEED=<s>                  training seed of the campaign run (default 301)
#   REP=<k>                   attempt number, part of the new name (default 1)
#   CORES=<list>              cores of the run, the launcher and its sampler (default 4-7)
#   THREADS=<n>               OpenMP/MKL threads of the trainer (default: size of CORES)
#   WINDOW=<s>                sample before the start (default 30)
#   INTERVAL=<s>              one sample per interval during the run (default 60)
#   DRY=1                     print the command, the names, the checks, the files and the
#                             preconditions; start nothing, write nothing
#   SAMPLE_ONLY=1             run only the sample before a start and print its verdict
#   ROOT_DIR, STUB_TRAINER, ALLOW_BUSY   tests only: a root other than the repository, a
#                             stand-in trainer, and a start on a busy machine; a record made
#                             with a stand-in trainer is always invalid
# The environment is the exact-mask environment; PACT_ENV, if set, must be exact.
#
# Usage:
#   ARM=released DRY=1 scripts/run_training_cost.sh
#   ARM=released setsid nohup scripts/run_training_cost.sh > train_log/training_cost_launch.log 2>&1 &
#   for R in N L S W LS LW SW LSW; do ARM=specialist COMBO=$R scripts/run_training_cost.sh; done
set -u
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniconda3/envs/fjsp/bin/python}
ARM=${ARM:-}
COMBO=${COMBO:-}
SEED=${SEED:-301}
REP=${REP:-1}
CORES=${CORES:-4-7}
WINDOW=${WINDOW:-30}
INTERVAL=${INTERVAL:-60}
DRY=${DRY:-0}
SAMPLE_ONLY=${SAMPLE_ONLY:-0}
say() { echo "[training-cost $(date '+%m-%d %H:%M')] $*"; }

case "$DRY" in 0|1) ;; *) say "ABORT: DRY must be 0 or 1, got '$DRY'"; exit 1 ;; esac
case "$SAMPLE_ONLY" in 0|1) ;; *) say "ABORT: SAMPLE_ONLY must be 0 or 1, got '$SAMPLE_ONLY'"; exit 1 ;; esac
case "${PACT_ENV:-exact}" in exact) ;; *) say "ABORT: PACT_ENV=$PACT_ENV; this measurement repeats exact-mask runs"; exit 1 ;; esac
export PACT_ENV=exact
unset PACT_FIRST_MASK PACT_MASK_TOL
# The launcher itself, and so the Python process it becomes and every sample command
# that process starts, stays on CORES.
if ! taskset -pc "$CORES" $$ > /dev/null; then say "ABORT: could not pin the launcher to CORES=$CORES"; exit 1; fi
ncores=$(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc)
THREADS=${THREADS:-$ncores}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
ARGS=(--seed "$SEED" --rep "$REP" --cores "$CORES" --threads "$THREADS"
      --window "$WINDOW" --interval "$INTERVAL" --py "$PY")
[ -n "${ROOT_DIR:-}" ] && ARGS+=(--root "$ROOT_DIR")            # tests only
[ -n "${STUB_TRAINER:-}" ] && ARGS+=(--stub_trainer "$STUB_TRAINER")   # tests only
[ "${ALLOW_BUSY:-0}" = "1" ] && ARGS+=(--allow_busy)                    # tests only, with a stub

if [ "$SAMPLE_ONLY" = "1" ]; then
  exec "$PY" scripts/training_cost_evidence.py sample "${ARGS[@]}"
fi
if [ -z "$ARM" ]; then say "ABORT: ARM must be released or specialist"; exit 1; fi
ARGS+=(--arm "$ARM" --combo "$COMBO")
if [ "$DRY" = "1" ]; then
  exec "$PY" scripts/training_cost_evidence.py plan "${ARGS[@]}"
fi
exec "$PY" scripts/training_cost_evidence.py run "${ARGS[@]}"
