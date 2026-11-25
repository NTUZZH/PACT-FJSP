#!/bin/bash
# Specialists that share the policy's training budget, under the exact mask
# (--first_mask exact --mask_ties exact): xshare-solo-<R>-s<seed>, one run per combination R
# and training seed of SEEDS (default 301-310, the ten seeds of the principal comparison; seeds
# 301 to 310 accepted), 250 updates each: 80 runs.
#
# Purpose. The principal comparison sets the policy (2000 updates on six combinations)
#   against eight specialists of 1000 updates each (xmask-solo-<R>-s<seed>,
#   scripts/run_exactmask_queue.sh), four times the policy's training in total. These runs
#   train the specialist's recipe for 250 updates, so that the eight specialists of one seed
#   together train for 8 x 250 = 2000 updates, the policy's number of updates and of training
#   episodes (20 per update for both). The policy is set against them on the test instances
#   of each combination at 10x25 (scripts/p127_budget_specialists.py): what one training
#   budget gives when it is spent on one policy or on eight specialists, and whether the lead
#   of the specialist under N holds when the specialist trains for a quarter of its updates.
#   That lead (+0.56% [+0.04, +1.07], inferior at both margins) is the ten-seed result of the
#   principal comparison; with the first five seeds alone N is equivalent. The comparison is
#   therefore the principal comparison repeated on the same ten training seeds, 301 to 310.
#   The numbers land in Section 6.1 (one or two sentences), in the section on the cost of a
#   rule change, in the decision table and in a supplement table. The comparison is not
#   planned in the statistical protocol, and the text says so.
# Ideal result. The policy's schedules are shorter than or equal to those of the
#   budget-sharing specialists on every combination: one training budget then serves the
#   plant better as one policy. If a budget-sharing specialist produces shorter schedules on
#   some combination, the text names that combination and the policy's difference to it, and
#   the decision table keeps the specialist for it. Both outcomes are reported as found.
# Contamination. New names only, prefix xshare- (no glob or regular expression of the
#   analysis, plan, evaluation, export and figure scripts matches it); the queue refuses to
#   overwrite a finished run and starts no evaluation, so no array enters test_results/. The
#   probes write the model suffix _xbprobe and the log train_log/xshare_probe.log (_xbprobe_<i>
#   and xshare_probe_<i>.log in lane i), names that no other queue writes or removes, and
#   remove their own files only. PACT_FIRST_MASK and PACT_MASK_TOL are unset and the two mask
#   options are passed. No wall-clock of these runs is a measurement.
# Data. Same sampler, validation set, recipe and seed as the 1000-update specialist of the
#   same combination (for the seeds 306 to 310 the specialists of the second five seeds,
#   scripts/seed_extension.py); only --max_updates differs. Before any training, a CPU probe
#   of one update per combination of this invocation must show (i) 32786 parameters, (ii) the exact
#   mask on the trainer's mask line and in the probe's configuration, (iii) a resolved
#   configuration equal to the stored configuration of xmask-solo-<R>-s301 except
#   model_suffix, seed_train, device and max_updates, and (iv) the MWKR norms line of the log
#   train_log/FAMILY_xmask-solo-<R>-s301.log. After each run: the completion line, 32786
#   parameters, the exact mask recorded, and a configuration equal to that of
#   xmask-solo-<R>-s<seed> except model_suffix and max_updates (250 against 1000)
#   (scripts/budgetspec_runs.py probe and trained). A failed check aborts the queue; a run
#   that ends without its completion line is removed and named.
#
# Tasks, names xshare-solo-<R>-s<seed>, in this order: per seed of SEEDS, the combinations N, L,
#   S, W, LS, LW, SW, LSW; 250 updates each, about 17 minutes with the GPU to itself (a quarter
#   of the 1000-update specialist; the estimate that DEADLINE uses) and about 32 minutes in one
#   of seven lanes that share the GPU (a quarter of the median 2.11 h of the 1000-update
#   specialists of the seeds 306 to 310, trained in seven lanes). Training runs on the GPU
#   (CUDA_VISIBLE_DEVICES=0); the probes run on the CPU.
# Lanes. LANE=<i>/7 takes the tasks whose index is i modulo 7: of the 80 runs, lanes 0 to 2
#   hold 12 and lanes 3 to 6 hold 11, about 6.5 and 6 hours per lane with the eight probes
#   (about 42 lane-hours in all).
#
# Options: SEEDS (default 301-310), HOLD_FILE (default train_log/GPU_HOLD), DEADLINE, CORES
#   (required outside LIST_ONLY), THREADS, LANE=<i>/<n>, ONLY, PROBE_ONLY=1 and LIST_ONLY=1 or
#   all, with the meaning stated in the header of scripts/run_longspec_queue.sh.
#
# The code is that of scripts/run_longspec_queue.sh: this file runs it with the variant xshare.
#
# Usage, seven lanes, one free core each (lane i on the i-th core of the list), a hold file of
#   their own, one queue log per lane, the GPU shared by the lanes:
#   i=0; for c in <seven free cores>; do LANE=$i/7 CORES=$c THREADS=1 \
#       HOLD_FILE=train_log/BUDGETSPEC_HOLD setsid nohup bash \
#       scripts/run_budgetspec_queue.sh > train_log/budgetspec_queue_lane$i.log 2>&1 &
#       i=$((i + 1)); done
# The probes alone, on one core:  CORES=<core> THREADS=1 PROBE_ONLY=1 bash scripts/run_budgetspec_queue.sh
if [ $# -ne 0 ]; then
  echo "[budgetspec] ABORT: no argument is taken (options are environment variables), got '$*'"; exit 1
fi
exec bash "$(dirname "$0")/run_longspec_queue.sh" xshare
