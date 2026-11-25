#!/usr/bin/env bash
# The analysis chain of the manuscript: from the finished evaluations to the
# macro files that manuscript/ reads, in one ordered and repeatable run.
#
# Every step is an analysis of stored arrays and records: no training, no
# evaluation, no solver and no timed measurement is started here. One step rolls a policy
# out: example_schedule rolls the selected run out greedily on one test instance under two
# combinations (CPU, one torch thread) to record the schedules it draws, and checks both
# makespans against the stored arrays. The steps read the
# exact-mask environment with five training seeds (scripts/arm_registry.py,
# PACT_ENV=exact, PACT_SEEDS=301-305; the step environments also reads the prior
# implementation on the seeds both environments hold) and apply the statistical protocol of
# manuscript/sections_new/s6_setup.tex (Section 6.4) and
# manuscript/supp_new/sn07_statistics.tex. Steps, in this order:
#
#   family                   p11 -> p27 -> p88 --n30 --require-sections all: policy against
#                            the specialists and the single-rule control (10x25), the best
#                            dispatching rule (10x25 and the production batches), and the
#                            sections F (rule-change pairs), G (the specialist of LSW on all
#                            eight combinations) and H (shops of 20 and 30 modules, cells of
#                            15 stations); p11 and p27 only write the files p88 reads back
#                            and checks; brings $PACT_OUT/paper/macros_family.tex to
#                            manuscript/macros_family.tex
#   seed_extension           p110: the principal comparison of step family (section A of p88,
#                            the policy against the specialist of each combination at 10x25)
#                            with ten training seeds, 301 to 310, when the extension rule of
#                            scripts/seed_extension.py applies in the five-seed record that step
#                            family writes; skipped when the rule does not apply or a run of the
#                            second five seeds is not trained and evaluated; brings
#                            $PACT_OUT/manuscript/macros_seedext.tex to manuscript/macros_seedext.tex
#   instance_counts          p125: the principal comparison of step family in counts of instances
#                            (policy shorter, equal, longer) and in hours, with the interval of the
#                            percent difference and the largest loss and gain; ten seeds, 301 to
#                            310, by the rule of step seed_extension (the ten-seed record of p110
#                            then read back), five otherwise; the five-seed record of step family
#                            is reproduced first; brings $PACT_OUT/manuscript/macros_counts.tex to
#                            manuscript/macros_counts.tex
#   long_specialists         p117 --pact-seeds $LONGSPEC_SEEDS (default 301-305): the specialists
#                            trained for the policy's 2000 updates against the specialists of 1000
#                            updates, and the policy against both, on each combination at 10x25,
#                            with the training curves of both runs; skipped while a run of 2000
#                            updates of these seeds is not trained and evaluated; brings
#                            $PACT_OUT/manuscript/macros_longspec.tex to manuscript/macros_longspec.tex
#   budget_specialists       p127 --seeds $BUDGETSPEC_SEEDS (default 301-310, the ten seeds of
#                            the principal comparison; p127 does not read PACT_SEEDS): the policy
#                            against eight specialists of 250 updates each, which share the
#                            policy's training budget, on each combination at 10x25 (verdicts,
#                            joint verdicts, pooled rows, the combined verdict with Holm over the
#                            eight), beside the specialists of 1000 updates, and the training
#                            effort of each side; skipped while a run of 250 updates of these seeds
#                            is not trained and evaluated; brings
#                            $PACT_OUT/manuscript/macros_budgetspec.tex to manuscript/macros_budgetspec.tex
#   heldout_splits          p122 --pact-seeds $HELDOUT_SEEDS (default 301-303): the two policies
#                            trained with other held-out pairs (LS and LSW, LW and LSW) against the
#                            specialists on all eight combinations at 10x25, beside the policy;
#                            skipped while a split run of these seeds is not trained and evaluated;
#                            brings $PACT_OUT/manuscript/macros_heldout.tex to
#                            manuscript/macros_heldout.tex
#   per_floor                p118: the per-floor batch of 27 modules (test set 27x25) by the
#                            procedure of section H of p88, beside 20x25 and 30x25 read from the
#                            record of step family and reproduced from the arrays; skipped while
#                            the evaluation of 27x25 is incomplete; brings
#                            $PACT_OUT/manuscript/macros_perfloor.tex to manuscript/macros_perfloor.tex
#   production_specialists   p128: the policy against the specialist of each combination and
#                            against the best dispatching rule at the production batches
#                            (50x25n30, 80x25n30) by the procedure of section H of p88, the
#                            specialists read from <set>_xmask_spec, the default level of p102
#                            and p103 checked against the arrays and records, and the pooled rows
#                            of 10, 20, 27, 30, 50 and 80 modules side by side (records of steps
#                            family and per_floor, reproduced); skipped while the evaluation of
#                            the specialists is incomplete; brings
#                            $PACT_OUT/manuscript/macros_prodspec.tex to manuscript/macros_prodspec.tex
#   ood                      p33: rule magnitudes outside the training ranges at 10x25, the policy
#                            against the per-instance best of FIFO, MOR, SPT and MWKR in
#                            distribution, with larger asymmetric setups (10x25_oodsetup) and with
#                            denser unavailability windows (10x25_oodwindow), by the procedure of
#                            FamRule<R> of step family (hierarchical interval, Holm over the eight
#                            combinations in distribution and over the four a shift reaches,
#                            combined verdict); the values in distribution are checked against the
#                            record of step family, the control on the combinations a shift cannot
#                            reach asserted bit for bit; brings $PACT_OUT/paper/macros_ood.tex to
#                            manuscript/macros_ood.tex
#   asym_setups              p119: asymmetric setups at the trained magnitude (10x25_asymsetup)
#                            beside the larger asymmetric setups of step ood, by p33's procedure
#                            and checked against its record, and the policy against the
#                            specialists on the three sets; skipped while the evaluation of
#                            10x25_asymsetup is incomplete; brings
#                            $PACT_OUT/manuscript/macros_asym.tex to manuscript/macros_asym.tex
#   wrong_descriptor         p121: the policy's schedules under a wrong constraint descriptor (two
#                            pairs of the combination in force and the combination configured),
#                            from the stored rollout records: makespan change, schedules the
#                            validator rejects against the true and the configured descriptor,
#                            and the configuration checks on the wrong and on six malformed
#                            descriptors; the rollouts (p121 rollout) stay outside this chain;
#                            brings $PACT_OUT/manuscript/macros_wrongdesc.tex to
#                            manuscript/macros_wrongdesc.tex
#   example_schedule         p123: the two example schedules of the selected run (the first test
#                            instance of 10x25 under N and LSW), rolled out greedily, checked and
#                            drawn as one Gantt chart; writes
#                            manuscript/sections_new/figures/fq0_example_schedule.pdf and .png and
#                            the two makespans to manuscript/macros_example.tex in place, and
#                            its record to out/exact/results/example_schedule.json
#                            whatever PACT_OUT is, so the step runs only with PACT_OUT=out/exact
#   seed_spread             p112: the spread of the mean makespan across training seeds at
#                            10x25, the policy and the specialist of each combination, with the
#                            paired test of equal variances; the seeds 306 to 310 and all ten
#                            seeds once the 45 runs of the second five are complete; brings
#                            $PACT_OUT/manuscript/macros_seedspread.tex to
#                            manuscript/macros_seedspread.tex
#   convergence              p116: whether training had stopped improving when it ended, from the
#                            validation lines of the training logs of the policy, the single-rule
#                            control, the eight specialists and the eight specialists trained for
#                            the policy's 2000 updates (seeds 301 to 305; skipped until all 40
#                            runs of 2000 updates have finished): the mean score of the last
#                            fifth of training against the fifth before, the share of training at
#                            which the kept checkpoint's score was first reached, the gain of the
#                            second half, and for the runs of 2000 updates the change of the mean
#                            score after update 1000; then make_figures.py --only
#                            fs_convergence; brings $PACT_OUT/manuscript/macros_convergence.tex to
#                            manuscript/macros_convergence.tex
#   release_dates          p69 analyze: release dates at 10x25, the policy against the best
#                            dispatching rule with and without release dates, from the unit files
#                            of p69 evaluate (every policy seed and the four dispatching rules);
#                            brings $PACT_OUT/manuscript/macros_release.tex to
#                            manuscript/macros_release.tex and the table to manuscript/tables/
#   cross_eval               p100: every specialist under every combination against the
#                            policy, the 8 x 8 matrix at 10x25 (groups principal and cross of
#                            the evaluation plan); brings $PACT_OUT/manuscript/macros_crosseval.tex
#                            to manuscript/macros_crosseval.tex
#   public                   p89: the benchmark from public instances (writes
#                            manuscript/macros_public.tex in place); its section 9, the gap
#                            to the references of pub_SD2_test_10x5+mix (records of p91),
#                            once all 800 references are valid
#   pubact                   p108 infer: the public-family policy trained under the active
#                            mask (pubx-act, training seeds 301 to 303) against the policy on
#                            pub_SD2_test_10x5+mix (writes manuscript/macros_pubact.tex in
#                            place); skipped while no file of the arm exists; reads seeds 301
#                            to 305 once a file of pubx-act-s304 or -s305 exists (the policy
#                            runs of the family under the active mask), seeds 301 to 303 before
#   public_act               p89 --family active: the benchmark from public instances with every
#                            recipe trained under the active mask (pubx-act, pubx-ctlact,
#                            pubx-soloact-<R>, pubx-eqact; seeds 301 to 305) and the dispatching
#                            rules rolled out under it (p109), the same sections as step
#                            public, and section 10, each recipe and the best rule under the
#                            active mask against the non-delay mask (writes
#                            manuscript/macros_pubfam_act.tex in place); skipped while no file
#                            of the family exists beyond the three runs of step pubact
#   rulecmp_public_act       p93 --family public --mask active, on the headline set and with
#                            --set on each of the five other public test sets: the three forms
#                            of the rule comparison with the policy, the control, the
#                            specialists and the dispatching rules under the active mask (writes
#                            manuscript/macros_rulecmp_public_act.tex and
#                            manuscript/macros_rulecmp_public_act_<tag>.tex in place); skipped
#                            while no output of a job of the campaign exists
#   public_seed_extension    p115: the policy against the specialists of step public_act on
#                            pub_SD2_test_10x5+mix (section 1 of p89 --family active) with ten
#                            training seeds, 301 to 310, an addition after the five-seed result
#                            was known (scripts/pub_seed_extension.py); the five-seed record of
#                            step public_act stays the reading of record and is reproduced
#                            first; skipped while a run of the second five seeds is not trained
#                            and evaluated; brings $PACT_OUT/manuscript/macros_pubseedext.tex to
#                            manuscript/macros_pubseedext.tex
#   rulecmp_public          p93 --family public: three forms of the rule comparison, SD2 10x5
#   rulecmp_public_<tag>     p93 --family public --set <set>, tags sdtwenty, brand, edata,
#                            rdata, vdata (released policy only)
#   rulecmp_factory          p93 --family factory
#   solver_selection         p98: the selection of the CP-SAT configuration on the records of
#                            the selection runs (eight candidates, instances 000 to 002 of N,
#                            S, SW and LSW at 50 and 80 modules; writes
#                            manuscript/macros_solver_selection.tex in place)
#   memetic_selection        p106: the selection of the configuration of the memetic algorithm
#                            on the records of its selection runs (three candidates, the same
#                            24 cells; writes manuscript/macros_memsel.tex in place)
#   timed                    p94: solver and memetic algorithm with a time budget against
#                            the policy (campaign tokens as designed: P94_CAMPAIGN_PROD,
#                            P94_CAMPAIGN_TRAIN; nine instances per combination at the
#                            production batches, CP-SAT on ten at 10x25)
#   timed_instance_sets      p94 --origin start --budgets 60 300 600 twice, --tag startnine on
#                            P94_CAMPAIGN_STARTNINE and --tag startall on P94_CAMPAIGN_STARTALL,
#                            then p130: the policy against the memetic algorithm and CP-SAT
#                            started from the rule schedule (memetic_rule@600, cpsat_rule@600) at
#                            60, 300 and 600 s of the start clock, on the nine instances of the
#                            design (003 to 011 of 50x25n30 and 80x25n30) beside all 27 evidence
#                            instances (003 to 029); the two p94 runs write
#                            results/timed_inference_<tag>.json and .md and
#                            results/timed_curves_<tag>.csv and no macro file; brings
#                            $PACT_OUT/manuscript/macros_timedsets.tex to
#                            manuscript/macros_timedsets.tex
#   memetic_parallel        p124 --slots $TIMED_SLOTS (default 16-23 8-15, the two core sets the
#                            records of the campaign come from): eight independent runs of the
#                            memetic algorithm (seeds 1 to 8 of memetic_rule@600, instances 003 to
#                            011 of 50x25n30 and 80x25n30) read as one method on eight cores,
#                            against the policy and CP-SAT; skipped while a cell has fewer than
#                            eight valid runs (p124 --check); once the eight runs started from
#                            the policy's schedule (memetic_policy@600, the same cells) are
#                            complete, the same run adds their comparisons, the decision table
#                            with both memetic columns as the best of eight (its four other
#                            columns checked against results/timed_inference.json of step timed)
#                            and $PACT_OUT/results/memetic_parallel_curves.csv; brings
#                            $PACT_OUT/manuscript/macros_memetic_parallel.tex to
#                            manuscript/macros_memetic_parallel.tex
#   memetic_one_minute       p137 --ref-slots $TIMED_SLOTS --pol-slots $TIMED_SLOTS, after
#                            memetic_parallel: two sets of eight runs of the memetic algorithm
#                            with a budget of 60 s on the cells of memetic_parallel (seeds 1 to 8
#                            of memetic_rule@60, core set 16-23, and of memetic_policy@60 started
#                            from the policy's schedule of the selected run, core sets
#                            $TIMED_SLOTS) read one minute after the request beside the 600-s
#                            runs from each start, the policy's greedy and sampled schedules and
#                            CP-SAT, and the one-minute rows of the decision table on eight cores
#                            of $PACT_OUT/results/memetic_parallel.json with the 60-s values in
#                            both memetic columns; skipped while a cell of either set lacks a
#                            valid 60-s run (p137 --check); brings
#                            $PACT_OUT/manuscript/macros_memetic_minute.tex to
#                            manuscript/macros_memetic_minute.tex
#   memetic_five_minute      p137 --budget 300 --slots $MFIVE_SLOTS --pol-slots $MFIVE_POL_SLOTS
#                            --ref-slots $TIMED_SLOTS, after memetic_parallel: two sets of eight
#                            runs of the memetic algorithm with a budget of 300 s on the cells of
#                            memetic_parallel (seeds 1 to 8 of memetic_rule@300, core sets
#                            $MFIVE_SLOTS, and of memetic_policy@300 started from the policy's
#                            schedule of the selected run, core sets $MFIVE_POL_SLOTS) read five
#                            minutes after the request beside the 600-s runs from each start,
#                            the policy's greedy and sampled schedules and CP-SAT, and the
#                            five-minute rows of the decision table on eight cores of
#                            $PACT_OUT/results/memetic_parallel.json with the 300-s values in
#                            both memetic columns; skipped while a cell of either set lacks a
#                            valid 300-s run (p137 --budget 300 --check); brings
#                            $PACT_OUT/manuscript/macros_memetic_five.tex to
#                            manuscript/macros_memetic_five.tex
#   decision_rows            p138 --ref-slots $TIMED_SLOTS --pol-slots $TIMED_SLOTS --five-slots
#                            $MFIVE_SLOTS --five-pol-slots $MFIVE_POL_SLOTS, after
#                            memetic_five_minute: per row of the decision table (12 rows at 50
#                            and 80 modules from the records of memetic_parallel,
#                            memetic_one_minute and memetic_five_minute, 8 rows at the training
#                            size from
#                            results/timed_inference.json of step timed) the paired difference
#                            of the lowest column minus the second lowest and of the best
#                            column that uses the policy minus the best that does not, with
#                            instance-level 95% intervals, on the records and cells that give
#                            the printed values; skipped while an input or a record is missing
#                            or differs (p138 --check); brings
#                            $PACT_OUT/manuscript/macros_decision_rows.tex to
#                            manuscript/macros_decision_rows.tex
#   instance_bound         p113: the makespan estimate at the empty schedule against the
#                            shortest schedule of the 600 s records of CP-SAT and the memetic
#                            algorithm at 10x25 and against the selected policy run; brings
#                            $PACT_OUT/manuscript/macros_ibound.tex to manuscript/macros_ibound.tex
#   lower_bound_reference    p120 with the campaign as launched (P120_CAMPAIGN_PROD,
#                            P120_CAMPAIGN_TRAIN) and P120_CAMPAIGN_SOLVER_TRAIN, the CP-SAT cells
#                            at 10x25: a lower-bound reference per test instance (the estimate at
#                            the empty schedule, the machine-load bound, the stored bounds of
#                            CP-SAT records) and the gap of every method to it at 10, 20, 30, 50
#                            and 80 modules; brings $PACT_OUT/manuscript/macros_lbref.tex to
#                            manuscript/macros_lbref.tex and, when p120 reports it written, the
#                            table $PACT_OUT/manuscript/tables/lbref_table.tex to manuscript/tables/
#   randomized_rules         p135 infer: randomized dispatching rules on eight cores within the
#                            time budget (a greedy randomized construction over FIFO, MOR, SPT
#                            and MWKR, 144 cells: instances 003 to 011 of 50x25n30 and 80x25n30,
#                            eight combinations) against the policy's greedy and sampled
#                            schedules, the best dispatching rule, the best of the eight memetic
#                            runs and CP-SAT started from the rule schedule, read through p124's
#                            readers; skipped while a cell lacks a valid record or a stored input
#                            is missing (p135 check, and the stored inputs as infer reads them);
#                            brings $PACT_OUT/manuscript/macros_randomized_rules.tex to
#                            manuscript/macros_randomized_rules.tex
#   memetic_public           p136 infer: the memetic algorithm on the ten Brandimarte instances
#                            without rules (regime N of the public family), three seeds, the gap
#                            to the best-known makespans at 60 s and 600 s of search; skipped
#                            while one of the 30 runs lacks a valid record; brings
#                            $PACT_OUT/manuscript/macros_memetic_public.tex to
#                            manuscript/macros_memetic_public.tex
#   secondary                p95 --require all: supporting training arms and the published
#                            learned scheduler
#   margin_and_pairs         p140: the verdict rule of the principal comparison applied to
#                            comparators known to differ from the specialist at 10x25 (the
#                            per-instance best of the four dispatching rules with ten seeds,
#                            beside the policy against the specialist and against the best rule
#                            read from the records of steps seed_extension, family and
#                            production_specialists; the sparse-reward arm of step secondary,
#                            the specialists of 250 updates of step budget_specialists, and each
#                            specialist under its seven other combinations), and the policy
#                            against two specialists that serve all eight combinations (seeds 301
#                            to 305, the arrays p100 reads); brings
#                            $PACT_OUT/manuscript/macros_margin_pairs.tex to
#                            manuscript/macros_margin_pairs.tex
#   environments             p96: the environment of the paper against the implementation
#                            released with the prior study, training seeds 301 to 303, 10x25
#   sensitivity              p87 infer: sensitivity of the factory benchmark, 22 levels one
#                            parameter at a time (policy and single-rule control against the
#                            specialists, policy against the best dispatching rule; the share
#                            of single-pair decisions and the setup share of machine capacity
#                            when the results carry the counts of evaluate --trace_counts);
#                            the four rules act on the pairs the mask admits
#                            (common_utils.masked_rule_action, as eval_family.py --rule_set
#                            masked), and infer refuses a rule file of another rule set;
#                            writes the supplement table manuscript/tables/sensitivity_table.tex
#   aware_rules              p72 --summary_only: the three further dispatching rules (ECT, SSPT,
#                            LLF) at 10, 20 and 30 modules and on the 30-instance production
#                            sets 50x25n30 and 80x25n30, then its inference step (the summary
#                            goes to $PACT_OUT; the inference step writes manuscript/ and results/)
#   best_of_seven            p126: the policy against the per-instance best of the seven
#                            dispatching rules beside the best of the four classical rules, at the
#                            sizes of step aware_rules, and the share of instances each rule wins;
#                            the summary of p72 at 10x25 is reproduced first; brings
#                            $PACT_OUT/manuscript/macros_bestseven.tex to manuscript/macros_bestseven.tex
#   mask_controls            p101 infer: random choice among the legal pairs, legal pairs per
#                            decision, window coverage and shop load, with the SPT and the
#                            policy trace; the ten schedules of the policy against the ten
#                            random rollouts when their cells exist (MASKCTL_TEN)
#   reference_replay         p105 infer: the reference schedules of the solver and the memetic
#                            algorithm replayed under the non-delay mask, an upper bound on what
#                            the mask costs (writes manuscript/macros_refreplay.tex in place)
#   mask_rules               p107 infer: three admissibility rules (nondelay, block, active)
#                            for the classical rules, the random control and the replay of the
#                            references (writes manuscript/macros_maskrule.tex in place)
#   mask_choice_validation   p141: the comparison of step mask_rules repeated on the 20
#                            validation instances of each benchmark (the best of four
#                            dispatching rules under the active mask and under block starts
#                            against the non-delay mask, the instances rebuilt as
#                            train_family.py builds them and checked against the MWKR norms of
#                            the training logs), beside the test-instance values read from the
#                            record of p107; brings $PACT_OUT/manuscript/macros_mask_validation.tex
#                            to manuscript/macros_mask_validation.tex
#   actmask                  p50 --require all: the policy trained with the active-schedule
#                            mask (xmask-active, training seeds 301 to 303) against the policy
#                            on the seeds both have, per combination and pooled, at every size
#                            the evaluation plan gives the arm (10x25); brings
#                            $PACT_OUT/manuscript/macros_actmask.tex to manuscript/macros_actmask.tex
#   scale_sensitivity        p102 infer, every arm: windows over the whole schedule at 50 and
#                            80 modules against the stored windows
#   five_classes             p103 infer, every arm: a fifth setup class at 10, 50 and 80
#                            modules against four classes (FIVE_DEFAULT_POLICY)
#   project_mix              p129 infer, every arm: single-shell project mixes (reinforced
#                            concrete alone, steel alone) at 10, 50 and 80 modules against the
#                            mixed test sets
#   figure_limits            make_figures.py --only fq4_limits --fq4-limits-variant A in
#                            full mode: the policy against the specialists and the best
#                            dispatching rule at the six shop sizes and the 22 levels of the
#                            sensitivity study (needs family, per_floor, production_specialists
#                            and sensitivity; the figure refuses while a size or a level is
#                            missing)
#   rc_trained               p131 infer, after project_mix: the policy trained on
#                            reinforced-concrete instances alone (xmask-rc, training seeds 301
#                            to 303) against the best of the four and of the seven dispatching
#                            rules, the policy and the specialists, on the reinforced-concrete
#                            sets and the mixed test sets at 10, 50 and 80 modules, with the
#                            check of the evaluation path and of the training instances against
#                            the test sets; skipped while a run or a result file is missing
#                            (p131 check); writes manuscript/macros_rctrained.tex and its record
#                            under results/ in place
#   divisor                  p104 infer: the selected run's greedy schedule of every 10x25 test
#                            instance alone, twice in one batch and with a companion at the
#                            largest processing time of the test set
#   deploy                   p75 --from_json: deployment-cost macros from the record of the
#                            manual measurement M1; brings $PACT_OUT/paper/macros_deploy.tex
#                            to manuscript/macros_deploy.tex
#   rule_change_cost         p99 count, then p99 infer --require all: what a rule change costs,
#                            one policy against one specialist per combination, from the records
#                            of the manual measurements M3 and M4; brings
#                            $PACT_OUT/manuscript/macros_rulechange_cost.tex to
#                            manuscript/macros_rulechange_cost.tex
#   model_facts              p111: parameter counts, the checkpoint size of the selected run, the
#                            median operation duration and the check of the design constants
#                            typed by hand in the macro files; brings
#                            $PACT_OUT/manuscript/macros_modelfacts.tex to
#                            manuscript/macros_modelfacts.tex
#   figures                  make_figures.py --only fq1 (needs family and public_act);
#                            panel (a) reads the ten-seed record of step seed_extension when
#                            $PACT_OUT/results/seed_extension.json exists, the five-seed
#                            record of step family otherwise; then --only fq2_reuse from the
#                            record of step cross_eval ($PACT_OUT/results/cross_eval.json and
#                            $PACT_OUT/manuscript/macros_crosseval.tex), and --only fq2_heldout
#                            (full mode) from the records of step heldout_splits
#                            ($PACT_OUT/results/heldout_splits.json and
#                            $PACT_OUT/manuscript/macros_heldout.tex) and of step secondary
#                            (results/secondary_inference.json and manuscript/macros_secondary.tex
#                            at the repository root, where p95 writes them), and --only
#                            fq5_rule_change from the records of step rule_change_cost
#                            ($PACT_OUT/results/rule_change_cost.json and
#                            $PACT_OUT/manuscript/macros_rulechange_cost.tex) and of step deploy
#                            ($DEPLOY_JSON and $PACT_OUT/paper/macros_deploy.tex); each of the three is
#                            left out, with one line, while a record it reads is missing or while
#                            its step was selected in the same run and did not finish
#   figure_timed            make_figures.py --only fq3: the time-budget figure (needs
#                            timed; the policy's time mark is read from DEPLOY_JSON when
#                            that record exists, and left out otherwise). When
#                            $PACT_OUT/results/memetic_parallel_curves.csv exists (step
#                            memetic_parallel with both sets of eight runs complete), the
#                            memetic curves are the best of eight runs
#                            (--memetic-parallel $PACT_OUT/results/memetic_parallel.json);
#                            otherwise they are the single run. With the best of eight and
#                            $PACT_OUT/results/memetic_one_minute.json (step memetic_one_minute),
#                            variant A also shows the memetic algorithm with a one-minute budget
#                            at 60 s (--memetic-minute), and with
#                            $PACT_OUT/results/memetic_five_minute.json as well (step
#                            memetic_five_minute) the five-minute budget at 300 s
#                            (--memetic-five). Last step
#
# Each step states its precondition and is skipped, with the reason, when the
# precondition does not hold. A step whose dependency (named in deps_of) was
# selected in the same run and did not finish is skipped too. A failed step stops
# the run and prints the end of its log. A step that ends without writing a declared
# output counts as failed. A macro file is copied to manuscript/ only after its step
# succeeded and only when the file is newer than the start of that step. A file
# marked "when reported" (the table of lower_bound_reference) is copied when the
# step's last line "<n> macros; wrote ..." names it, and then it must be newer than
# the start of the step; when the line does not name it, it is left out with a note.
#
# p87, p89, p93, p94, p95, p96, p98, p101 to p108, p131 and the inference step of p72
# write manuscript/ and results/ in place. p87, p89, p93, p94 and p72 check their macro names
# against every manuscript/macros_*.tex of the repository and skip only the identical path,
# so a run under another output root would find its own earlier file there as a collision
# (p96, p108 and p89 --family active skip every file with the name of their own output; p98 and
# p101 to p107 skip their own path under both roots). A refusal of these scripts removes their earlier outputs, so no stale
# number stays. p100 and p50 write through arm_registry.out under $PACT_OUT, and so do p69
# (step release_dates), p99, p111, p112, p113, p116 to p122, p124 to p126, p130, p135 to p138, p140, p141 and p33 (step ood,
# $PACT_OUT/paper/macros_ood.tex, $PACT_OUT/results/ood.json and .md). p123 writes its
# figure to manuscript/sections_new/figures/ and its record to out/exact/results/ whatever
# PACT_OUT is.
#
# Usage (from any folder; the script works in the repository root):
#   DRY=1 scripts/run_analysis.sh            steps, commands, preconditions and
#                                                which preconditions hold now; nothing
#                                                is pinned, started or written
#   CORE=<n> scripts/run_analysis.sh         every ready step, each command pinned
#                                                with taskset to core n, one thread
#   CORE=<n> ONLY="family figures" scripts/run_analysis.sh
#
# Variables:
#   CORE               one processor core for every command of a real run (required)
#   DRY                1: print only
#   ONLY               step names, separated by spaces or commas
#   PY                 interpreter (default: the project's environment)
#   PACT_OUT           output root of the registry-routed files (default out/exact)
#   RULECMP_K          sampled schedules per instance of the rule comparison (passed on)
#   HGNN_LR            learning rate of the HGNN runs for p95 (default: the selection)
#   P94_CAMPAIGN_PROD, P94_CAMPAIGN_TRAIN   the timed campaign as designed, for steps timed
#                      and figure_timed: an instance range in a token is the design of its arm
#                      (default production batches: every token on instances 3 to 11;
#                      training size: the memetic algorithm and the policy records on 0 to 29,
#                      CP-SAT on 0 to 9, "10x25 = memetic_rule@600 cpsat_rule@600:0-9
#                      policy_sample@60"); p94 refuses a size named in two descriptions
#   P94_CAMPAIGN_STARTNINE, P94_CAMPAIGN_STARTALL   the two campaigns of step
#                      timed_instance_sets, the p94 runs with --tag startnine and --tag startall
#                      that p130 reads (defaults: the two rule-started searches on the nine
#                      instances of the design, "50x25n30 80x25n30 = memetic_rule@600:3-11
#                      cpsat_rule@600:3-11", and the same on every evidence instance, 3-29)
#   P120_CAMPAIGN_PROD, P120_CAMPAIGN_TRAIN   the campaign as launched for step
#                      lower_bound_reference (defaults: every token on its evidence instances,
#                      so p120 reads the two rule-started searches on instances 3 to 29 of the
#                      production batches and the memetic algorithm on 0 to 29 at 10x25)
#   P120_CAMPAIGN_SOLVER_TRAIN  the CP-SAT cells of the campaign at 10x25 for step
#                      lower_bound_reference (default "10x25 = cpsat_rule@600:0-9"; the
#                      launch description P120_CAMPAIGN_TRAIN names no CP-SAT token, and p120
#                      gives a set without a token no column)
#   TIMED_SLOTS        the core sets the records of the timed campaign come from, for step
#                      memetic_parallel (--slots of p124; default "16-23 8-15"), for the
#                      600-s runs of step memetic_one_minute (--ref-slots of p137) and for its
#                      60-s runs started from the policy's schedule (--pol-slots of p137), and
#                      the same two for step decision_rows (--ref-slots, --pol-slots of p138)
#   MFIVE_SLOTS        the core sets of the 300-s runs started from the rule schedule, for step
#                      memetic_five_minute (--slots of p137 --budget 300) and step decision_rows
#                      (--five-slots of p138); default "16-23"
#   MFIVE_POL_SLOTS    the core sets of the 300-s runs started from the policy's schedule, for
#                      step memetic_five_minute (--pol-slots of p137 --budget 300) and step
#                      decision_rows (--five-pol-slots of p138); default "8-15"
#   LONGSPEC_SEEDS     training seeds of step long_specialists (--pact-seeds of p117;
#                      default 301-305)
#   BUDGETSPEC_SEEDS   training seeds of step budget_specialists (--seeds of p127, its own
#                      setting; PACT_SEEDS stays as above; default 301-310)
#   HELDOUT_SEEDS      training seeds of step heldout_splits (--pact-seeds of p122; default
#                      301-303)
#   DEPLOY_JSON       record of the manual measurement M1
#                      (default $PACT_OUT/results/deploy_cost_cpu_1thread.json)
#   MASKCTL_TEN        the ten schedules of the policy in step mask_controls: auto (default:
#                      reported when every size holds the cells of every run or of the
#                      selected run only, left out when no cell exists, the step stops in
#                      any other state), require, or off
#   MASKCTL_TEN_RUNS   the runs each size must hold when the ten schedules are reported
#                      (--ten_runs of p101 infer; default: every run at 10x25 and the
#                      selected run at 50x25n30 and 80x25n30)
#   FIVE_DEFAULT_POLICY  --default_policy of step five_classes, run (default) or stored; it
#                      must be the value the p103 evaluate parts were started with
#
# Logs: train_log/analysis_chain/<step>.log, with the marker .<step>.started whose
# modification time is the start of the step (the folder is created when the first step of
# a real run starts, so a real run that skips every step writes nothing).
# The run ends with a table of the macro files of manuscript/ (modification time,
# macro count, which document inputs each) and a check that no macro name is defined in
# two of the files that manuscript/main.tex and manuscript/supplement.tex input; a document
# that does not exist is left out with one line, and without both (the public release
# holds no source of the manuscript) the table lists the macro files and their generators
# and the check is left out with one line. Exit status: 0 when every selected step
# finished or was skipped and the check passes, 1 when a step failed, 2 when the
# check finds a duplicate name, 3 on a usage error.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 3

DRY=${DRY:-0}
REAL_RUN=1; [ "$DRY" = 1 ] && REAL_RUN=0      # DRY is set to 1 for the command list of a step
ONLY=${ONLY:-}
CORE=${CORE:-}
PY=${PY:-$HOME/miniconda3/envs/fjsp/bin/python}
LOGDIR=train_log/analysis_chain
P94_CAMPAIGN_PROD=${P94_CAMPAIGN_PROD:-"50x25n30 80x25n30 = memetic_rule@600:3-11 memetic_policy@600:3-11 cpsat_rule@600:3-11 cpsat_policy@600:3-11 policy_sample@120:3-11"}
P94_CAMPAIGN_TRAIN=${P94_CAMPAIGN_TRAIN:-"10x25 = memetic_rule@600 cpsat_rule@600:0-9 policy_sample@60"}
P94_CAMPAIGN_STARTNINE=${P94_CAMPAIGN_STARTNINE:-"50x25n30 80x25n30 = memetic_rule@600:3-11 cpsat_rule@600:3-11"}
P94_CAMPAIGN_STARTALL=${P94_CAMPAIGN_STARTALL:-"50x25n30 80x25n30 = memetic_rule@600:3-29 cpsat_rule@600:3-29"}
P120_CAMPAIGN_PROD=${P120_CAMPAIGN_PROD:-"50x25n30 80x25n30 = memetic_rule@600 memetic_policy@600 cpsat_rule@600 cpsat_policy@600 policy_sample@120"}
P120_CAMPAIGN_TRAIN=${P120_CAMPAIGN_TRAIN:-"10x25 = memetic_rule@600 policy_sample@60"}
P120_CAMPAIGN_SOLVER_TRAIN=${P120_CAMPAIGN_SOLVER_TRAIN:-"10x25 = cpsat_rule@600:0-9"}
TIMED_SLOTS=${TIMED_SLOTS:-"16-23 8-15"}
MFIVE_SLOTS=${MFIVE_SLOTS:-"16-23"}
MFIVE_POL_SLOTS=${MFIVE_POL_SLOTS:-"8-15"}
LONGSPEC_SEEDS=${LONGSPEC_SEEDS:-301-305}
BUDGETSPEC_SEEDS=${BUDGETSPEC_SEEDS:-301-310}
HELDOUT_SEEDS=${HELDOUT_SEEDS:-301-303}
read -ra TIMED_SLOT_LIST <<<"$TIMED_SLOTS"
for x in "${TIMED_SLOT_LIST[@]}"; do
  [[ "$x" =~ ^[0-9]+(-[0-9]+)?$ ]] || {
    echo "REFUSING: TIMED_SLOTS=$TIMED_SLOTS, want core sets such as \"16-23 8-15\"" >&2; exit 3; }
done
read -ra MFIVE_SLOT_LIST <<<"$MFIVE_SLOTS"
read -ra MFIVE_POL_SLOT_LIST <<<"$MFIVE_POL_SLOTS"
for x in "${MFIVE_SLOT_LIST[@]}" "${MFIVE_POL_SLOT_LIST[@]}"; do
  [[ "$x" =~ ^[0-9]+(-[0-9]+)?$ ]] || {
    echo "REFUSING: MFIVE_SLOTS=$MFIVE_SLOTS MFIVE_POL_SLOTS=$MFIVE_POL_SLOTS, want core sets such as \"16-23\"" >&2; exit 3; }
done
HGNN_LR=${HGNN_LR:-}
MASKCTL_TEN=${MASKCTL_TEN:-auto}
MASKCTL_TEN_RUNS=${MASKCTL_TEN_RUNS:-10x25=all,50x25n30=selected,80x25n30=selected}
FIVE_DEFAULT_POLICY=${FIVE_DEFAULT_POLICY:-run}
case $MASKCTL_TEN in auto|require|off) ;; *)
  echo "REFUSING: MASKCTL_TEN=$MASKCTL_TEN, want auto, require or off" >&2; exit 3;; esac
case $FIVE_DEFAULT_POLICY in run|stored) ;; *)
  echo "REFUSING: FIVE_DEFAULT_POLICY=$FIVE_DEFAULT_POLICY, want run or stored" >&2; exit 3;; esac

export PACT_ENV=exact
export PACT_SEEDS=301-305
export PACT_OUT=${PACT_OUT:-out/exact}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=
export PYTHONDONTWRITEBYTECODE=1
# the analyses read the default result roots of the registry; the completeness check
# of eval_newseeds_plan.py must look at the same folders
unset PACT_RESULTS EVAL_GROUPS PUBLIC PUBSENS PUBEQ PUBACT PUBACTFAM PUBACTSENS PUBACTEQ LEGACY45 HGNN EXACT_SEEDS LEGACY_SEEDS \
      RESULTS STORED PUB_RESULTS CACHE_ROOT
[ -n "$HGNN_LR" ] && export HGNN_LR
[ -n "${RULECMP_K:-}" ] && export RULECMP_K
DEPLOY_JSON=${DEPLOY_JSON:-$PACT_OUT/results/deploy_cost_cpu_1thread.json}

PUB_TAGS=(sdtwenty brand edata rdata vdata)
declare -A PUB_SET=([sdtwenty]=pub_SD2_test_20x10+mix [brand]=pub_Brandimarte
                    [edata]=pub_Hurink_edata [rdata]=pub_Hurink_rdata [vdata]=pub_Hurink_vdata)
STEPS=(family seed_extension instance_counts long_specialists budget_specialists heldout_splits per_floor
       production_specialists ood
       asym_setups wrong_descriptor example_schedule seed_spread convergence release_dates
       cross_eval public pubact public_act rulecmp_public_act public_seed_extension rulecmp_public)
for t in "${PUB_TAGS[@]}"; do STEPS+=("rulecmp_public_$t"); done
STEPS+=(rulecmp_factory solver_selection memetic_selection timed timed_instance_sets memetic_parallel
        memetic_one_minute memetic_five_minute decision_rows instance_bound lower_bound_reference
        randomized_rules memetic_public secondary margin_and_pairs environments sensitivity aware_rules best_of_seven
        mask_controls reference_replay mask_rules mask_choice_validation actmask scale_sensitivity five_classes
        project_mix figure_limits rc_trained divisor
        deploy rule_change_cost model_facts figures figure_timed)
SENS_ARGS=(--mask exact --arm_names xmask --seeds 301,302,303,304,305)
AWARE_SIZES=(10x25 20x25 30x25 50x25n30 80x25n30)

if [ "$DRY" != 1 ]; then
  if ! [[ "$CORE" =~ ^[0-9]+$ ]]; then
    echo "REFUSING: a real run needs CORE=<one core number> (or DRY=1)" >&2
    exit 3
  fi
fi
if [ ! -x "$PY" ]; then
  echo "REFUSING: interpreter $PY not found (set PY)" >&2
  exit 3
fi

declare -A SELECTED=()
if [ -n "$ONLY" ]; then
  for w in ${ONLY//,/ }; do
    found=0
    for s in "${STEPS[@]}"; do [ "$s" = "$w" ] && found=1; done
    if [ "$found" = 0 ]; then
      echo "REFUSING: unknown step '$w' in ONLY (steps: ${STEPS[*]})" >&2
      exit 3
    fi
    SELECTED[$w]=1
  done
else
  for s in "${STEPS[@]}"; do SELECTED[$s]=1; done
fi

# ---------------------------------------------------------------- helper (Python)
# Read-only checks: completeness of evaluation groups, the run selection, the
# rule-comparison and timed-campaign records, the deployment record, and the table
# and duplicate check of the macro files. Nothing here writes a file.
read -r -d '' HELPER <<'PYEOF'
import collections, contextlib, glob, io, json, os, re, sys, time
ROOT = os.getcwd()
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, ROOT)


def say(msg):
    print(f'      {msg}', flush=True)


def plan_check(*groups):
    """scripts/eval_newseeds_plan.py check for the named groups (exit 0 = complete)."""
    os.environ['EVAL_GROUPS'] = ' '.join(groups)
    os.environ['EXACT_SEEDS'] = '301 302 303 304 305'
    import eval_newseeds_plan as P
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = P.check()
    lines = buf.getvalue().splitlines()
    inc = [x.strip() for x in lines if 'INCOMPLETE' in x]
    for x in lines:
        if x.startswith('complete:'):
            say(f'eval_newseeds_plan check ({" ".join(groups)}): {x.strip()}')
    for x in inc[:8]:
        say(x)
    if len(inc) > 8:
        say(f'... and {len(inc) - 8} more incomplete folders')
    return rc == 0


def select_seed():
    import arm_registry as REG
    s, rec = REG.select_seed('released')
    say(f'selected run s{s} (validation scores: '
        + ', '.join(f's{k} {v[0]:.5f}' for k, v in rec.items()) + ')')
    return True


def pubeq_state():
    import p89_public_family as P89
    st = P89.Inputs(P89.REPO).eq_state()
    say('equal-experience arm of the public family: '
        + ('every file present, section 8 is computed' if st else
           'no file present, section 8 is left out'))
    return True


def pubact_state():
    """The arm of the public family under the active mask (p108 readiness, read only, seeds of
    p108 --seeds auto): a 'yes' when every file of its runs and of the comparator runs of
    those seeds exists and every reference is counted; no file of the arm is a 'no' that
    skips the step, a partial arm a 'no' that names what is missing (p108 refuses it)."""
    import p108_public_active_arm as P108
    x = P108.readiness(P108.REPO)
    st = x['arm']
    say(f'arm {P108.STEM} of the public family under the active mask, seeds {x["seeds"][0]} to '
        f'{x["seeds"][-1]}: {st["found"]} of {st["expected"]} '
        f'files, ' + {'none': 'no file present, the step waits for the runs and their '
                              'evaluation (EVAL_GROUPS=pubact)',
                      'partial': 'partial (p108 refuses a partial arm)',
                      'complete': 'every file present'}[st['state']])
    if st['state'] == 'partial':
        for f in st['missing'][:4] + [f'unexpected {u}' for u in st['unexpected'][:4]] + \
                [f'not finished {u}' for u in st['unfinished']]:
            say(f'  {f}')
        if len(st['missing']) > 4:
            say(f'  ... and {len(st["missing"]) - 4} more missing')
    if x['comparator_missing']:
        say(f'comparator {P108.COMPARATOR}: {len(x["comparator_missing"])} files missing, first '
            f'{x["comparator_missing"][0]}')
    rf = x['references']
    say(f'references ({rf["folder"]}): {rf["valid"]} of {rf["expected"]} counted')
    return x['ready']


def pubactfam_state():
    """The public family under the active mask (p89 --family active; read only). A 'no' that
    skips the step while no file of a run of the family exists apart from the policy runs of
    seeds 301 to 303 that step pubact reads; a 'yes' when every file of every principal run of
    seeds 301 to 305 exists, every run has finished, every rule array of p109 under the active
    mask exists and the equal-experience recipe pubx-eqact has every file or none; anything
    else is a 'no' that names what is missing (p89 refuses a partial family)."""
    import arm_registry as REG
    import p89_public_family as P89
    I = P89.Inputs(P89.REPO, P89.Fam('active'))
    fs = P89.family_state(I, P89.SEEDS)
    n_miss = sum(len(v) for v in fs['missing'].values())
    beyond = [x for arm, combo, _ in I.runs() for s in P89.SEEDS
              if not (arm == 'released' and s in REG.PUB_ACT_SEEDS)
              for x in P89.run_files(I, arm, combo, s) if os.path.exists(I.p(x))]
    if not beyond:
        say('public family under the active mask: no file of a run beyond the policy runs of '
            'seeds 301 to 303, the step waits for the family (ARMS=actfam and ARMS=acteq of '
            'scripts/run_pubfamily_queue.sh, EVAL_GROUPS="pubactfam pubactsens pubacteq")')
        return False
    ok = True
    if n_miss:
        ok = False
        say(f'public family under the active mask: partial, {n_miss} files of the principal runs '
            f'missing or unfinished (p89 refuses a partial family); per seed '
            + ', '.join(f's{s} {len(v)}' for s, v in fs['missing'].items()))
        for x in [x for s in P89.SEEDS for x in fs['missing'][s]][:4]:
            say(f'  missing {x}')
    else:
        say('public family under the active mask: every file of the principal runs of seeds '
            f'{P89.SEEDS[0]} to {P89.SEEDS[-1]} present')
    if fs['rules_missing']:
        ok = False
        say(f'rule arrays under the active mask: {len(fs["rules_missing"])} files missing, first '
            f'{fs["rules_missing"][0]} (scripts/p109_public_active_rules.py run)')
    else:
        say('rule arrays under the active mask: every file present')
    try:
        eq = I.eq_state()
        say(f'equal-experience recipe {I.fam.eq_stem}: '
            + ('every file present, section 8 is computed' if eq else
               'no file present, section 8 is left out'))
    except BaseException as e:
        ok = False
        msg = str(e).strip().splitlines()
        say(f'equal-experience recipe {I.fam.eq_stem}: {msg[0][:300] if msg else type(e).__name__}')
    return ok


def pubref_state():
    """References of the public benchmark (p89 section 9, records of p91): counts per
    combination; a 'no' only when a counted regime-N reference contradicts the best-known
    table, which p89 would refuse (the stored makespans are checked by p89 itself)."""
    import p89_public_family as P89
    x = P89.reference_summary(P89.REPO)
    st = x['state']
    say(f'references of {P89.REF_SET} ({x["folder"]}): {st["total"]["valid"]} of {st["expected"]} '
        f'valid, {st["total"]["present"]} present, {st["total"]["proved"]} proved optimal, '
        f'{len(x["not_counted"])} not counted; valid per combination '
        + ', '.join(f'{r} {v["valid"]}' for r, v in st['per'].items()))
    bk = x['best_known_N']
    say(f'regime N against the best-known values: below {bk["below"]}, equal {bk["equal"]}, above '
        f'{bk["above"]} (of {bk["counted"]}); section 9 is '
        + ('computed' if st['complete_all'] else 'left out until every reference is valid'))
    for p in x['problems'][:5]:
        say(f'contradiction (p89 refuses): {p}')
    return not x['problems']


def rulecmp_ready(name, set_name='', mask=''):
    import rulecmp_plan as RP
    fam = RP.Family(name, set_name=set_name or None, mask=mask or None)
    js = RP.jobs(fam)
    by = collections.OrderedDict()
    for j in js:
        by.setdefault(j.kind, collections.Counter())[j.status] += 1
    say(f'rule comparison {fam.label}: ' + '; '.join(
        f'{k} ' + ', '.join(f'{v} {s}' for s, v in sorted(c.items())) for k, c in by.items()))
    ok = bool(js) and all(j.status == 'done' for j in js)
    miss = RP.missing_inputs(fam)
    if any(miss.values()):
        ok = False
        say('missing inputs: ' + ', '.join(f'{len(v)} {k}' for k, v in miss.items() if v))
    recs = fam.check_records()
    if recs:
        ok = False
        say('records: ' + '; '.join(recs)[:300])
    _, probs = fam.vali_select()
    if probs:
        ok = False
        say('validation settings: ' + '; '.join(probs)[:300])
    if name == 'public' and not fam.headline:
        vr, why = RP.validation_rules(fam)
        if not vr:
            ok = False
            say(f'rule of form (c): {why}')
    return ok


def rulecmp_act_state():
    """The rule comparison of the public family under the active mask (p93 --mask active on
    the six public test sets; read only). A 'no' that skips the step while no output of a job
    of the campaign exists on any set; a 'yes' when every job of every set is done, no input is
    missing and the rule of form (c) is readable; anything else is a 'no' that names what is
    missing (p93 refuses a partial campaign)."""
    import rulecmp_plan as RP
    fams = [RP.Family('public', set_name=st, mask='active') for st in RP.PUB_SET_INFO]
    begun = [RP.rel(f.sub(k)) for f in fams for k in RP.SUB if os.path.exists(f.sub(k))]
    if not begun:
        say('rule comparison under the active mask: no output of a job on any of the six public '
            'sets, the step waits for the campaign (MASK=active scripts/run_rule_comparison.sh)')
        return False
    ok = [rulecmp_ready('public', f.set, 'active') for f in fams]
    say('rule comparison under the active mask: '
        + ('every job of the six public sets done' if all(ok) else
           f'partial, {ok.count(False)} of {len(ok)} sets not ready (p93 refuses a partial '
           f'campaign)'))
    return all(ok)


def timed_ready(*specs):
    import arm_registry as REG
    import p94_timed_inference as T
    P = T.P
    a = T.build_parser().parse_args(sum((['--campaign', s] for s in specs), []))
    ok, sel = True, None
    try:
        sel, _ = T.select_policy_seed()
        say(f'selected run s{sel}')
    except (REG.MissingResult, SystemExit) as e:
        ok = False
        say(f'run selection unavailable: {str(e).splitlines()[0][:200]}')
    defaults = P.build_parser().parse_args(['table'])
    slots = [P.parse_cores(s) for s in (a.slots or defaults.slots)]
    bars = a.bars or defaults.bars
    if not os.path.exists(bars):
        ok = False
        say(f'frozen CPU-share bars {bars} absent')
    chosen, why = T.select_cpsat_arm('test_results/FAMILY', slots, bars)
    if chosen is None:
        ok = False
        say(f'pilot selection unavailable ({len(why)} problem(s); first: '
            f'{why[0][:160] if why else "-"})')
    else:
        say(f'CP-SAT configuration chosen by the pilot: {chosen}')
    for spec in specs:
        sizes, toks = T.parse_campaign(spec)
        keep = []
        for t in toks:
            if t.partition('@')[0] == T.POLICY_ARM:
                # read by p94 itself on the clock since the request, with the rule records
                for size in sizes:
                    n_pol = len(glob.glob(f'or_solution/FAMILY/{size}_polsample*/*/instance_*.json'))
                    n_rule = len(glob.glob(f'or_solution/FAMILY/{size}_rulestart/*/instance_*.json'))
                    say(f'{t} at {size}: {n_pol} policy records and {n_rule} rule records on disk')
                    if not n_pol or not n_rule:
                        ok = False
                continue
            runner, start = P.ARMS[t.partition('@')[0]]
            if start == 'policy' and sel is None:
                ok = False
                say(f'{t}: the folders of policy-started records need the selected run')
            elif runner == 'cpsat' and chosen is None:
                ok = False
                say(f'{t}: the folders of CP-SAT records need the pilot\'s choice')
            else:
                keep.append(t)
        if not keep:
            continue
        # the planner's parser wants a known CP-SAT configuration; without the pilot's
        # choice no CP-SAT token is kept, so the name given here selects no folder
        camp = T.make_campaign(sizes, keep, a, sel, chosen or 'ref_fullhint')
        for seg in camp.segments:
            st = P.cell_states(camp, seg)
            cnt = collections.Counter(s for _, s, _ in st.values())
            say(f'{seg.label} at {" ".join(sizes)}: {len(st)} cells, '
                + ', '.join(f'{v} {k}' for k, v in sorted(cnt.items())))
            if cnt.get('valid', 0) != len(st):
                ok = False
    return ok


def p88_sections():
    """Inputs of the sections F, G and H of p88 in the active environment and seeds: every
    array present with one row per test instance (p88 --require-sections all refuses
    otherwise)."""
    import p88_family_inference as F88
    ok = True
    for sec in F88.SECTIONS:
        st = F88.section_state(sec)
        say(f'p88 section {sec} ({st["title"]}): {st["arrays"]} arrays, '
            + ('every one present with one row per test instance' if st['ready'] else
               f'not ready: {st["reason"][:300]}'))
        ok = ok and st['ready']
    return ok


def p96_ready():
    """Every array of both environments that p96 reads (seeds 301 to 303, 10x25)."""
    import p96_two_environments as P96
    miss = P96.missing_inputs()
    for e, v in miss.items():
        say(f'p96, {e}: ' + ('every array present with one row per test instance' if not v
                             else f'{len(v)} problem(s), first {v[0][:200]}'))
    return not any(miss.values())


def report(what, n, miss, bad, limit=6):
    """One line of counts and the first missing or unusable files of a readiness check."""
    say(f'{what}: {n - len(miss) - len(bad)} of {n} present with one row per test instance'
        + (f'; {len(miss)} missing' if miss else '') + (f'; {len(bad)} unusable' if bad else ''))
    for x in (miss + bad)[:limit]:
        say(f'  {x}')
    if len(miss) + len(bad) > limit:
        say(f'  ... and {len(miss) + len(bad) - limit} more')
    return not miss and not bad


def record_ok(js, names, want=None):
    """'' when the JSON record of a cell exists, covers exactly these instances and holds
    the wanted fields; otherwise the reason."""
    if not os.path.exists(js):
        return 'missing'
    rec = json.load(open(js))
    if rec.get('instances') != names:
        return f'{len(rec.get("instances") or [])} instances, the study needs {len(names)}'
    for k, v in (want or {}).items():
        got = rec.get(k, (rec.get('design') or {}).get(k))
        if got != v:
            return f'{k} is {got!r}, the study needs {v!r}'
    return ''


def rows_ok(path, n):
    import numpy as np
    if not os.path.exists(path):
        return 'missing'
    a = np.load(path)
    return '' if a.shape[0] >= n else f'{a.shape[0]} rows, the study needs {n}'


def p100_ready():
    """Every array of p100 (policy, diagonal and off-combination runs of the active seeds)
    with its checkpoint, log and configuration, and the off-combination arrays the
    principal campaign also holds equal (p100 load_all, read only)."""
    import arm_registry as REG
    import p100_cross_eval as X
    try:
        X.check_layout()
        n, cache, compared = X.load_all(ROOT, REG.seeds())
    except X.Refused as e:
        lines = str(e).replace('refused: ', '').splitlines()
        say(f'p100: {lines[0][:300]}')
        for x in lines[1:7]:
            say(f'  {x.strip()[:200]}')
        if len(lines) > 7:
            say(f'  ... ({len(lines) - 7} more lines)')
        return False
    say(f'p100: {len(cache)} arrays present with {n} rows each; {compared} off-combination '
        f'arrays equal to those of the principal campaign')
    return True


def p101_ready(ten='auto'):
    """Inputs of p101 infer --trace_arms SPT,policy: the cells of the random control, of
    the SPT trace and of the policy trace of every active seed, the stored rule arrays and
    the released policy's stored greedy arrays at every size and combination; the cells of
    the ten schedules of the policy (none under auto, or at every size the cells of every
    run or of the selected run only; under require one of the two at every size), with the
    selected run when they exist."""
    import arm_registry as REG
    import p101_mask_controls as P
    root, seeds = P.DEFAULT_ROOT, list(REG.seeds())
    models = [P.policy_model(s) for s in seeds]
    miss, bad, n = [], [], 0
    present, ten_bad = {}, []
    for size in P.SIZES:
        names = P.names_of(size)
        for combo in P.R8:
            labels = ['random', 'trace_SPT'] + [P.trace_label(m) for m in models]
            for lb in labels:
                n += 1
                npz, js = P.cell_paths(root, size, combo, lb)
                why = 'missing' if not os.path.exists(npz) else record_ok(js, names)
                if why:
                    (miss if why == 'missing' else bad).append(f'{npz}: {why}')
            paths = [os.path.join(P.ROOT_DIR, P.RESULTS_ROOT, f'{size}_xmask',
                                  f'Result_{r}_{combo}.npy') for r in P.RULES]
            paths += [P.stored_greedy_path(size, combo, m) for m in models]
            for p in paths:
                n += 1
                why = rows_ok(p, len(names))
                if why:
                    (miss if why == 'missing' else bad).append(
                        f'{os.path.relpath(p, P.ROOT_DIR)}: {why}')
        present[size] = P.ten_cells_present(root, size, P.R8, models)
        for combo, have in present[size].items():
            for m in have:
                npz, js = P.cell_paths(root, size, combo, P.ten_label(m))
                why = record_ok(js, names, {'draws': P.N_DRAWS, 'model': m})
                if why:
                    ten_bad.append(f'{npz}: {why}')
    ok = report('p101 cells of the random control, the SPT and policy traces, stored rule '
                'and policy arrays', n, miss, bad)
    nt = sum(len(v) for P_ in present.values() for v in P_.values())
    if ten == 'off':
        say(f'ten schedules of the policy: left out (MASKCTL_TEN=off); {nt} cells present')
        return ok
    if not nt:
        if ten == 'auto':
            say('ten schedules of the policy: no cell present, the step runs without them '
                '(MASKCTL_TEN=auto)')
            return ok
        say('ten schedules of the policy: no cell present; MASKCTL_TEN=require needs at every '
            'size the cells of every run or of the selected run')
        return False
    try:
        class A:
            policy_models, policy_seeds = None, seeds
        sel = P.ten_select(A)
    except SystemExit as e:
        say(f'ten schedules of the policy: {nt} cells present, but {str(e)[:300]}')
        return False
    forms, ten_ok = {}, not ten_bad
    for size, pr in present.items():
        form, why = P.ten_form_of(pr, models, sel['model'])
        forms[size] = form
        if form == 'other':
            say(f'ten schedules of the policy, {size}: {why}')
            ten_ok = False
    if any(f is None for f in forms.values()):
        say('ten schedules of the policy: cells at some sizes only ('
            + ', '.join(f'{s} {"none" if f is None else P.TEN_RUN_TEXT.get(f, f)}'
                        for s, f in forms.items()) + ')')
        ten_ok = False
    for x in ten_bad[:6]:
        say(f'  {x}')
    if ten_ok:
        say('ten schedules of the policy: ' + ', '.join(f'{s} {P.TEN_RUN_TEXT[f]}'
                                                        for s, f in forms.items())
            + f'; selected run s{sel["seed"]}')
    return ok and ten_ok


def p50_ready():
    """Inputs of p50 in the exact environment (p50 assess, read only): the arrays of
    xmask-active and of the policy on the seeds both have, at every folder and combination
    the plan gives the arm, with checkpoints, logs, configurations (the arm records
    mask_mode active, both the exact mask options and the same training budget) and
    evaluation summaries."""
    import p50_actmask_analysis as A
    x = A.assess(ROOT)
    sets = ', '.join(f'{f} ({len(c)} combinations)' for f, c in x['sets'])
    say(f'p50: {A.ARM} against the policy on seeds {",".join(map(str, x["seeds"]))} at {sets}')
    ok = report('p50 files (arrays, checkpoints, logs, configurations, summaries)', x['files'],
                x['missing'], x['problems'])
    if not x['arm_arrays']:
        say(f'no result array of {A.ARM} exists')
    return ok and bool(x['arm_arrays'])


def p102_ready():
    """Inputs of p102 infer with every arm: the manifests of the level win_full and every
    result file of the rules, the policy, the specialists and the single-rule control of
    the active seeds at both levels."""
    import arm_registry as REG
    import p102_scale_sensitivity as P
    root, seeds = P.DEFAULT_ROOT, list(REG.seeds())
    miss, bad, n = [], [], 0
    for size in P.SIZES:
        names = P.P101.names_of(size)
        mp = os.path.join(root, 'data', 'win_full', size, 'manifest.json')
        n += 1
        if not os.path.exists(mp):
            miss.append(f'{mp}: missing (build phase)')
        for level, sz, combo, arm, s, label in P.all_units(P.LEVELS, [size], P.WIN_COMBOS,
                                                           P.ARM_CHOICES, seeds):
            n += 1
            p = P.result_path(root, level, sz, combo, label)
            why = 'missing' if not os.path.exists(p) else record_ok(
                p[:-4] + '.json', names, {'level': level, 'regime': combo, 'label': label})
            if why:
                (miss if why == 'missing' else bad).append(f'{p}: {why}')
    return report('p102 result files (rules, policy, specialists, single-rule control; levels '
                  'default and win_full)', n, miss, bad)


def p103_ready(default_policy='run'):
    """Inputs of p103 infer with every arm: the manifests of level five and every result
    file of the rules, the policy and the specialists of the active seeds at both levels,
    with zero violations; with --default_policy stored, a default-level policy run whose
    exact-campaign array exists is read from that array instead."""
    import arm_registry as REG
    import p103_five_classes as P
    root, seeds = P.DEFAULT_ROOT, list(REG.seeds())

    class A:
        environment = 'exact'
    A.default_policy = default_policy
    miss, bad, n = [], [], 0
    for size in P.SIZES:
        names = P.P101.names_of(size)
        mp = P.manifest_path(root, size)
        n += 1
        if not os.path.exists(mp):
            miss.append(f'{mp}: missing (build phase)')
        units = P.all_units(P.LEVELS, [size], P.COMBOS, P.ARM_CHOICES, seeds, 'exact')
        need = P.units_needed(units, A)
        for u in units:
            level, sz, combo, arm, s, label = u
            n += 1
            if u not in need:
                p = P.stored_policy_path(sz, combo, label, 'exact')
                why = rows_ok(p, len(names))
            else:
                p = P.result_path(root, 'exact', level, sz, combo, label)
                why = 'missing' if not os.path.exists(p) else record_ok(
                    p[:-4] + '.json', names, {'level': level, 'regime': combo, 'label': label,
                                              'class_count': P.LEVEL_K[level],
                                              'violations': 0, 'validated': len(names)})
            if why:
                (miss if why == 'missing' else bad).append(f'{p}: {why}')
    return report(f'p103 result files (rules, policy, specialists; levels default and five; '
                  f'--default_policy {default_policy})', n, miss, bad)


def p129_ready():
    """Inputs of p129 infer with every arm: the manifests of the four new sets and every
    result file of the seven rules, the policy and the specialists of the active seeds at
    the levels mix_rc and mix_steel (10, 50 and 80 modules), with zero violations, and the
    default-level arrays it reads."""
    import arm_registry as REG
    import p129_project_mix as P
    root, data, seeds = P.DEFAULT_ROOT, P.DEFAULT_DATA, list(REG.seeds())
    miss, bad, n = [], [], 0
    for level in P.MIX_LEVELS:
        for size in P.SIZES:
            if P.is_new(level, size):
                mp = P.manifest_path(data, level, size)
                n += 1
                if not os.path.exists(mp):
                    miss.append(f'{mp}: missing (build phase)')
                    continue
            names = P.level_names(level, size, data)
            for u in P.all_units([level], [size], P.R8, P.ARM_CHOICES, seeds):
                n += 1
                p = P.result_path(root, *u[:3], u[5])
                why = 'missing' if not os.path.exists(p) else record_ok(
                    p[:-4] + '.json', names, {'level': level, 'regime': u[2], 'label': u[5],
                                              'violations': 0, 'validated': len(names)})
                if why:
                    (miss if why == 'missing' else bad).append(f'{p}: {why}')
    for size in P.SIZES:
        for combo in P.R8:
            for p in P.default_paths(size, combo, seeds).values():
                n += 1
                why = rows_ok(p, P.P101.SIZES[size]['n'])
                if why:
                    (miss if why == 'missing' else bad).append(f'{p}: {why}')
    for key, p in P.record_paths().items():       # the records infer compares its rows with
        n += 1
        if not os.path.exists(p):
            miss.append(f'{p}: missing (record of {key})')
    return report('p129 result files (rules, policy, specialists; levels mix_rc and mix_steel), '
                  'default-level arrays and the records of p88, p128 and p87', n, miss, bad)


def p135_ready():
    """The condition of a full run of p135 infer: every cell of the design has a valid record,
    and every stored input that p124's readers return is present and accepted."""
    import p135_randomized_rules as P
    a = P.resolve(P.build_parser().parse_args(['infer']))     # the options of infer
    cells = P.plan_cells(a)
    gaps = [c for c in cells if c['state'] != 'done']
    say(f'p135 records: {len(cells) - len(gaps)} of {len(cells)} cells valid')
    for c in gaps[:6]:
        say(f'  {c["record"]}: {P.STATE_WORD.get(c["state"], c["state"])}')
    stored = P.read_stored(a)
    say(f'p135 stored inputs: {len(stored["missing"])} missing, {len(stored["rejected"])} '
        f'rejected, {len(stored["problems"])} problems; selected run {stored["selected_run"]}, '
        f'CP-SAT configuration {stored["cpsat_configuration"]}')
    for x in (list(stored['missing']) + list(stored['rejected']) + list(stored['problems']))[:6]:
        say(f'  {x}')
    return (not gaps and not stored['missing'] and not stored['rejected']
            and not stored['problems'] and stored['selected_run'] is not None
            and stored['cpsat_configuration'] is not None)


def p136_ready():
    """The condition of a full run of p136 infer: a valid record for each of its runs, read
    with the options of infer, and a best-known makespan for each instance."""
    import p136_memetic_public_check as P
    a = P.build_parser().parse_args(['plan'])
    a.names = [x.strip() for x in a.instances.split(',') if x.strip()]
    a.seeds = P.TCP.parse_instances(a.seeds)
    a.cores_allowed = None             # each record is classified by its own run_cores
    P.best_known()                     # refuses when an instance has no best-known value
    cells = P.cells(a)
    gaps = [c for c in cells if c['state'] != 'valid']
    say(f'p136 records: {len(cells) - len(gaps)} of {len(cells)} runs valid')
    for c in gaps[:6]:
        say(f'  {c["record"]}: {c["state"]}')
    return not gaps and float(a.budget) >= max(P.READINGS)


def p104_ready():
    """Inputs of p104 infer: the selected run of the active seeds and its cell of every
    combination, covering the 100 test instances, written for that run and checkpoint and
    not marked as a development run."""
    import arm_registry as REG
    import p104_divisor_probe as P
    s, model, _ = P.selected(list(REG.seeds()))
    say(f'selected run s{s} ({model})')
    names = P.P101.names_of(P.SIZE)
    ckpt = os.path.join('trained_network', 'FAMILY', f'{model}.pth')
    want = {'model': model, 'development': False}
    if os.path.exists(ckpt):
        want['checkpoint_sha256'] = P.P101.sha256_file(ckpt)
    miss, bad = [], []
    for combo in P.R8:
        npz, js = P.P101.cell_paths(P.DEFAULT_ROOT, P.SIZE, combo, P.label_of(model))
        why = 'missing' if not os.path.exists(npz) else record_ok(js, names, want)
        if why:
            (miss if why == 'missing' else bad).append(f'{npz}: {why}')
    return report('p104 cells of the selected run', len(P.R8), miss, bad)


def p98_ready():
    """Inputs of p98 (its analysis, which writes nothing): the records of the selection runs
    of the eight CP-SAT candidates and the frozen CPU-share bars, accepted by
    timed_campaign_plan.select_pilot, every record with its warm start, CPU share and an
    objective equal to the value the selection rule scores."""
    import p98_solver_selection as S
    a = S.build_parser().parse_args([])
    cwd = os.getcwd()
    try:
        os.chdir(S.ROOT)
        with contextlib.redirect_stdout(io.StringIO()):
            R = S.analyse(S.P.rel(os.path.abspath(a.records_root)),
                          S.P.rel(os.path.abspath(a.bars)), a)
    except S.Refusal as e:
        why = e.args[0]
        say(f'p98: {len(why)} problem(s) in the records of the selection runs; first: '
            f'{why[0][:300] if why else "-"}')
        return False
    finally:
        os.chdir(cwd)
    sel = R['selection']
    say(f'p98: records of the selection runs complete, {sel["runs"]} runs per candidate, row '
        f'{sel["chosen_row"]} ({sel["chosen"]}) selected')
    return True


def p106_ready():
    """Inputs of p106 (its analysis, which writes nothing): the records of the three candidate
    configurations of the memetic algorithm on the 24 cells of the selection runs, each record
    validated and of the design (seed, budget, starting schedule)."""
    import p106_memetic_selection as M
    try:
        runs, configs, derived = M.read_records(M.DEFAULT_RECORDS)
        R = M.analyse(runs, configs, derived, M.DEFAULT_RECORDS)
    except M.Refusal as e:
        why = e.args[0]
        say(f'p106: {len(why)} problem(s) in the records of the selection runs; first: '
            f'{why[0][:300] if why else "-"}')
        return False
    say(f'p106: records of the selection runs complete, {len(R["cells"])} cells of '
        f'{len(R["candidates"])} configurations, configuration {R["selected"]} selected')
    return True


def case_report(what, n, why):
    """One line of counts of the planned cases of p105 or p107, with the reasons of the cases
    that do not count."""
    by = collections.Counter(w.split(': ', 1)[1].split(' (')[0] for w in why)
    say(f'{what}: {n - len(why)} of {n} planned cases count'
        + ('' if not why else '; ' + ', '.join(f'{v} {k}' for k, v in by.most_common())))
    for x in why[:4]:
        say(f'  {x[:200]}')
    return not why


def p105_ready():
    """Inputs of p105 infer --require all (read only): every planned case counts, that is its
    reference record is valid in its owning campaign and the case holds this design and this
    reference record; the stored rule arrays that infer reads."""
    import argparse
    import p105_reference_replay as P
    a = argparse.Namespace(root=P.DEFAULT_ROOT, sets=list(P.SETS), refs=list(P.REFS),
                           combos=list(P.R8), n=None, cpsat_arm=None)
    states = P.reference_states(a)
    cases, why = P.case_list(a), []
    for s, r, c, nm in cases:
        st, w, path, _ = P.ref_state(states, s, r, c, nm)
        if st != 'valid':
            why.append(f'{s}/{r}/{c}/{nm}: reference {st}' + (f' ({w})' if w else ''))
            continue
        cs, _ = P.case_state(a.root, s, r, c, nm, P.read_reference(s, r, path)[4])
        if cs != 'done':
            why.append(f'{s}/{r}/{c}/{nm}: ' + ('case not run' if cs == 'absent' else
                                                 'case of another design or reference record'))
    for s in a.sets:
        for c in a.combos:
            P.rule_arrays(s, c)
    return case_report('p105 cases', len(cases), why)


def p107_ready():
    """Inputs of p107 infer (read only): every case of every set, combination and planned
    instance counts, with the design and the reference records of p105, every reference
    valid; the stored rule arrays that infer reads."""
    import argparse
    import p107_mask_rule_variants as P
    a = argparse.Namespace(root=P.DEFAULT_ROOT, sets=list(P.SETS), refs=list(P.DEFAULT_REFS),
                           combos=list(P.R8), n=None, cpsat_arm=None)
    states = P.states_for(a)
    cases, why = P.case_list(a), []
    for s, c, nm in cases:
        rec, w = P.read_case(a.root, s, c, nm, a.refs, P.case_references(states, s, c, nm, a.refs))
        if rec is None:
            why.append(w)
    for s in a.sets:
        for c in a.combos:
            P.P105.rule_arrays(s, c)
    return case_report('p107 cases', len(cases), why)


def p99_ready():
    """Inputs of p99 infer --require all other than the count that the step writes first (read
    only): one usable training-cost record of the policy and of each specialist under
    train_log/training_cost (scripts/run_training_cost.sh, measurement M3), accepted by p99's own
    reader, and a valid record of p99 checks at the three sizes (measurement M4)."""
    import argparse
    import arm_registry as REG
    import p99_rule_change_cost as P
    ok = True
    try:
        got = P.load_records(argparse.Namespace(records=None, records_dir=None, root=P.ROOT), [])
    except BaseException as e:
        msg = str(e).strip().splitlines()
        say(f'p99: training-cost records refused: {msg[-1][:300] if msg else type(e).__name__}')
        got, ok = {}, False
    want = [('released', None)] + [('specialist', r) for r in REG.COMBOS]
    miss = [a if c is None else f'{a} {c}' for a, c in want if (a, c) not in got]
    say(f'p99: training-cost records under {P.TCE.RECORD_DIR}: {len(want) - len(miss)} of '
        f'{len(want)} arms' + (f'; missing: {", ".join(miss)}' if miss else ''))
    path = os.path.join(os.environ.get('PACT_OUT', '.'), P.CHECKS_JSON)
    try:
        C = P.load_json(path, 'rule_change_checks', [])
    except BaseException as e:
        msg = str(e).strip().splitlines()
        say(f'p99: {path} refused: {msg[-1][:300] if msg else type(e).__name__}')
        return False
    if C is None:
        say(f'p99: {path} absent (p99 checks, measurement M4)')
        return False
    sizes = [x for x in P.DEFAULT_SIZES.split(',') if x not in C.get('sizes', {})]
    say(f'p99: {path}: valid {C.get("valid")}' + (f'; sizes not timed: {", ".join(sizes)}'
                                                     if sizes else ''))
    return ok and not miss and C.get('valid') is True and not sizes


def p123_ready():
    """The run p123 draws (its MODEL) is the run with the lowest validation score among the
    active seeds (arm_registry.select_seed)."""
    import arm_registry as REG
    import p123_example_schedule as P
    s, _ = REG.select_seed('released')
    sel = REG.PREFIX + REG.model('released', None, s)
    say(f'p123 draws {P.MODEL}; the selected run is {sel}'
        + ('' if sel == P.MODEL else ' (they differ, the figure would show another run)'))
    return sel == P.MODEL


def deploy_ready(path):
    import arm_registry as REG
    if not os.path.exists(path):
        say(f'{path} absent (manual measurement M1)')
        return False
    rec = json.load(open(path))
    sizes = rec.get('sizes', {})
    bad = [k for k, r in sizes.items() if not r.get('valid', False)]
    say(f'{path}: valid {rec.get("valid")}, environment {rec.get("environment")}, '
        f'model {rec.get("model")}, sizes {", ".join(sizes)}, torch threads '
        f'{rec.get("torch_num_threads")}')
    ok = bool(rec.get('valid')) and rec.get('environment') == 'exact' and sizes and not bad
    try:
        s, _ = REG.select_seed('released')
        want = REG.model('released', None, s)
        if not str(rec.get('model', '')).endswith(want):
            ok = False
            say(f'the record measures {rec.get("model")}, the selected run is {want}')
    except (REG.MissingResult, SystemExit) as e:
        ok = False
        say(f'run selection unavailable: {str(e).splitlines()[0][:200]}')
    return bool(ok)


# ---------------------------------------------------------------- macro files
DOCS = (('M', 'manuscript/main.tex'), ('S', 'manuscript/supplement.tex'))
RX_IN = re.compile(r'\\(?:input|InputIfFileExists)\s*\{([^}]+)\}')
RX_DEF = re.compile(r'\\(?:newcommand|renewcommand|providecommand)\*?\s*\{?\s*\\([A-Za-z@]+)'
                    r'|\\def\s*\\([A-Za-z@]+)')
RX_GEN = re.compile(r'scripts/[\w.+-]+\.py')
RX_SET = re.compile(r'environment (\w+), training seeds ([\d,]+)')
RX_HAND = re.compile(r'(?i)\btyped by hand\b')
CHAIN = (['manuscript/macros_family.tex', 'manuscript/macros_seedext.tex', 'manuscript/macros_public.tex',
          'manuscript/macros_rulecmp_public.tex', 'manuscript/macros_rulecmp_factory.tex',
          'manuscript/macros_timed.tex', 'manuscript/macros_secondary.tex',
          'manuscript/macros_environments.tex', 'manuscript/macros_sensitivity.tex',
          'manuscript/macros_aware_rules.tex', 'manuscript/macros_deploy.tex',
          'manuscript/macros_crosseval.tex', 'manuscript/macros_maskctl.tex',
          'manuscript/macros_actmask.tex', 'manuscript/macros_pubact.tex',
          'manuscript/macros_pubfam_act.tex', 'manuscript/macros_pubseedext.tex',
          'manuscript/macros_rulecmp_public_act.tex',
          'manuscript/macros_scalesens.tex', 'manuscript/macros_fiveclass.tex',
          'manuscript/macros_projectmix.tex', 'manuscript/macros_rctrained.tex',
          'manuscript/macros_divisor.tex', 'manuscript/macros_solver_selection.tex',
          'manuscript/macros_memsel.tex', 'manuscript/macros_refreplay.tex',
          'manuscript/macros_maskrule.tex', 'manuscript/macros_mask_validation.tex',
          'manuscript/macros_seedspread.tex',
          'manuscript/macros_convergence.tex', 'manuscript/macros_release.tex', 'manuscript/macros_ibound.tex',
          'manuscript/macros_rulechange_cost.tex', 'manuscript/macros_modelfacts.tex',
          'manuscript/macros_margin_pairs.tex',
          'manuscript/macros_counts.tex', 'manuscript/macros_longspec.tex',
          'manuscript/macros_budgetspec.tex', 'manuscript/macros_heldout.tex', 'manuscript/macros_perfloor.tex',
          'manuscript/macros_prodspec.tex',
          'manuscript/macros_ood.tex', 'manuscript/macros_asym.tex',
          'manuscript/macros_wrongdesc.tex', 'manuscript/macros_memetic_parallel.tex',
          'manuscript/macros_memetic_minute.tex', 'manuscript/macros_memetic_five.tex',
          'manuscript/macros_decision_rows.tex',
          'manuscript/macros_timedsets.tex', 'manuscript/macros_lbref.tex', 'manuscript/macros_bestseven.tex',
          'manuscript/macros_randomized_rules.tex', 'manuscript/macros_memetic_public.tex']
         + [f'manuscript/macros_rulecmp_public_{t}.tex'
            for t in ('sdtwenty', 'brand', 'edata', 'rdata', 'vdata')]
         + [f'manuscript/macros_rulecmp_public_act_{t}.tex'
            for t in ('sdtwenty', 'brand', 'edata', 'rdata', 'vdata')])


def strip_comments(text):
    return '\n'.join(re.sub(r'(?<!\\)%.*', '', x) for x in text.splitlines())


def inputs_of(doc):
    """Macro files a document inputs, resolved as TEXINPUTS=.:..: resolves them."""
    out = []
    d = os.path.dirname(doc)
    for name in RX_IN.findall(strip_comments(open(doc).read())):
        nm = name if name.endswith('.tex') else name + '.tex'
        if not os.path.basename(nm).startswith('macros'):
            continue
        hit = None
        for base in (d, 'manuscript'):
            p = os.path.normpath(os.path.join(base, nm))
            if os.path.exists(p):
                hit = p
                break
        out.append((nm, hit))
    return out


def defs(path):
    names = []
    for a, b in RX_DEF.findall(strip_comments(open(path, errors='replace').read())):
        names.append(a or b)
    return names


def macro_table(run_start):
    run_start = float(run_start)
    used = collections.defaultdict(set)
    absent = []
    # the sources of the manuscript are not part of the public release: a document that
    # does not exist is left out of the input column, the duplicate check and the list of
    # files no build inputs
    docs = [(key, doc) for key, doc in DOCS if os.path.exists(doc)]
    for key, doc in docs:
        for nm, p in inputs_of(doc):
            if p:
                used[p].add(key)
            else:
                absent.append(f'{doc}: {nm}')
    files = sorted(set(glob.glob('manuscript/macros*.tex') + glob.glob('manuscript/*/macros*.tex'))
                   | set(used))
    print('  file                                          modified          macros '
          'input by   generator, setting')
    for f in files:
        names = defs(f)
        head = ''.join(open(f, errors='replace').readlines()[:3])
        g = RX_GEN.search(head)
        gen = 'typed by hand' if RX_HAND.search(head) else (g.group(0) if g else '-')
        st = RX_SET.search(head)
        mt = os.path.getmtime(f)
        new = '*' if run_start and mt >= run_start else ' '
        print(f'  {f:45s} {time.strftime("%Y-%m-%d %H:%M", time.localtime(mt))}{new} '
              f'{len(names):6d}   {"".join(sorted(used.get(f, ""))) or "-":7s}  '
              f'{gen}'
              + (f', {st.group(1)}, seeds {st.group(2)}' if st else ''))
    print('  input by: M manuscript/main.tex, S manuscript/supplement.tex'
          + ('; * written by this run' if run_start else ''))
    for key, doc in DOCS:
        if (key, doc) not in docs:
            print(f'  document not found, left out of the column "input by": {doc}')
    if not os.path.exists('manuscript/build.sh'):
        print('  build script of the manuscript not found, its copy check is left out: '
              'manuscript/build.sh')
    if not docs:
        print('  no source of the manuscript is present (the public release holds none), so '
              'no input and no duplicate macro name is checked')
        return 0
    for x in absent:
        print(f'  input not found (optional inputs are skipped by LaTeX): {x}')
    # duplicate check over the files the two documents input
    where = collections.defaultdict(list)
    within = []
    for f in sorted(used):
        names = defs(f)
        c = collections.Counter(names)
        within += [f'{f}: \\{n} defined {k} times' for n, k in c.items() if k > 1]
        for n in c:
            where[n].append(f)
    dup = {n: fs for n, fs in where.items() if len(fs) > 1}
    print(f'  duplicate check over {len(used)} input macro files, {len(where)} names: '
          + ('no name is defined in two files' if not dup else
             f'{len(dup)} names defined in two or more files'))
    for n, fs in sorted(dup.items())[:40]:
        print(f'    \\{n}: {", ".join(fs)}')
    if len(dup) > 40:
        print(f'    ... and {len(dup) - 40} more')
    for x in within[:20]:
        print(f'    {x}')
    # files of this chain that no build inputs yet, against everything that is input
    pending = [f for f in CHAIN if os.path.exists(f) and f not in used]
    for f in pending:
        clash = sorted(set(defs(f)) & set(where))
        print(f'  not input by any build yet: {f} ({len(defs(f))} macros); '
              + (f'{len(clash)} names collide with input files, e.g. '
                 + ', '.join('\\' + c for c in clash[:5]) if clash else
                 'no name collides with the input files'))
    # manuscript/build.sh copies files of paper/ into manuscript/ before each build; it
    # must copy only the files that manuscript/ lacks (its function copy_absent). The build
    # script is a source of the manuscript, which the public release does not hold:
    # without it this check is left out with one line (above).
    build = open('manuscript/build.sh').read() if os.path.exists('manuscript/build.sh') else ''
    if re.search(r'\.\./paper/macros', build) and 'copy_absent' not in build:
        for f in CHAIN:
            src = os.path.join('paper', os.path.basename(f))
            if os.path.exists(f) and os.path.exists(src) and \
                    open(f, 'rb').read() != open(src, 'rb').read():
                print(f'  WARNING: manuscript/build.sh copies {src} over {f} '
                      f'(the two differ); build after removing {src} from that copy')
    return 0 if not (dup or within) else 2


def main():
    cmd, args = sys.argv[1], sys.argv[2:]
    fn = {'plan-check': plan_check, 'select-seed': select_seed, 'pubeq-state': pubeq_state,
          'pubref-state': pubref_state, 'pubact-state': pubact_state,
          'pubactfam-state': pubactfam_state, 'rulecmp-act-state': rulecmp_act_state,
          'rulecmp-ready': rulecmp_ready, 'timed-ready': timed_ready,
          'deploy-ready': deploy_ready, 'macro-table': macro_table,
          'p88-sections': p88_sections, 'p96-ready': p96_ready, 'p100-ready': p100_ready,
          'p101-ready': p101_ready, 'p102-ready': p102_ready, 'p103-ready': p103_ready,
          'p129-ready': p129_ready, 'p135-ready': p135_ready, 'p136-ready': p136_ready,
          'p104-ready': p104_ready, 'p50-ready': p50_ready, 'p98-ready': p98_ready,
          'p106-ready': p106_ready, 'p105-ready': p105_ready, 'p107-ready': p107_ready,
          'p99-ready': p99_ready, 'p123-ready': p123_ready}[cmd]
    try:
        r = fn(*args)
    except BaseException as e:            # a refusal of a checked script is a 'no'
        msg = str(e).strip().splitlines()
        say(f'{cmd}: {type(e).__name__}: {msg[0][:400] if msg else ""}')
        sys.exit(1)
    if cmd == 'macro-table':
        sys.exit(r)
    sys.exit(0 if r else 1)


main()
PYEOF

# ---------------------------------------------------------------- utilities
quote() {
  local out='' w
  for w in "$@"; do out+="$(printf '%q' "$w") "; done
  printf '%s' "${out% }"
}

pin() {               # the prefix of every command of a real run
  if [ "$DRY" = 1 ]; then printf 'taskset -c %s' "${CORE:-<CORE>}"; else printf 'taskset -c %s' "$CORE"; fi
}

helper() {            # read-only checks; pinned in a real run, unpinned in a dry run
  if [ "$DRY" = 1 ]; then
    "$PY" -c "$HELPER" "$@"
  else
    taskset -c "$CORE" "$PY" -c "$HELPER" "$@"
  fi
}

check_cmd() {         # a script's own read-only check; pinned in a real run, unpinned in a dry run
  if [ "$DRY" = 1 ]; then
    "$PY" "$@"
  else
    taskset -c "$CORE" "$PY" "$@"
  fi
}

LOG=/dev/null
run_cmd() {           # one analysis command: printed in a dry run, run and logged otherwise
  if [ "$DRY" = 1 ]; then
    printf '      %s %s\n' "$(pin)" "$(quote "$PY" "$@")"
    return 0
  fi
  printf '$ %s %s\n' "$(pin)" "$(quote "$PY" "$@")" >>"$LOG"
  local t0 rc
  t0=$(date +%s)
  taskset -c "$CORE" "$PY" "$@" >>"$LOG" 2>&1
  rc=$?
  printf '[exit %s after %s s]\n' "$rc" "$(( $(date +%s) - t0 ))" >>"$LOG"
  return "$rc"
}

# written_since <file> <marker>: the file exists and is not older than the marker that
# was touched when the step started (nanosecond modification times)
written_since() { [ -f "$1" ] && ! [ "$2" -nt "$1" ]; }

# reported_written <log> <file>: the last line "<n> macros; wrote a, b, ..." of the step's log
# names the file
reported_written() {
  local line item want
  line=$(grep -E '^[0-9]+ macros; wrote ' "$1" 2>/dev/null | tail -n 1)
  [ -n "$line" ] || return 1
  want=$(realpath -m -- "$2")
  line=${line#* macros; wrote }
  local IFS=','
  for item in $line; do
    item=${item# }
    [ "$(realpath -m -- "$item")" = "$want" ] && return 0
  done
  return 1
}

# note <text>: a line of a step's commands that is not a command; printed in a dry run and in
# the command list of a real run, written to the step's log when the step runs
note() {
  if [ "$DRY" = 1 ]; then printf '      %s\n' "$*"; else printf '# %s\n' "$*" >>"$LOG"; fi
}

# missing_of <file>...: the files that do not exist, each preceded by a space
missing_of() {
  local f out=''
  for f in "$@"; do [ -f "$f" ] || out+=" $f"; done
  printf '%s' "$out"
}

# finished_or_unselected <step>...: in a real run, none of the steps was selected in this run
# without finishing (a dry run finishes nothing, so there every step counts)
finished_or_unselected() {
  local d
  for d in "$@"; do
    if [ "$REAL_RUN" = 1 ] && [ -n "${SELECTED[$d]:-}" ] && [ "${STATUS[$d]:-}" != done ]; then
      return 1
    fi
  done
  return 0
}

FQ2_REUSE_IN=("$PACT_OUT/results/cross_eval.json" "$PACT_OUT/manuscript/macros_crosseval.tex")
FQ2_HELDOUT_IN=("$PACT_OUT/results/heldout_splits.json" "$PACT_OUT/manuscript/macros_heldout.tex"
                results/secondary_inference.json manuscript/macros_secondary.tex)
FQ5_RULE_CHANGE_IN=("$PACT_OUT/results/rule_change_cost.json" "$PACT_OUT/manuscript/macros_rulechange_cost.tex"
                    "$DEPLOY_JSON" "$PACT_OUT/paper/macros_deploy.tex")
# figure_left_out <figure>: why step figures leaves fq2_reuse, fq2_heldout or fq5_rule_change
# out, empty when it draws the figure
figure_left_out() {
  local m
  case $1 in
    fq2_reuse)
      m=$(missing_of "${FQ2_REUSE_IN[@]}")
      if [ -n "$m" ]; then echo "the record of step cross_eval is missing:$m"
      elif ! finished_or_unselected cross_eval; then
        echo "step cross_eval was selected in this run and did not finish"; fi;;
    fq2_heldout)
      m=$(missing_of "${FQ2_HELDOUT_IN[@]}")
      if [ -n "$m" ]; then echo "a record of step heldout_splits or secondary is missing:$m"
      elif ! finished_or_unselected heldout_splits secondary; then
        echo "step heldout_splits or secondary was selected in this run and did not finish"; fi;;
    fq5_rule_change)
      m=$(missing_of "${FQ5_RULE_CHANGE_IN[@]}")
      if [ -n "$m" ]; then echo "a record of step rule_change_cost or deploy is missing:$m"
      elif ! finished_or_unselected rule_change_cost deploy; then
        echo "step rule_change_cost or deploy was selected in this run and did not finish"; fi;;
  esac
}

# ---------------------------------------------------------------- the steps
title_of() {
  case $1 in
    family) echo "factory benchmark: policy against the specialists, held-out combinations, single-rule control, best dispatching rule at 10x25 and the production batches (Q1, Q2, S8, S9); rule-change pairs, the LSW specialist on all eight combinations, larger shops and 15-station cells (sections F, G, H)";;
    seed_extension) echo "the principal comparison with ten training seeds, 301 to 310: the policy against the specialist of each combination at 10x25, verdicts at the 0.5% and 1% margins, the joint verdicts and pooled rows, beside the five-seed rows of step family (table of the principal comparison; run when the extension rule applies)";;
    instance_counts) echo "the principal comparison in counts and in hours: per combination and pooled the shares of instances on which the policy is shorter, equal and longer than the specialist, the mean difference in hours with its interval, the largest loss and gain (five seeds; ten when the extension rule applies and the second five are complete)";;
    long_specialists) echo "specialists trained for the policy's number of updates, seeds $LONGSPEC_SEEDS, 10x25: the specialist of 2000 updates against the specialist of 1000 updates, the policy against both, and the training curves of both runs";;
    budget_specialists) echo "specialists that share the policy's training budget (eight of 250 updates, as many updates and episodes as the policy), seeds $BUDGETSPEC_SEEDS, 10x25: the policy against them on each combination and pooled (verdicts at 0.5% and 1%, joint verdicts, the combined verdict with Holm over the eight), beside the specialists of 1000 updates; the specialist of 250 updates against the specialist of 1000 updates; the training effort of each side";;
    heldout_splits) echo "policies trained with other held-out pairs of combinations (LS and LSW, LW and LSW), seeds $HELDOUT_SEEDS, 10x25: each against the specialist on all eight combinations and against the policy on its held-out combinations";;
    per_floor) echo "the per-floor batch, 27 modules on the factory of 25 stations: the policy against the specialists and against the best dispatching rule, beside 20 and 30 modules";;
    production_specialists) echo "the production batches, 50 and 80 modules on the factory of 25 stations: the policy against the specialist of each combination and against the best dispatching rule, and the pooled rows of 10 to 80 modules side by side";;
    ood) echo "rule magnitudes outside the training ranges at 10x25: larger asymmetric setups and denser unavailability windows, the policy against the per-instance best of four dispatching rules by the procedure of the paper (hierarchical interval, Holm over the combinations each shift reaches, combined verdict), the drawn ranges, infeasible schedules";;
    asym_setups) echo "asymmetric setups at the trained magnitude beside the larger asymmetric setups: the policy against the best of four dispatching rules (the procedure of step ood) and against the specialists";;
    wrong_descriptor) echo "a wrong constraint descriptor: makespan change, schedules the validator rejects against the true and the configured descriptor, and the configuration checks on the wrong and on six malformed descriptors";;
    example_schedule) echo "the example schedules of the selected run on the first test instance of 10x25 under N and LSW, one Gantt chart (figure fq0_example_schedule)";;
    seed_spread) echo "spread of the mean makespan across training seeds at 10x25:the policy and the specialist of each combination, smallest and largest per-seed deviation, the spread of the per-seed differences and the paired test of equal variances (seeds 301 to 305; 306 to 310 and all ten once the second five are complete)";;
    convergence) echo "whether training had stopped improving when it ended: the mean validation score of the last fifth of training against the fifth before, the share of training at which the kept checkpoint's score was first reached and the gain of the second half, for the eight specialists, the policy and the single-rule control, and the change of the mean validation score after update 1000 of the specialists trained for 2000 updates (supplement section on training and figure fs_convergence, one clause of Section 5.3)";;
    release_dates) echo "release dates at 10x25: the policy against the best dispatching rule with and without release dates, per combination and pooled, schedules checked by both validators and the makespan estimate audited (supplement table of release dates)";;
    public) echo "benchmark from public instances: policy, control and specialists on six test sets, halved lags (Q1, Q2, S11)";;
    pubact) echo "the public-family policy trained under the active mask against the policy, seeds 301 to 303 (301 to 305 once the family under the active mask holds seeds 304 and 305), SD2 10x5: difference per combination and pooled, combinations with the interval below and above zero, per-seed pooled difference, gaps to the references, validation (supplement section on the controls for the mask, one sentence of the main text)";;
    public_act) echo "benchmark from public instances under the active mask: policy, control and specialists on six test sets, halved lags, three times the training, gaps to the references, and every recipe and the best rule under the active mask against the non-delay mask (S11 under the active mask)";;
    rulecmp_public_act) echo "rule comparison in three forms under the active mask, public family: SD2 10x5 with all three arms and the five other public test sets with the policy, the dispatching rules rolled out under the active mask (counterpart of steps rulecmp_public and rulecmp_public_<tag>)";;
    public_seed_extension) echo "benchmark from public instances under the active mask with ten training seeds, 301 to 310: the policy against the specialist of each combination on pub_SD2_test_10x5+mix, verdicts at the 0.5% and 1% margins, the claims over several combinations and the pooled rows, beside the five-seed record of step public_act (an addition after the five-seed result was known)";;
    figures) echo "first result figure, fq1_specialists variants A and B (Q1); the specialists under the other combinations, fq2_reuse, the combinations held out of training, fq2_heldout, and the cost of a rule change, fq5_rule_change, when their records exist";;
    rulecmp_public) echo "rule comparison in three forms, public family, SD2 10x5, all three arms (Q3, Q4 forced-decision share)";;
    rulecmp_public_*) echo "rule comparison in three forms, public set ${PUB_SET[${1#rulecmp_public_}]}, released policy (Q3)";;
    rulecmp_factory) echo "rule comparison in three forms, factory benchmark 10x25, all three arms (Q3, Q4 forced-decision share)";;
    solver_selection) echo "selection of the CP-SAT configuration on the selection runs at 50 and 80 modules: score of the selection rule, runs with the shortest makespan, improvement over the warm start, CPU share, memory and the size of the full model (supplement tables of the solver selection and of the model size)";;
    memetic_selection) echo "selection of the configuration of the memetic algorithm on the selection runs at 50 and 80 modules: score and cells won of the three candidate configurations (supplement table of the memetic selection)";;
    timed) echo "solver and memetic algorithm with a time budget against the policy, 1 s to 600 s (Q3 time-budget figure and decision table)";;
    timed_instance_sets) echo "the timed comparison on the nine instances of its design beside all 27 evidence instances at 50 and 80 modules: the policy against the memetic algorithm and CP-SAT started from the rule schedule at 60, 300 and 600 s of the start clock, the mean difference, its interval and direction on each set, the shift between the two sets and the entries whose direction differs (supplement table of the two instance sets)";;
    memetic_parallel) echo "the memetic algorithm with eight independent runs read as one method on eight cores, against the policy and CP-SAT at 50 and 80 modules, and the spread across the eight runs; with the eight runs started from the policy's schedule complete, also the decision table on eight cores and the curves of the best of eight";;
    memetic_one_minute) echo "the memetic algorithm with a budget of 60 s, eight runs per cell from the rule schedule and eight from the policy's schedule, read one minute after the request beside the eight 600-s runs from each start, the policy and CP-SAT at 50 and 80 modules, and the one-minute rows of the decision table on eight cores with the 60-s runs in both memetic columns";;
    memetic_five_minute) echo "the memetic algorithm with a budget of 300 s, eight runs per cell from the rule schedule and eight from the policy's schedule, read five minutes after the request beside the eight 600-s runs from each start, the policy and CP-SAT at 50 and 80 modules, and the five-minute rows of the decision table on eight cores with the 300-s runs in both memetic columns";;
    decision_rows) echo "per row of the decision table (twelve rows at 50 and 80 modules, eight at the training size) the paired difference of the lowest column minus the second lowest and of the best column that uses the policy minus the best that does not, with instance-level 95% intervals, and the rows whose two lowest entries are tied";;
    lower_bound_reference) echo "a lower-bound reference per test instance and the gap of the policy, the specialists, the best dispatching rule, the memetic algorithm and CP-SAT to it, 10 to 80 modules (supplement table)";;
    randomized_rules) echo "randomized dispatching rules on eight cores within the time budget at 50 and 80 modules: a greedy randomized construction over FIFO, MOR, SPT and MWKR against the policy's greedy and sampled schedules, the best dispatching rule, the best of the eight memetic runs and CP-SAT started from the rule schedule (supplement tables of the randomized rules, one sentence of the discussion)";;
    memetic_public) echo "the memetic algorithm on the ten Brandimarte instances as published, seeds 1 to 3: the gap to the best-known makespans at 60 s and 600 s of search (supplement table of the memetic algorithm on public instances)";;
    best_of_seven) echo "the policy against the per-instance best of seven dispatching rules beside the best of the four classical rules, and the share of instances each rule wins";;
    instance_bound) echo "the makespan estimate at the empty schedule against the shortest schedule found by CP-SAT or the memetic algorithm in 600 s and against the selected policy run, 10x25, per combination and pooled (end of the section on the use of the bound during scheduling)";;
    secondary) echo "supporting training arms and the published learned scheduler (S10)";;
    margin_and_pairs) echo "what the verdict rule of the principal comparison returns for comparators known to differ from the specialist at 10x25 (the best of four dispatching rules with ten seeds beside the policy against the specialist and against the best rule, the sparse-reward arm, the specialists of 250 updates, each specialist under its seven other combinations), and the policy against two specialists chosen by the lag rule, against the specialist of N alone and against every pair under an oracle choice (seeds 301 to 305)";;
    environments) echo "the environment of the paper against the implementation released with the prior study, seeds 301 to 303, 10x25: policy, specialists, control, rules, both verdict tables";;
    sensitivity) echo "sensitivity of the factory benchmark, 22 levels one parameter at a time: policy and single-rule control against the specialists, policy against the best dispatching rule, share of single-pair decisions and setup share of machine capacity (supplement table)";;
    aware_rules) echo "three further dispatching rules (ECT, SSPT, LLF) against the policy at 10, 20 and 30 modules and on the 30-instance production sets, and the best of them against the best classical rule";;
    cross_eval) echo "every specialist under every combination against the policy, 8 x 8 matrix at 10x25, pooled off-combination rows and counts";;
    mask_controls) echo "controls for the non-delay mask: random choice among the legal pairs against SPT, the best classical rule and the policy, legal pairs per decision, window coverage and shop load at 10, 50 and 80 modules; the policy's best of ten schedules against the best of ten random rollouts when that arm exists (supplement tables, Sections 3.2 and 7.1)";;
    reference_replay) echo "what the non-delay mask costs: the reference schedules (CP-SAT on SD2 10x5, the memetic algorithm and CP-SAT of the timed campaign at 10, 50 and 80 modules) replayed under the mask, gap to the references, share of references the mask can build, and the bound of the best classical rule (supplement section on the controls for the mask, one sentence of the main text)";;
    mask_rules) echo "three admissibility rules without training (nondelay, block, active): the four classical rules, the random control and the replay of the references under each rule against nondelay (supplement section on the controls for the mask, one sentence of the main text)";;
    mask_choice_validation) echo "the choice of the mask repeated on the validation instances: the best of four dispatching rules under the active mask and under block starts against the non-delay mask on the 20 validation instances of each benchmark, beside the same comparison on the test instances (supplement section on the controls for the mask)";;
    actmask) echo "the policy trained with the active-schedule mask against the policy, seeds 301 to 303, 10x25: difference per combination and pooled, combinations with the interval above and below zero (supplement section on the controls for the mask, one sentence of the main text)";;
    scale_sensitivity) echo "windows drawn over the whole schedule at 50 and 80 modules: policy, specialists, single-rule control and rules against the stored windows (sensitivity table of the supplement)";;
    five_classes) echo "a fifth setup class at 10, 50 and 80 modules: policy, specialists and rules against four classes (sensitivity table of the supplement, discussion)";;
    project_mix) echo "single-shell project mixes at 10, 50 and 80 modules: policy against specialists and rules with reinforced-concrete or steel modules alone, beside the mixed test sets (sensitivity table of the supplement, discussion of the production claim)";;
    rc_trained) echo "the policy trained on reinforced-concrete instances alone, seeds 301 to 303, against the best of four and of seven dispatching rules, the policy and the specialists on the reinforced-concrete sets and the mixed test sets at 10, 50 and 80 modules (supplement tables of this policy, one sentence of the results on the limits of the policy)";;
    divisor) echo "batch composition and the greedy schedule: the selected run on every 10x25 test instance and combination alone, twice in one batch and with a companion at the largest processing time of the test set (supplement S2.6)";;
    deploy) echo "deployment cost and re-planning time from the record of measurement M1 (Q5)";;
    rule_change_cost) echo "what a rule change costs: training time, models, files and validation rollouts of one policy against one specialist per combination, and the time of the computed checks (table of the rule change)";;
    model_facts) echo "parameter counts of the policy network (and of the descriptor-input variant and the HGNN scheduler when their runs exist), the checkpoint size of the selected run, the median operation duration, and the check of the design constants typed by hand";;
    figure_timed) echo "time-budget figure, fq3_time_budget variants A and B and the supplement companion (Q3)";;
    figure_limits) echo "figure of the policy's limits, fq4_limits variant A: the policy against the specialists and the best dispatching rule over shop sizes and benchmark parameters (Q4)";;
  esac
}

deps_of() {
  case $1 in
    seed_extension) echo "family";;
    instance_counts) echo "family";;       # reproduces the five-seed record of step family
    per_floor) echo "family";;             # reads section H of that record
    production_specialists) echo "family per_floor";;   # reads and reproduces both records
    project_mix) echo "family sensitivity production_specialists";;   # compares with their records
    rc_trained) echo "project_mix";;       # compares its rule rows with that record
    ood) echo "family";;                   # checks its rows in distribution against FamRule
    asym_setups) echo "ood";;              # checks its rule rows against the record of p33
    best_of_seven) echo "aware_rules";;    # reproduces the summary of p72
    mask_choice_validation) echo "mask_rules";;   # reads its record
    margin_and_pairs) echo "family seed_extension budget_specialists production_specialists secondary";;   # reads their records
    seed_spread) echo "family";;
    convergence) echo "";;                 # reads the training logs only
    figures) echo "family public_act";;
    figure_timed) echo "timed";;
    timed_instance_sets) echo "timed";;
    memetic_one_minute) echo "memetic_parallel";;   # reads its record
    memetic_five_minute) echo "memetic_parallel";;   # reads its record
    decision_rows) echo "memetic_parallel memetic_one_minute memetic_five_minute";;   # reads their records
    figure_limits) echo "family per_floor production_specialists sensitivity";;
    rulecmp_public_act) echo "public_act";;
    public_seed_extension) echo "public_act";;
    rulecmp_public|rulecmp_public_*) echo "public";;
    *) echo "";;
  esac
}

pre_text() {
  case $1 in
    family)
      echo "evaluation groups principal, rules and n30 complete (scripts/eval_newseeds_plan.py check)"
      echo "the run with the lowest validation score is selectable (arm_registry.select_seed)"
      echo "every input array of the sections F, G and H exists with one row per test instance (p88 section_state; --require-sections all refuses otherwise)";;
    seed_extension)
      echo "the extension rule applies in the five-seed record $PACT_OUT/results/family_inference.json of step family: the verdict of at least one combination of the principal comparison is inconclusive at the 1% margin (scripts/seed_extension.py triggered; when the rule does not apply the step is skipped)"
      echo "all 45 runs of the second five seeds trained and evaluated (scripts/seed_extension.py check): xmask-v1-s306 to -s310 and xmask-solo-<R>-s306 to -s310, each with its checkpoint, training log with the completion line and 32786 parameters, configuration recording the exact mask, validation record, and one row per test instance in every array at 10x25"
      echo "p110 reproduces the stored five-seed section A from the arrays before it computes ten seeds, and the validation norms of the ten policy runs are equal (p110 refuses otherwise)"
      echo "the self-test of p110 on synthetic arrays passes (p110 --dry in a temporary folder; real run only)";;
    instance_counts)
      echo "the five-seed record $PACT_OUT/results/family_inference.json of step family exists, and every array of the principal comparison exists with one row per test instance for the seeds the rule of step seed_extension selects: 301 to 305, or 301 to 310 when the extension rule applies and all 45 runs of the second five seeds are complete, then also equal validation norms of the ten policy runs and the ten-seed record $PACT_OUT/results/seed_extension.json of p110 (p125 --check)"
      echo "p125 reproduces the five-seed record (and with ten seeds the ten-seed record) before it computes (p125 refuses otherwise)";;
    long_specialists)
      echo "the eight runs of 2000 updates of each seed of $LONGSPEC_SEEDS (scripts/longspec_runs.py state) trained (finished training logs) and evaluated on their combination at 10x25 (scripts/run_longspec_eval.sh), and the arrays of the specialists and of the policy of the same seeds, every array with one row per test instance (p117 --check; while a run is not trained and evaluated the step is skipped)";;
    budget_specialists)
      echo "the eight runs of 250 updates of each seed of $BUDGETSPEC_SEEDS (scripts/budgetspec_runs.py state) trained (finished training log, 32786 parameters, exact mask, the configuration of the specialist of 1000 updates except the name and max_updates) and evaluated on their combination at 10x25 (scripts/run_budgetspec_eval.sh), the arrays of the specialists and of the policy of the same seeds with one row per test instance, and the configurations of all three (p127 --check; while a run is not trained and evaluated the step is skipped)"
      echo "the policy against the specialist of 1000 updates is held against the record of the same seeds ($PACT_OUT/results/seed_extension.json of p110 for 301-310, $PACT_OUT/results/family_inference.json of p88 for 301-305); p127 reproduces it before it writes (p127 refuses otherwise)";;
    heldout_splits)
      echo "the split runs b and c of each seed of $HELDOUT_SEEDS (scripts/xsplit_runs.py state) trained (finished training log, 32786 parameters, exact mask, the configuration of the policy except the name and the training combinations) and evaluated on all eight combinations at 10x25 (scripts/run_heldout_splits_eval.sh), and the arrays of the specialists and of the policy of the same seeds, one row per test instance (p122 --check; while a run is not trained and evaluated the step is skipped)";;
    per_floor)
      echo "every array of 27x25, 20x25 and 30x25 that p118 reads exists with one row per test instance: the policy on the eight combinations, each specialist on its own, the four dispatching rules (27x25 from scripts/run_further_sets_eval.sh SET=27x25; while its evaluation is incomplete the step is skipped)"
      echo "$PACT_OUT/results/family_inference.json is the record of p88 for this setting with section H computed (p118 --check); p118 reproduces its numbers of 20x25 and 30x25 from the arrays before it writes (p118 refuses otherwise)";;
    production_specialists)
      echo "every array of 50x25n30 and 80x25n30 that p128 reads exists with one row per test instance: the policy on the eight combinations and the four dispatching rules in the standard folders, each specialist of seeds $PACT_SEEDS on its own combination in <set>_xmask_spec (scripts/run_production_specialists_eval.sh; while that evaluation is incomplete the step is skipped)"
      echo "the records $PACT_OUT/results/family_inference.json (p88, section H computed) and $PACT_OUT/results/per_floor.json (p118) of this setting, and the records results/scale_sensitivity.json and results/five_classes.json with their default-level arrays (p128 --check)"
      echo "p128 refuses unless the arrays of the policy, the specialists and the four rules on the combinations p102 and p103 hold at level default equal theirs, the means and interval ends of their policy-against-rule rows and their policy-against-specialist rows equal its own, and the rows of 10, 20, 27 and 30 modules are reproduced from the arrays";;
    ood)
      echo "the three folders p33 reads (its list SETS through arm_registry.res_dir: in distribution, larger asymmetric setups, denser unavailability windows) hold the policy's arrays of seeds $PACT_SEEDS and the arrays of FIFO, MOR, SPT and MWKR on the eight combinations, each with one row per test instance, and the policy's validator summaries, and the record of step family exists ($PACT_OUT/results/family_inference.json and $PACT_OUT/paper/macros_family.tex) (p33 --check)"
      echo "on the combinations a shift cannot reach, the arrays of the policy and of the four rules equal those in distribution bit for bit, and the values in distribution equal FamRule<R> of the record of step family (mean, interval ends, Holm-adjusted p-values, verdict) (p33 refuses otherwise); its comparison with the descriptor-input variant belongs to the earlier environment and is left out here";;
    asym_setups)
      echo "every array and validator summary p119 reads exists, every array with one row per test instance: the policy on the eight combinations and each specialist on its own (seeds $PACT_SEEDS) and the four dispatching rules in distribution, with the larger asymmetric setups and with the asymmetric setups at the trained magnitude (10x25_asymsetup; while its evaluation is incomplete the step is skipped), the policy's summaries, and the record $PACT_OUT/results/ood.json of step ood (p119 --check)"
      echo "the policy against the best dispatching rule in distribution and with the larger asymmetric setups equals the record of step ood exactly (p119 refuses otherwise)";;
    wrong_descriptor)
      echo "every rollout record of p121 rollout for the seeds $PACT_SEEDS and the combinations LSW, N, SW and LS exists with the design of this run (model, checkpoint hash, mask options, the 100 test instances) and reproduces the stored array of its run and combination; the descriptors, the base instances and the stored arrays exist (p121 --check)";;
    example_schedule)
      echo "the checkpoint and configuration of the run p123 draws, the first descriptor of N and LSW in data/FAMILY/10x25 with its base instance, that run's stored greedy arrays of N and LSW, the Times font files and pdffonts (p123 --check)"
      echo "the run p123 draws (MODEL of scripts/p123_example_schedule.py) is the run with the lowest validation score among seeds $PACT_SEEDS (arm_registry.select_seed)"
      echo "PACT_OUT is out/exact, the folder of p123's record"
      echo "both schedules reproduce the stored makespans and pass the validators (p123 stops otherwise)";;
    seed_spread)
      echo "every array of the first five seeds at 10x25 exists with one row per test instance: the policy on the eight combinations and each specialist on its own (p112 --check)"
      echo "the seed sets 306 to 310 and 301 to 310 are computed when all 45 runs of the second five seeds are trained and evaluated (scripts/seed_extension.py), and left out otherwise; the step runs again without harm once they are"
      echo "the self-test of p112 on synthetic arrays passes (p112 --dry in a temporary folder; real run only)";;
    convergence)
      echo "the training log of every run of the policy, the single-rule control and the eight specialists, seeds $PACT_SEEDS, names its run, its training combinations and its number of updates, holds the completion line and one validation line per 25 updates, and reports as its best score the minimum of its validation curve (p116 --check)"
      echo "the same holds for the training log of every specialist trained for 2000 updates, eight combinations and seeds $PACT_SEEDS (40 runs; names and logs from scripts/longspec_runs.py, as step long_specialists reads them): the step holds only when all 40 have finished (p116 --check counts them on a line of their own)";;
    release_dates)
      echo "every unit of p69 evaluate (each combination and test instance at 10x25, every policy seed and the four dispatching rules, with and without release dates) exists under test_results/FAMILY/release_p69 with the design of this run (models, checkpoint hashes, mask options, rules, draw)"
      echo "the rollouts without release dates reproduce the stored arrays of test_results/FAMILY/10x25_xmask exactly (p69 check)";;
    public)
      echo "evaluation groups public and pubsens complete (scripts/eval_newseeds_plan.py check)"
      echo "the equal-experience arm pubx-eq has every file or none (p89 refuses a partial arm)"
      echo "no counted reference of p91 contradicts the best-known table in regime N (p89 refuses otherwise); section 9 (gap to the references of pub_SD2_test_10x5+mix) is computed once all 800 references are valid, and left out with the counts until then";;
    pubact)
      echo "the arm pubx-act under the active mask has every file or none: checkpoint, training log with the completion line, validation record, configuration, and the eight arrays and the summary on pub_SD2_test_10x5+mix of seeds 301 to 303, or of seeds 301 to 305 once a file of seed 304 or 305 exists (p108 refuses a partial arm; with no file the step is skipped)"
      echo "every file of the comparator pubx-v1 (seeds 301 to 303) and all 800 references of p91 counted (p108 plan lists what is missing)";;
    public_act)
      echo "evaluation groups pubactfam and pubactsens complete (scripts/eval_newseeds_plan.py check)"
      echo "the public family under the active mask has every file or none: every principal run of seeds 301 to 305 (pubx-act, pubx-ctlact, pubx-soloact-<R>) with its checkpoint, training log with the completion line, validation record, configuration, arrays and summaries on the six test sets and the sensitivity set (p89 --family active refuses a partial family; with no file beyond the three runs of step pubact the step is skipped)"
      echo "every rule array of scripts/p109_public_active_rules.py under the active mask (test_results/PUBFAMILY/rules_act) on the six test sets and the sensitivity set"
      echo "the equal-experience recipe pubx-eqact has every file or none (section 8 under the active mask is left out with none)"
      echo "the non-delay family complete as for step public (section 10 reads it), and no counted reference of p91 contradicts the best-known table in regime N";;
    rulecmp_public_act)
      echo "the rule comparison under the active mask has every file or none: every job of 'scripts/rulecmp_plan.py --family public --mask active' (sampled schedules, traces, tie-break seeds 51 to 53, the validation records of the headline set) on the headline set and the five other public test sets is done, no input missing (p93 --mask active refuses a partial campaign; with no output of a job the step is skipped)"
      echo "the rule of form (c) of every other set is readable from the headline set's validation job under the active mask"
      echo "manuscript/macros_pubfam_act.tex exists (form (a) is cross-checked against it)";;
    public_seed_extension)
      echo "the five-seed record of step public_act exists: results/public_family_act.json and manuscript/macros_pubfam_act.tex of p89 --family active, seeds 301 to 305 (p115 recomputes its section 1 on pub_SD2_test_10x5+mix from the arrays and refuses unless every number, word and macro equals the record at the printed digits)"
      echo "all 45 runs of the second five seeds trained and evaluated (scripts/pub_seed_extension.py check): pubx-act-s306 to -s310 and pubx-soloact-<R>-s306 to -s310, each with its checkpoint, training log with the completion line, 29602 parameters and the lines of the exact and the active mask, configuration equal to that of the same recipe at seed 301 except the name and the seed, validation record, and the arrays the comparison reads on pub_SD2_test_10x5+mix with one row per test instance and the summary (when a run is not complete the step is skipped and the five-seed record stays the only reading)"
      echo "the self-test of p115 on synthetic arrays passes (p115 --dry)";;
    figures)
      echo "$PACT_OUT/results/family_inference.json, $PACT_OUT/paper/macros_family.tex, results/public_family_act.json and manuscript/macros_pubfam_act.tex exist"
      echo "when $PACT_OUT/results/seed_extension.json exists, $PACT_OUT/manuscript/macros_seedext.tex exists too (the figure then draws the ten-seed record)"
      echo "fq2_reuse is drawn when ${FQ2_REUSE_IN[*]} exist and step cross_eval, when selected, finished; fq2_heldout (full mode) when ${FQ2_HELDOUT_IN[*]} exist and steps heldout_splits and secondary, when selected, finished; fq5_rule_change when ${FQ5_RULE_CHANGE_IN[*]} exist and steps rule_change_cost and deploy, when selected, finished; otherwise each is left out with one line and the step continues";;
    rulecmp_public)
      echo "every job of scripts/rulecmp_plan.py for the public headline set is done (sampled, trace, ties, vali), no input missing"
      echo "manuscript/macros_public.tex exists (form (a) is cross-checked against it)";;
    rulecmp_public_*)
      echo "every job of scripts/rulecmp_plan.py for ${PUB_SET[${1#rulecmp_public_}]} is done, no input missing"
      echo "the rule of form (c) is readable from the headline set's validation job"
      echo "manuscript/macros_public.tex exists";;
    rulecmp_factory)
      echo "every job of scripts/rulecmp_plan.py for the factory family is done, no input missing";;
    solver_selection)
      echo "the records of the selection runs of the eight CP-SAT candidates (test_results/FAMILY/<size>_cparms/pilot: instances 000 to 002 of N, S, SW and LSW at 50x25n30 and 80x25n30) and the frozen CPU-share bars, accepted by timed_campaign_plan.select_pilot, every record with its warm start, CPU share and an objective equal to the scored value (p98 refuses otherwise)";;
    memetic_selection)
      echo "the records of the three candidate configurations of the memetic algorithm (baselines/memetic/pilot_out/r2_sel, folders A, B and C: instances 000 to 002 of N, S, SW and LSW at 50x25n30 and 80x25n30), each validated and of the design: seed, budget and the best rule's schedule as the start (p106 refuses otherwise)";;
    timed)
      echo "evaluation groups principal, rules and n30 complete (arrays and exported schedules)"
      echo "the run selection, the frozen CPU-share bars and the pilot's CP-SAT choice are available"
      echo "every cell of the campaign as designed has a valid record (timed_campaign_plan.cell_states): '$P94_CAMPAIGN_PROD' and '$P94_CAMPAIGN_TRAIN', an instance range in a token the design of its arm";;
    timed_instance_sets)
      echo "the run selection, the frozen CPU-share bars and the pilot's CP-SAT choice are available"
      echo "every cell of the campaign of the evidence record has a valid record (timed_campaign_plan.cell_states): '$P94_CAMPAIGN_STARTALL'; the campaign of the design record, '$P94_CAMPAIGN_STARTNINE', reads a subset of its instances (p94 refuses a cell without a valid record, p130 instances of the design outside the evidence instances)"
      echo "p130 refuses unless the two tagged records share the selected run, the clock, the budgets, the setting, the CP-SAT configuration and the two arms, and every entry exists in both";;
    memetic_parallel)
      echo "every cell of the eight memetic runs (seeds 1 to 8 of memetic_rule@600, instances 003 to 011 of 50x25n30 and 80x25n30, eight combinations) valid for the planner on the core sets $TIMED_SLOTS (one thread on one core of a set, cpu_share at least 0.9), the CP-SAT cells of the pilot's configuration, the rule and policy records of the request clock, and the arrays of the rules and of the selected run (p124 --check; while a cell has fewer than eight valid runs the step is skipped); the runs started from the policy's schedule are optional until every one of their cells is valid, and from then on the step also needs results/timed_inference.json of step timed";;
    memetic_one_minute)
      echo "every cell of both sets of eight 60-s runs (seeds 1 to 8 of memetic_rule@60 on the core set 16-23 and of memetic_policy@60 of the selected run on the core sets $TIMED_SLOTS, instances 003 to 011 of 50x25n30 and 80x25n30, eight combinations, 2,304 runs) valid for the planner (one thread on one core, cpu_share at least 0.9), with the configuration, code and start caches of the 600-s run of the same start, seed and cell; every input of step memetic_parallel on the core sets $TIMED_SLOTS, its runs started from the policy's schedule complete; its record $PACT_OUT/results/memetic_parallel.json, a full run on the same setting with the decision table on eight cores (p137 --check; while a cell of either set lacks a valid 60-s run the step is skipped)"
      echo "p137 refuses unless p124's smp_vs_best at 60 s, the two memetic columns of the one-minute rows of its decision table, the bold columns of its twelve rows and its count of rows whose lowest column uses the policy, recomputed on the same records, equal that record";;
    memetic_five_minute)
      echo "every cell of both sets of eight 300-s runs (seeds 1 to 8 of memetic_rule@300 on the core sets $MFIVE_SLOTS and of memetic_policy@300 of the selected run on the core sets $MFIVE_POL_SLOTS, instances 003 to 011 of 50x25n30 and 80x25n30, eight combinations, 2,304 runs) valid for the planner (one thread on one core, cpu_share at least 0.9), with the configuration, code and start caches of the 600-s run of the same start, seed and cell; every input of step memetic_parallel on the core sets $TIMED_SLOTS, its runs started from the policy's schedule complete; its record $PACT_OUT/results/memetic_parallel.json, a full run on the same setting with the decision table on eight cores (p137 --budget 300 --check; while a cell of either set lacks a valid 300-s run the step is skipped)"
      echo "p137 refuses unless p124's smp_vs_best at 300 s, the two memetic columns of the five-minute rows of its decision table and the bold columns of its twelve rows, recomputed on the same records, equal that record";;
    decision_rows)
      echo "every input of steps memetic_parallel, memetic_one_minute and memetic_five_minute on the core sets $TIMED_SLOTS, $MFIVE_SLOTS and $MFIVE_POL_SLOTS (the eight 600-s runs, the eight 60-s runs and the eight 300-s runs of each start, CP-SAT from each start, the rule and policy records of the request clock, the arrays), the records of the three steps under $PACT_OUT (full runs on the same setting with their decision-table rows) and results/timed_inference.json of step timed (a full run on the request clock with the table on the common instances of the training size), the records of the timed campaign of the training size that p94 read, and one selected training run, one B and one set of instances in the three records (p138 --check; while an input or a record is missing or differs the step is skipped)"
      echo "p138 refuses unless every row of the decision table, recomputed on the same records and cells (p124.decision_size, p94's table on the common instances), equals the row of its record (values, bold columns, cells): the rows at 60 s the record of memetic_one_minute, the rows at 300 s the record of memetic_five_minute, the rows at 600 s the record of memetic_parallel";;
    lower_bound_reference)
      echo "the descriptors and the arrays of the policy (every seed, eight combinations), of the specialists (own combination at 10x25, 20x25 and 30x25) and of the four dispatching rules with one row per descriptor, the selected run's configuration, and every cell of the 600 s tokens of the two rule-started searches valid in the campaign as launched: '$P120_CAMPAIGN_PROD', '$P120_CAMPAIGN_TRAIN' and '$P120_CAMPAIGN_SOLVER_TRAIN' (p120 --check with the same descriptions)"
      echo "the bound of the job chains and the machine-load bound never exceed a feasible makespan of their cell (p120 refuses otherwise)";;
    randomized_rules)
      echo "a valid record for each of the 144 cells of p135 (instances 003 to 011 of 50x25n30 and 80x25n30, eight combinations, eight workers, the budget of the design), each classified by the core slot of its run, and every stored input that infer reads through p124's readers: the selected training run, the CP-SAT configuration, the eight memetic runs and CP-SAT started from the rule schedule, and the policy's records of the request clock, none missing or rejected (p135-ready)"
      echo "the four rules at alpha = 0 rebuild the stored rule makespans (p135 check; not part of the precondition, since it rolls out every cell again)";;
    memetic_public)
      echo "a valid record for each of the 30 runs of p136 (Mk01 to Mk10 of pub_Brandimarte under regime N, seeds 1 to 3, 600 s from the rule schedule), classified as infer classifies them (one thread on one core of the run's cores, cpu_share at least 0.9, the cores idle before the batch, not a development run), and the best-known makespan of each instance in the table of arm_registry.PUB_SETS (p136-ready)";;
    instance_bound)
      echo "the run selection and the pilot's CP-SAT choice are available, and the greedy arrays of the selected policy run exist with one row per test instance on the eight combinations"
      echo "at least one test instance of each combination has a valid 600 s record of both searches at 10x25 (CP-SAT started from the rule schedule and the memetic algorithm; timed_campaign_plan.cell_states through p94's readers); p113 reads every instance that has both (p113 --check)";;
    secondary)
      echo "evaluation groups principal, secondary and hgnn complete (hgnn needs the learning-rate selection or HGNN_LR)";;
    margin_and_pairs)
      echo "the records $PACT_OUT/results/family_inference.json (p88, with section G), $PACT_OUT/results/seed_extension.json (p110, ten seeds), $PACT_OUT/results/production_specialists.json (p128), $PACT_OUT/results/budget_specialists.json (p127, ten seeds) and results/secondary_inference.json (p95, block sparse computed), each of the exact environment with B = 20000 and alpha = 0.05"
      echo "the arrays of FIFO, MOR, SPT and MWKR at 10x25, of the policy and the specialist of each combination with seeds 301 to 310 at 10x25, and every array p100 reads with its checks (seeds 301 to 305, the cross campaign) (p140 --check)"
      echo "p140 reproduces the five-seed principal comparison of the record of p88 and its section G, and the means of the records it reads, before it writes (p140 refuses otherwise)";;
    environments)
      echo "evaluation groups principal and rules complete (scripts/eval_newseeds_plan.py check)"
      echo "every array of both environments that p96 reads exists with one row per test instance, the stored runs of the prior implementation included (test_results/FAMILY/10x25, seeds 301 to 303)";;
    sensitivity)
      echo "every result file of p87 evaluate under test_results/FAMILY/sens_p87/results/exact (seeds 301 to 305, every level, arm and rule), records matching their files, every rule record stating the rule set masked (common_utils.masked_rule_action), the level's instances, no failed check C6 (p87 infer --check)"
      echo "the two measured variables need the counts of evaluate --trace_counts in every file; without them the step runs and reports them as not computed";;
    aware_rules)
      echo "evaluation groups principal, rules and n30 complete (scripts/eval_newseeds_plan.py check)"
      echo "the arrays Result_{ECT,SSPT,LLF}_<R>.npy of the p72 campaign in test_results/FAMILY/<size>_xmask for ${AWARE_SIZES[*]}, eight combinations";;
    best_of_seven)
      echo "the seven rule arrays and the policy arrays that p72's readers find at ${AWARE_SIZES[*]}, the summary of p72 ($PACT_OUT/results/aware_rules.json and .md) and its inference record results/aware_rules_inference.json, written by step aware_rules (p126 --check)"
      echo "p126 reproduces p72's summary at 10x25 before it computes (p126 refuses otherwise)";;
    cross_eval)
      echo "evaluation groups principal and cross complete (scripts/eval_newseeds_plan.py check)"
      echo "every array p100 reads exists with one row per test instance, with its checkpoint, training log and configuration, and the off-combination arrays the principal campaign also holds are equal (p100 load_all)";;
    mask_controls)
      echo "the cells of the random control, of the SPT trace and of the policy trace of seeds $PACT_SEEDS exist for every size and combination with one row per test instance (p101 run, run --trace_arm policy)"
      echo "the stored arrays of FIFO, MOR, SPT, MWKR and of the released policy (seeds $PACT_SEEDS) exist in test_results/FAMILY/<size>_xmask with one row per test instance"
      echo "the ten schedules of the policy (p101 run --trace_arm polten): MASKCTL_TEN=$MASKCTL_TEN; with auto no cell, or at every size the cells of every run or of the selected run only; with require one of the two at every size; the selected run determinable when they exist";;
    reference_replay)
      echo "every planned case of p105 counts: its reference record is valid in its owning campaign (the records of p91 on pub_SD2_test_10x5+mix, the memetic algorithm and CP-SAT of the timed campaign at 10x25, 50x25n30 and 80x25n30, CP-SAT with the pilot's configuration) and the case in the case folder of p105 (DEFAULT_ROOT of scripts/p105_reference_replay.py) holds this design and this reference record (p105 infer --require all refuses otherwise)"
      echo "the stored exact-mask arrays of FIFO, MOR, SPT and MWKR of every set and combination";;
    mask_rules)
      echo "every planned case of p107 counts: every set, combination and instance in the case folder of p107 (DEFAULT_ROOT of scripts/p107_mask_rule_variants.py) with this design and the reference records of p105, every reference valid (p107 infer refuses otherwise)"
      echo "the stored exact-mask arrays of FIFO, MOR, SPT and MWKR of every set and combination";;
    mask_choice_validation)
      echo "the training logs and configurations of the policy runs xmask-v1-s304 and pubx-act-s304 and of the specialist runs of SW and LSW of seed 304 of both benchmarks, with equal validation settings, the stored validation records of the public benchmark (pub_vali_dir), and the record results/mask_rule_variants.json of p107 in agreement with manuscript/macros_maskrule.tex (p141 --check)"
      echo "p141 rebuilds the validation instances and refuses unless they reproduce the MWKR norms of the training logs to the printed digit";;
    actmask)
      echo "the arrays of xmask-active and of the released policy on the seeds both have (of $PACT_SEEDS: 301 to 303) at every folder and combination the evaluation plan gives the arm (group secondary: 10x25, eight combinations), one row per test instance, not older than their checkpoints (p50 assess)"
      echo "their checkpoints, training logs with the completion line, configurations (the arm records mask_mode active, both record the exact mask options, their seed and the same training budget) and evaluation summaries that record the exact mask";;
    scale_sensitivity)
      echo "the manifests of the level win_full and every result file of p102 evaluate (seven rules; policy, specialists and single-rule control of seeds $PACT_SEEDS; levels default and win_full; 50x25n30 and 80x25n30; W, LW, SW, LSW) with one row per test instance";;
    five_classes)
      echo "the manifests of level five and every result file of p103 evaluate (seven rules; policy and specialists of seeds $PACT_SEEDS; levels default and five; 10x25, 50x25n30, 80x25n30; S, LS, SW, LSW) with one row per test instance and zero violations; --default_policy $FIVE_DEFAULT_POLICY";;
    project_mix)
      echo "the manifests of the four new single-shell sets (p129 build) and every result file of p129 evaluate (seven rules; policy and specialists of seeds $PACT_SEEDS; levels mix_rc and mix_steel; 10x25, 50x25n30, 80x25n30; eight combinations) with one row per test instance and zero violations, and the default-level arrays of the three sizes (the specialists at 50 and 80 modules from scripts/run_production_specialists_eval.sh)"
      echo "the records $PACT_OUT/results/family_inference.json (p88), $PACT_OUT/results/production_specialists.json (p128) and results/sensitivity_inference.json (p87); every 10-module unit of p87 equal to p129's on every instance, and the rows of the default level and of the 10-module levels equal to these records (p129 infer refuses otherwise)";;
    rc_trained)
      echo "the runs xmask-rc-s301 to -s303 trained (checkpoint and completion line of the training log, 32786 parameters, the configuration of the policy of the same seed except the training mix), the stored mix_rc results of p129 (policy, specialists, seven rules), the default-level arrays of the three sizes, the sets mix_rc of p129 built, and every result file of the design in the results folder of p131 (p131 check)"
      echo "the rollout of the policy's seed 301 by this path equals the stored greedy makespans, no training or validation instance equals a test instance, and the mean makespans of the seven rules equal the record results/project_mix.json of step project_mix (p131 infer refuses otherwise)";;
    divisor)
      echo "the run with the lowest validation score among seeds $PACT_SEEDS is selectable (arm_registry.select_seed)"
      echo "the cells of p104 run for that run exist for all eight combinations with the 100 test instances, not marked as a development run";;
    deploy)
      echo "$DEPLOY_JSON exists, is valid at every size (cpu_share rule), was measured in the exact environment on the selected run";;
    rule_change_cost)
      echo "one usable training-cost record of the policy and of each specialist under train_log/training_cost (scripts/run_training_cost.sh, measurement M3; valid, not a stub, of this environment, one per arm and combination)"
      echo "the record of p99 checks $PACT_OUT/results/rule_change_checks.json is valid at 10x25, 50x25n30 and 80x25n30 (measurement M4)"
      echo "the checkpoints, configurations and validation records of the selected runs that p99 count reads (p99 count refuses otherwise)";;
    model_facts)
      echo "the training logs and configurations of every principal run of seeds $PACT_SEEDS, the checkpoint of the selected policy run and the test descriptors of data/FAMILY/10x25 (p111 --check)"
      echo "the design constants typed by hand equal the values in force (p111 stops otherwise)";;
    figure_timed)
      echo "results/timed_curves.csv and results/timed_inference.json of one full p94 run exist"
      echo "$DEPLOY_JSON gives the policy's time at 50 and 80 modules; without it the figure is drawn without that mark";;
    figure_limits)
      echo "the records the figure reads in full mode exist: $PACT_OUT/results/family_inference.json and $PACT_OUT/paper/macros_family.tex (p88), $PACT_OUT/results/per_floor.json and $PACT_OUT/manuscript/macros_perfloor.tex (p118), $PACT_OUT/results/production_specialists.json and $PACT_OUT/manuscript/macros_prodspec.tex (p128), results/sensitivity_inference.json and manuscript/macros_sensitivity.tex (p87 infer)"
      echo "all six sizes and all 22 levels are in them (the figure script refuses otherwise and names what is missing)";;
  esac
}

check_of() {
  local ok=0
  case $1 in
    family)
      helper plan-check principal rules n30 || ok=1
      helper select-seed || ok=1
      helper p88-sections || ok=1;;
    seed_extension)
      check_cmd scripts/seed_extension.py triggered | sed 's/^/      /'
      [ "${PIPESTATUS[0]}" = 0 ] || ok=1
      check_cmd scripts/seed_extension.py check | sed 's/^/      /'
      [ "${PIPESTATUS[0]}" = 0 ] || ok=1
      if [ "$DRY" = 1 ]; then
        echo "      self-test of p110 on synthetic arrays: run in a real run only"
      else
        check_cmd scripts/p110_seed_extension.py --dry | sed 's/^/      /'
        [ "${PIPESTATUS[0]}" = 0 ] || ok=1
      fi;;
    instance_counts)
      check_cmd scripts/p125_instance_counts.py --check | sed 's/^/      /'
      [ "${PIPESTATUS[0]}" = 0 ] || ok=1;;
    long_specialists)
      check_cmd scripts/p117_long_specialists.py --check --pact-seeds "$LONGSPEC_SEEDS" | sed 's/^/      /'
      [ "${PIPESTATUS[0]}" = 0 ] || ok=1;;
    budget_specialists)
      check_cmd scripts/p127_budget_specialists.py --check --seeds "$BUDGETSPEC_SEEDS" | sed 's/^/      /'
      [ "${PIPESTATUS[0]}" = 0 ] || ok=1;;
    heldout_splits)
      check_cmd scripts/p122_heldout_splits.py --check --pact-seeds "$HELDOUT_SEEDS" | sed 's/^/      /'
      [ "${PIPESTATUS[0]}" = 0 ] || ok=1;;
    per_floor)
      check_cmd scripts/p118_per_floor.py --check | sed 's/^/      /'
      [ "${PIPESTATUS[0]}" = 0 ] || ok=1;;
    production_specialists)
      check_cmd scripts/p128_production_specialists.py --check | sed 's/^/      /'
      [ "${PIPESTATUS[0]}" = 0 ] || ok=1;;
    ood)
      check_cmd scripts/p33_ood_table.py --check | sed 's/^/      /'
      [ "${PIPESTATUS[0]}" = 0 ] || ok=1;;
    asym_setups)
      check_cmd scripts/p119_asym_setups.py --check | sed 's/^/      /'
      [ "${PIPESTATUS[0]}" = 0 ] || ok=1;;
    wrong_descriptor)
      check_cmd scripts/p121_wrong_descriptor.py --check | sed 's/^/      /'
      [ "${PIPESTATUS[0]}" = 0 ] || ok=1;;
    example_schedule)
      check_cmd scripts/p123_example_schedule.py --check | sed 's/^/      /'
      [ "${PIPESTATUS[0]}" = 0 ] || ok=1
      helper p123-ready || ok=1
      if [ "$(realpath -m -- "$PACT_OUT")" != "$(realpath -m -- out/exact)" ]; then
        echo "      PACT_OUT=$PACT_OUT: p123 writes its record to out/exact/results/ whatever PACT_OUT is; the step runs with PACT_OUT=out/exact only"
        ok=1
      fi;;
    seed_spread)
      check_cmd scripts/p112_seed_spread.py --check | sed 's/^/      /'
      [ "${PIPESTATUS[0]}" = 0 ] || ok=1
      if [ "$DRY" = 1 ]; then
        echo "      self-test of p112 on synthetic arrays: run in a real run only"
      else
        check_cmd scripts/p112_seed_spread.py --dry | sed 's/^/      /'
        [ "${PIPESTATUS[0]}" = 0 ] || ok=1
      fi;;
    convergence)
      check_cmd scripts/p116_convergence.py --check | sed 's/^/      /'
      [ "${PIPESTATUS[0]}" = 0 ] || ok=1;;
    release_dates)
      check_cmd scripts/p69_release_dates.py check | sed 's/^/      /'
      [ "${PIPESTATUS[0]}" = 0 ] || ok=1;;
    public)
      helper plan-check public pubsens || ok=1
      helper pubeq-state || ok=1
      helper pubref-state || ok=1;;
    pubact)
      helper pubact-state || ok=1;;
    public_act)
      helper pubactfam-state || ok=1
      if [ "$ok" = 0 ]; then
        helper plan-check pubactfam pubactsens || ok=1
        helper plan-check public pubsens || ok=1
        helper pubref-state || ok=1
      fi;;
    rulecmp_public_act)
      helper rulecmp-act-state || ok=1
      [ -f manuscript/macros_pubfam_act.tex ] || { echo "      missing manuscript/macros_pubfam_act.tex"; ok=1; };;
    public_seed_extension)
      local f
      for f in results/public_family_act.json manuscript/macros_pubfam_act.tex; do
        [ -f "$f" ] || { echo "      missing $f"; ok=1; }
      done
      check_cmd scripts/pub_seed_extension.py check | sed 's/^/      /'
      [ "${PIPESTATUS[0]}" = 0 ] || ok=1
      if [ "$DRY" = 1 ]; then
        echo "      self-test of p115 on synthetic arrays: run in a real run only"
      else
        check_cmd scripts/p115_public_seed_extension.py --dry | sed 's/^/      /'
        [ "${PIPESTATUS[0]}" = 0 ] || ok=1
      fi;;
    figures)
      local f
      for f in "$PACT_OUT/results/family_inference.json" "$PACT_OUT/paper/macros_family.tex" \
               results/public_family_act.json manuscript/macros_pubfam_act.tex; do
        if [ ! -f "$f" ]; then echo "      missing $f"; ok=1; fi
      done
      if [ -f "$PACT_OUT/results/seed_extension.json" ] && [ ! -f "$PACT_OUT/manuscript/macros_seedext.tex" ]; then
        echo "      missing $PACT_OUT/manuscript/macros_seedext.tex (the ten-seed record exists without its macros)"; ok=1
      fi;;
    rulecmp_public)
      helper rulecmp-ready public || ok=1
      [ -f manuscript/macros_public.tex ] || { echo "      missing manuscript/macros_public.tex"; ok=1; };;
    rulecmp_public_*)
      helper rulecmp-ready public "${PUB_SET[${1#rulecmp_public_}]}" || ok=1
      [ -f manuscript/macros_public.tex ] || { echo "      missing manuscript/macros_public.tex"; ok=1; };;
    rulecmp_factory)
      helper rulecmp-ready factory || ok=1;;
    solver_selection)
      helper p98-ready || ok=1;;
    memetic_selection)
      helper p106-ready || ok=1;;
    timed)
      helper plan-check principal rules n30 || ok=1
      helper timed-ready "$P94_CAMPAIGN_PROD" "$P94_CAMPAIGN_TRAIN" || ok=1;;
    timed_instance_sets)
      helper timed-ready "$P94_CAMPAIGN_STARTALL" || ok=1;;
    memetic_parallel)
      check_cmd scripts/p124_memetic_parallel.py --check --slots "${TIMED_SLOT_LIST[@]}" | sed 's/^/      /'
      [ "${PIPESTATUS[0]}" = 0 ] || ok=1;;
    memetic_one_minute)
      check_cmd scripts/p137_memetic_one_minute.py --check --ref-slots "${TIMED_SLOT_LIST[@]}" \
                --pol-slots "${TIMED_SLOT_LIST[@]}" | sed 's/^/      /'
      [ "${PIPESTATUS[0]}" = 0 ] || ok=1;;
    memetic_five_minute)
      check_cmd scripts/p137_memetic_one_minute.py --budget 300 --check --slots "${MFIVE_SLOT_LIST[@]}" \
                --pol-slots "${MFIVE_POL_SLOT_LIST[@]}" --ref-slots "${TIMED_SLOT_LIST[@]}" | sed 's/^/      /'
      [ "${PIPESTATUS[0]}" = 0 ] || ok=1;;
    decision_rows)
      check_cmd scripts/p138_decision_rows.py --check --ref-slots "${TIMED_SLOT_LIST[@]}" --pol-slots "${TIMED_SLOT_LIST[@]}" --five-slots "${MFIVE_SLOT_LIST[@]}" --five-pol-slots "${MFIVE_POL_SLOT_LIST[@]}" | sed 's/^/      /'
      [ "${PIPESTATUS[0]}" = 0 ] || ok=1;;
    lower_bound_reference)
      check_cmd scripts/p120_lower_bound_reference.py --check --campaign "$P120_CAMPAIGN_PROD" \
                --campaign "$P120_CAMPAIGN_TRAIN" --campaign "$P120_CAMPAIGN_SOLVER_TRAIN" | sed 's/^/      /'
      [ "${PIPESTATUS[0]}" = 0 ] || ok=1;;
    randomized_rules)
      helper p135-ready || ok=1;;
    memetic_public)
      helper p136-ready || ok=1;;
    instance_bound)
      check_cmd scripts/p113_instance_bound.py --check | sed 's/^/      /'
      [ "${PIPESTATUS[0]}" = 0 ] || ok=1;;
    secondary)
      helper plan-check principal secondary hgnn || ok=1;;
    margin_and_pairs)
      check_cmd scripts/p140_margin_and_pairs.py --check | sed 's/^/      /'
      [ "${PIPESTATUS[0]}" = 0 ] || ok=1;;
    environments)
      helper plan-check principal rules || ok=1
      helper p96-ready || ok=1;;
    sensitivity)
      check_cmd scripts/p87_sensitivity.py infer --check "${SENS_ARGS[@]}" | sed 's/^/      /'
      [ "${PIPESTATUS[0]}" = 0 ] || ok=1;;
    aware_rules)
      helper plan-check principal rules n30 || ok=1
      local sz r g n=0 miss=()
      for sz in "${AWARE_SIZES[@]}"; do for r in ECT SSPT LLF; do for g in N L S W LS LW SW LSW; do
        n=$((n + 1))
        [ -f "test_results/FAMILY/${sz}_xmask/Result_${r}_${g}.npy" ] || miss+=("test_results/FAMILY/${sz}_xmask/Result_${r}_${g}.npy")
      done; done; done
      echo "      p72 campaign: $((n - ${#miss[@]})) of $n rule arrays present"
      if [ "${#miss[@]}" != 0 ]; then
        printf '      missing %s\n' "${miss[@]:0:6}"
        [ "${#miss[@]}" -gt 6 ] && echo "      ... and $(( ${#miss[@]} - 6 )) more"
        ok=1
      fi;;
    best_of_seven)
      check_cmd scripts/p126_best_of_seven.py --check | sed 's/^/      /'
      [ "${PIPESTATUS[0]}" = 0 ] || ok=1;;
    cross_eval)
      helper plan-check principal cross || ok=1
      helper p100-ready || ok=1;;
    mask_controls)
      helper p101-ready "$MASKCTL_TEN" || ok=1;;
    reference_replay)
      helper p105-ready || ok=1;;
    mask_rules)
      helper p107-ready || ok=1;;
    mask_choice_validation)
      check_cmd scripts/p141_mask_choice_validation.py --check | sed 's/^/      /'
      [ "${PIPESTATUS[0]}" = 0 ] || ok=1;;
    actmask)
      helper p50-ready || ok=1;;
    scale_sensitivity)
      helper p102-ready || ok=1;;
    five_classes)
      helper p103-ready "$FIVE_DEFAULT_POLICY" || ok=1;;
    project_mix)
      helper p129-ready || ok=1;;
    rc_trained)
      check_cmd scripts/p131_rc_trained_policy.py check | sed 's/^/      /'
      [ "${PIPESTATUS[0]}" = 0 ] || ok=1;;
    divisor)
      helper p104-ready || ok=1;;
    deploy)
      helper deploy-ready "$DEPLOY_JSON" || ok=1;;
    rule_change_cost)
      helper p99-ready || ok=1;;
    model_facts)
      check_cmd scripts/p111_model_facts.py --check | sed 's/^/      /'
      [ "${PIPESTATUS[0]}" = 0 ] || ok=1;;
    figure_timed)
      local f
      for f in results/timed_curves.csv results/timed_inference.json; do
        if [ ! -f "$f" ]; then echo "      missing $f"; ok=1; fi
      done
      [ -f "$DEPLOY_JSON" ] || echo "      note: $DEPLOY_JSON not found, the figure gets no mark for the policy's time"
      [ -f "$PACT_OUT/results/memetic_parallel_curves.csv" ] || echo "      note: $PACT_OUT/results/memetic_parallel_curves.csv not found, the memetic curves are the single run"
      [ -f "$PACT_OUT/results/memetic_one_minute.json" ] || echo "      note: $PACT_OUT/results/memetic_one_minute.json not found, the figure gets no points of the one-minute budget"
      [ -f "$PACT_OUT/results/memetic_five_minute.json" ] || echo "      note: $PACT_OUT/results/memetic_five_minute.json not found, the figure gets no points of the five-minute budget";;
    figure_limits)
      local f
      for f in "$PACT_OUT/results/family_inference.json" "$PACT_OUT/paper/macros_family.tex" \
               "$PACT_OUT/results/per_floor.json" "$PACT_OUT/manuscript/macros_perfloor.tex" \
               "$PACT_OUT/results/production_specialists.json" "$PACT_OUT/manuscript/macros_prodspec.tex" \
               results/sensitivity_inference.json manuscript/macros_sensitivity.tex; do
        if [ ! -f "$f" ]; then echo "      missing $f"; ok=1; fi
      done;;
  esac
  return "$ok"
}

cmds_of() {
  case $1 in
    family)
      run_cmd scripts/p11_equivalence.py || return 1
      run_cmd scripts/p27_seed_inference.py || return 1
      run_cmd scripts/p88_family_inference.py --n30 --require-sections all || return 1;;
    seed_extension)
      run_cmd scripts/p110_seed_extension.py || return 1;;
    instance_counts)
      run_cmd scripts/p125_instance_counts.py || return 1;;
    long_specialists)
      run_cmd scripts/p117_long_specialists.py --pact-seeds "$LONGSPEC_SEEDS" || return 1;;
    budget_specialists)
      run_cmd scripts/p127_budget_specialists.py --seeds "$BUDGETSPEC_SEEDS" || return 1;;
    heldout_splits)
      run_cmd scripts/p122_heldout_splits.py --pact-seeds "$HELDOUT_SEEDS" || return 1;;
    per_floor)
      run_cmd scripts/p118_per_floor.py || return 1;;
    production_specialists)
      run_cmd scripts/p128_production_specialists.py || return 1;;
    ood)
      run_cmd scripts/p33_ood_table.py || return 1;;
    asym_setups)
      run_cmd scripts/p119_asym_setups.py || return 1;;
    wrong_descriptor)
      run_cmd scripts/p121_wrong_descriptor.py || return 1;;
    example_schedule)
      run_cmd scripts/p123_example_schedule.py || return 1;;
    seed_spread)
      run_cmd scripts/p112_seed_spread.py || return 1;;
    convergence)
      run_cmd scripts/p116_convergence.py || return 1
      run_cmd scripts/make_figures.py --only fs_convergence || return 1;;
    release_dates)
      run_cmd scripts/p69_release_dates.py analyze || return 1;;
    public)
      run_cmd scripts/p89_public_family.py || return 1;;
    pubact)
      run_cmd scripts/p108_public_active_arm.py infer || return 1;;
    public_act)
      run_cmd scripts/p89_public_family.py --family active || return 1;;
    public_seed_extension)
      run_cmd scripts/p115_public_seed_extension.py || return 1;;
    rulecmp_public_act)
      run_cmd scripts/p93_rule_comparison.py --family public --mask active || return 1
      local t
      for t in "${PUB_TAGS[@]}"; do
        run_cmd scripts/p93_rule_comparison.py --family public --mask active --set "${PUB_SET[$t]}" || return 1
      done;;
    figures)
      run_cmd scripts/make_figures.py --only fq1 --factory-seeds auto || return 1
      local why
      why=$(figure_left_out fq2_reuse)
      if [ -z "$why" ]; then
        run_cmd scripts/make_figures.py --only fq2_reuse --crosseval "${FQ2_REUSE_IN[0]}" \
                --crosseval-macros "${FQ2_REUSE_IN[1]}" || return 1
      else
        note "fq2_reuse left out: $why"
      fi
      why=$(figure_left_out fq2_heldout)
      if [ -z "$why" ]; then
        run_cmd scripts/make_figures.py --only fq2_heldout --factory-seeds auto \
                --heldout-splits "${FQ2_HELDOUT_IN[0]}" --heldout-splits-macros "${FQ2_HELDOUT_IN[1]}" \
                --secondary "${FQ2_HELDOUT_IN[2]}" --secondary-macros "${FQ2_HELDOUT_IN[3]}" || return 1
      else
        note "fq2_heldout left out: $why"
      fi
      why=$(figure_left_out fq5_rule_change)
      if [ -z "$why" ]; then
        run_cmd scripts/make_figures.py --only fq5_rule_change \
                --rule-change "${FQ5_RULE_CHANGE_IN[0]}" --rule-change-macros "${FQ5_RULE_CHANGE_IN[1]}" \
                --deploy "${FQ5_RULE_CHANGE_IN[2]}" --deploy-macros "${FQ5_RULE_CHANGE_IN[3]}" || return 1
      else
        note "fq5_rule_change left out: $why"
      fi;;
    rulecmp_public)
      run_cmd scripts/p93_rule_comparison.py --family public || return 1;;
    rulecmp_public_*)
      run_cmd scripts/p93_rule_comparison.py --family public --set "${PUB_SET[${1#rulecmp_public_}]}" || return 1;;
    rulecmp_factory)
      run_cmd scripts/p93_rule_comparison.py --family factory || return 1;;
    solver_selection)
      run_cmd scripts/p98_solver_selection.py || return 1;;
    memetic_selection)
      run_cmd scripts/p106_memetic_selection.py || return 1;;
    timed)
      run_cmd scripts/p94_timed_inference.py --campaign "$P94_CAMPAIGN_PROD" \
              --campaign "$P94_CAMPAIGN_TRAIN" --primary 600 || return 1;;
    timed_instance_sets)
      run_cmd scripts/p94_timed_inference.py --origin start --budgets 60 300 600 --primary 600 \
              --tag startnine --campaign "$P94_CAMPAIGN_STARTNINE" || return 1
      run_cmd scripts/p94_timed_inference.py --origin start --budgets 60 300 600 --primary 600 \
              --tag startall --campaign "$P94_CAMPAIGN_STARTALL" || return 1
      run_cmd scripts/p130_timed_instance_sets.py || return 1;;
    memetic_parallel)
      run_cmd scripts/p124_memetic_parallel.py --slots "${TIMED_SLOT_LIST[@]}" || return 1;;
    memetic_one_minute)
      run_cmd scripts/p137_memetic_one_minute.py --ref-slots "${TIMED_SLOT_LIST[@]}" --pol-slots "${TIMED_SLOT_LIST[@]}" || return 1;;
    memetic_five_minute)
      run_cmd scripts/p137_memetic_one_minute.py --budget 300 --slots "${MFIVE_SLOT_LIST[@]}" --pol-slots "${MFIVE_POL_SLOT_LIST[@]}" --ref-slots "${TIMED_SLOT_LIST[@]}" || return 1;;
    decision_rows)
      run_cmd scripts/p138_decision_rows.py --ref-slots "${TIMED_SLOT_LIST[@]}" --pol-slots "${TIMED_SLOT_LIST[@]}" --five-slots "${MFIVE_SLOT_LIST[@]}" --five-pol-slots "${MFIVE_POL_SLOT_LIST[@]}" || return 1;;
    instance_bound)
      run_cmd scripts/p113_instance_bound.py || return 1;;
    lower_bound_reference)
      run_cmd scripts/p120_lower_bound_reference.py --campaign "$P120_CAMPAIGN_PROD" \
              --campaign "$P120_CAMPAIGN_TRAIN" --campaign "$P120_CAMPAIGN_SOLVER_TRAIN" || return 1;;
    randomized_rules)
      run_cmd scripts/p135_randomized_rules.py infer || return 1;;
    memetic_public)
      run_cmd scripts/p136_memetic_public_check.py infer || return 1;;
    secondary)
      if [ -n "$HGNN_LR" ]; then
        run_cmd scripts/p95_secondary_inference.py --require all --hgnn-lr "$HGNN_LR" || return 1
      else
        run_cmd scripts/p95_secondary_inference.py --require all || return 1
      fi;;
    margin_and_pairs)
      run_cmd scripts/p140_margin_and_pairs.py || return 1;;
    environments)
      run_cmd scripts/p96_two_environments.py || return 1;;
    sensitivity)
      run_cmd scripts/p87_sensitivity.py infer "${SENS_ARGS[@]}" || return 1;;
    aware_rules)
      run_cmd scripts/p72_aware_rules.py --summary_only || return 1;;
    best_of_seven)
      run_cmd scripts/p126_best_of_seven.py || return 1;;
    cross_eval)
      run_cmd scripts/p100_cross_eval.py || return 1;;
    mask_controls)
      run_cmd scripts/p101_mask_controls.py infer --trace_arms SPT,policy \
              --policy_seeds "$PACT_SEEDS" --ten_arm "$MASKCTL_TEN" \
              --ten_runs "$MASKCTL_TEN_RUNS" || return 1;;
    reference_replay)
      run_cmd scripts/p105_reference_replay.py infer || return 1;;
    mask_rules)
      run_cmd scripts/p107_mask_rule_variants.py infer || return 1;;
    mask_choice_validation)
      run_cmd scripts/p141_mask_choice_validation.py || return 1;;
    actmask)
      run_cmd scripts/p50_actmask_analysis.py --require all || return 1;;
    scale_sensitivity)
      run_cmd scripts/p102_scale_sensitivity.py infer --seeds "$PACT_SEEDS" || return 1;;
    five_classes)
      run_cmd scripts/p103_five_classes.py infer --seeds "$PACT_SEEDS" \
              --default_policy "$FIVE_DEFAULT_POLICY" || return 1;;
    project_mix)
      run_cmd scripts/p129_project_mix.py infer --seeds "$PACT_SEEDS" || return 1;;
    rc_trained)
      run_cmd scripts/p131_rc_trained_policy.py infer || return 1;;
    divisor)
      run_cmd scripts/p104_divisor_probe.py infer --seeds "$PACT_SEEDS" || return 1;;
    deploy)
      run_cmd scripts/p75_deploy_cost.py --from_json "$DEPLOY_JSON" --force || return 1;;
    rule_change_cost)
      run_cmd scripts/p99_rule_change_cost.py count || return 1
      run_cmd scripts/p99_rule_change_cost.py infer --require all || return 1;;
    model_facts)
      run_cmd scripts/p111_model_facts.py || return 1;;
    figure_timed)
      if [ -f "$PACT_OUT/results/memetic_parallel_curves.csv" ] && \
         [ -f "$PACT_OUT/results/memetic_one_minute.json" ] && \
         [ -f "$PACT_OUT/results/memetic_five_minute.json" ]; then
        run_cmd scripts/make_figures.py --only fq3 --policy-time "$DEPLOY_JSON" \
                --memetic-parallel "$PACT_OUT/results/memetic_parallel.json" \
                --memetic-minute "$PACT_OUT/results/memetic_one_minute.json" \
                --memetic-five "$PACT_OUT/results/memetic_five_minute.json" || return 1
      elif [ -f "$PACT_OUT/results/memetic_parallel_curves.csv" ] && \
         [ -f "$PACT_OUT/results/memetic_one_minute.json" ]; then
        run_cmd scripts/make_figures.py --only fq3 --policy-time "$DEPLOY_JSON" \
                --memetic-parallel "$PACT_OUT/results/memetic_parallel.json" \
                --memetic-minute "$PACT_OUT/results/memetic_one_minute.json" || return 1
      elif [ -f "$PACT_OUT/results/memetic_parallel_curves.csv" ]; then
        run_cmd scripts/make_figures.py --only fq3 --policy-time "$DEPLOY_JSON" \
                --memetic-parallel "$PACT_OUT/results/memetic_parallel.json" || return 1
      else
        run_cmd scripts/make_figures.py --only fq3 --policy-time "$DEPLOY_JSON" || return 1
      fi;;
    figure_limits)
      run_cmd scripts/make_figures.py --only fq4_limits --fq4-limits-variant A \
              --pact-env "$PACT_ENV" --pact-seeds "$PACT_SEEDS" --pact-out "$PACT_OUT" || return 1;;
  esac
  return 0
}

outs_of() {           # files each step must write; checked against its start time
  case $1 in
    family) printf '%s\n' "$PACT_OUT/paper/macros_eq.tex" "$PACT_OUT/paper/macros_seed.tex" \
              "$PACT_OUT/paper/macros_family.tex" "$PACT_OUT/results/family_inference.json" \
              "$PACT_OUT/results/family_inference.md";;
    seed_extension) printf '%s\n' "$PACT_OUT/manuscript/macros_seedext.tex" \
                      "$PACT_OUT/results/seed_extension.json" "$PACT_OUT/results/seed_extension.md";;
    instance_counts) printf '%s\n' "$PACT_OUT/manuscript/macros_counts.tex" \
                       "$PACT_OUT/results/instance_counts.json" "$PACT_OUT/results/instance_counts.md";;
    long_specialists) printf '%s\n' "$PACT_OUT/manuscript/macros_longspec.tex" \
                        "$PACT_OUT/results/long_specialists.json" "$PACT_OUT/results/long_specialists.md";;
    budget_specialists) printf '%s\n' "$PACT_OUT/manuscript/macros_budgetspec.tex" \
                          "$PACT_OUT/results/budget_specialists.json" "$PACT_OUT/results/budget_specialists.md";;
    heldout_splits) printf '%s\n' "$PACT_OUT/manuscript/macros_heldout.tex" \
                      "$PACT_OUT/results/heldout_splits.json" "$PACT_OUT/results/heldout_splits.md";;
    per_floor) printf '%s\n' "$PACT_OUT/manuscript/macros_perfloor.tex" \
                 "$PACT_OUT/results/per_floor.json" "$PACT_OUT/results/per_floor.md";;
    production_specialists) printf '%s\n' "$PACT_OUT/manuscript/macros_prodspec.tex" \
                              "$PACT_OUT/results/production_specialists.json" \
                              "$PACT_OUT/results/production_specialists.md";;
    ood) printf '%s\n' "$PACT_OUT/paper/macros_ood.tex" \
           "$PACT_OUT/results/ood.json" "$PACT_OUT/results/ood.md";;
    asym_setups) printf '%s\n' "$PACT_OUT/manuscript/macros_asym.tex" \
                   "$PACT_OUT/results/asym_setups.json" "$PACT_OUT/results/asym_setups.md";;
    wrong_descriptor) printf '%s\n' "$PACT_OUT/manuscript/macros_wrongdesc.tex" \
                        "$PACT_OUT/results/wrong_descriptor.json" "$PACT_OUT/results/wrong_descriptor.md";;
    example_schedule) printf '%s\n' out/exact/results/example_schedule.json \
                        manuscript/sections_new/figures/fq0_example_schedule.pdf \
                        manuscript/sections_new/figures/fq0_example_schedule.png \
                        manuscript/macros_example.tex;;
    seed_spread) printf '%s\n' "$PACT_OUT/manuscript/macros_seedspread.tex" \
                   "$PACT_OUT/results/seed_spread.json" "$PACT_OUT/results/seed_spread.md";;
    convergence) printf '%s\n' "$PACT_OUT/manuscript/macros_convergence.tex" \
                   "$PACT_OUT/results/convergence.json" \
                   manuscript/sections_new/figures/fs_convergence.pdf \
                   manuscript/sections_new/figures/fs_convergence.png;;
    release_dates) printf '%s\n' "$PACT_OUT/manuscript/macros_release.tex" \
                     "$PACT_OUT/manuscript/tables/release_table.tex" \
                     "$PACT_OUT/results/release_dates.json" "$PACT_OUT/results/release_dates.md";;
    public) printf '%s\n' manuscript/macros_public.tex results/public_family_inference.json \
              results/public_family_inference.md;;
    pubact) printf '%s\n' manuscript/macros_pubact.tex results/public_active_arm.json \
              results/public_active_arm.md;;
    public_act) printf '%s\n' manuscript/macros_pubfam_act.tex results/public_family_act.json \
                  results/public_family_act.md;;
    public_seed_extension) printf '%s\n' "$PACT_OUT/manuscript/macros_pubseedext.tex" \
                             "$PACT_OUT/results/public_seed_extension.json" "$PACT_OUT/results/public_seed_extension.md";;
    rulecmp_public_act) local t
                        printf '%s\n' manuscript/macros_rulecmp_public_act.tex \
                          results/rulecmp_public_act.json results/rulecmp_public_act.md
                        for t in "${PUB_TAGS[@]}"; do
                          printf '%s\n' "manuscript/macros_rulecmp_public_act_$t.tex" \
                            "results/rulecmp_public_act_$t.json" "results/rulecmp_public_act_$t.md"
                        done;;
    figures) local v
             for v in fq1_specialists_A fq1_specialists_B; do
               printf '%s\n' "manuscript/sections_new/figures/$v.pdf" "manuscript/sections_new/figures/$v.png"
             done
             for v in fq2_reuse fq2_heldout fq5_rule_change; do
               [ -z "$(figure_left_out "$v")" ] || continue
               printf '%s\n' "manuscript/sections_new/figures/$v.pdf" "manuscript/sections_new/figures/$v.png"
             done;;
    rulecmp_public) printf '%s\n' manuscript/macros_rulecmp_public.tex results/rulecmp_public.json \
                      results/rulecmp_public.md;;
    rulecmp_public_*) local t=${1#rulecmp_public_}
                      printf '%s\n' "manuscript/macros_rulecmp_public_$t.tex" \
                        "results/rulecmp_public_$t.json" "results/rulecmp_public_$t.md";;
    rulecmp_factory) printf '%s\n' manuscript/macros_rulecmp_factory.tex results/rulecmp_factory.json \
                       results/rulecmp_factory.md;;
    solver_selection) printf '%s\n' manuscript/macros_solver_selection.tex results/solver_selection.json \
                        results/solver_selection.md;;
    memetic_selection) printf '%s\n' manuscript/macros_memsel.tex results/memetic_selection.json \
                         results/memetic_selection.md;;
    timed) printf '%s\n' manuscript/macros_timed.tex results/timed_inference.json \
             results/timed_inference.md results/timed_curves.csv;;
    timed_instance_sets) printf '%s\n' results/timed_inference_startnine.json \
                           results/timed_inference_startnine.md results/timed_curves_startnine.csv \
                           results/timed_inference_startall.json results/timed_inference_startall.md \
                           results/timed_curves_startall.csv "$PACT_OUT/manuscript/macros_timedsets.tex" \
                           "$PACT_OUT/results/timed_instance_sets.json" \
                           "$PACT_OUT/results/timed_instance_sets.md";;
    memetic_parallel) printf '%s\n' "$PACT_OUT/manuscript/macros_memetic_parallel.tex" \
                        "$PACT_OUT/results/memetic_parallel.json" "$PACT_OUT/results/memetic_parallel.md";;
    memetic_one_minute) printf '%s\n' "$PACT_OUT/manuscript/macros_memetic_minute.tex" \
                          "$PACT_OUT/results/memetic_one_minute.json" "$PACT_OUT/results/memetic_one_minute.md";;
    memetic_five_minute) printf '%s\n' "$PACT_OUT/manuscript/macros_memetic_five.tex" \
                           "$PACT_OUT/results/memetic_five_minute.json" "$PACT_OUT/results/memetic_five_minute.md";;
    decision_rows) printf '%s\n' "$PACT_OUT/manuscript/macros_decision_rows.tex" \
                     "$PACT_OUT/results/decision_rows.json" "$PACT_OUT/results/decision_rows.md";;
    instance_bound) printf '%s\n' "$PACT_OUT/manuscript/macros_ibound.tex" \
                      "$PACT_OUT/results/instance_bound.json" "$PACT_OUT/results/instance_bound.md";;
    lower_bound_reference) printf '%s\n' "$PACT_OUT/manuscript/macros_lbref.tex" \
                             "$PACT_OUT/results/lower_bound_reference.json" \
                             "$PACT_OUT/results/lower_bound_reference.md";;
    randomized_rules) printf '%s\n' "$PACT_OUT/manuscript/macros_randomized_rules.tex" \
                        "$PACT_OUT/results/randomized_rules.json" "$PACT_OUT/results/randomized_rules.md";;
    memetic_public) printf '%s\n' "$PACT_OUT/manuscript/macros_memetic_public.tex" \
                      "$PACT_OUT/results/memetic_public_check.json" "$PACT_OUT/results/memetic_public_check.md";;
    secondary) printf '%s\n' manuscript/macros_secondary.tex results/secondary_inference.json \
                 results/secondary_inference.md;;
    margin_and_pairs) printf '%s\n' "$PACT_OUT/manuscript/macros_margin_pairs.tex" \
                        "$PACT_OUT/results/margin_sensitivity.json" "$PACT_OUT/results/margin_sensitivity.md" \
                        "$PACT_OUT/results/two_specialists.json" "$PACT_OUT/results/two_specialists.md";;
    environments) printf '%s\n' manuscript/macros_environments.tex results/environments.json \
                    results/environments.md;;
    sensitivity) printf '%s\n' manuscript/macros_sensitivity.tex manuscript/tables/sensitivity_table.tex \
                   results/sensitivity_inference.json results/sensitivity_inference.md \
                   results/sensitivity_inference.csv;;
    aware_rules) printf '%s\n' "$PACT_OUT/results/aware_rules.json" "$PACT_OUT/results/aware_rules.md" \
                   "$PACT_OUT/paper/macros_aware.tex" manuscript/macros_aware_rules.tex \
                   results/aware_rules_inference.json results/aware_rules_inference.md \
                   manuscript/tables/aware_rules_table.tex manuscript/tables/aware_rules_table_prod.tex;;
    best_of_seven) printf '%s\n' "$PACT_OUT/manuscript/macros_bestseven.tex" \
                     "$PACT_OUT/results/best_of_seven.json" "$PACT_OUT/results/best_of_seven.md";;
    cross_eval) printf '%s\n' "$PACT_OUT/manuscript/macros_crosseval.tex" \
                  "$PACT_OUT/results/cross_eval.json" "$PACT_OUT/results/cross_eval.md";;
    mask_controls) printf '%s\n' manuscript/macros_maskctl.tex results/mask_controls.json \
                     results/mask_controls.md;;
    reference_replay) printf '%s\n' manuscript/macros_refreplay.tex results/reference_replay.json \
                        results/reference_replay.md;;
    mask_rules) printf '%s\n' manuscript/macros_maskrule.tex results/mask_rule_variants.json \
                  results/mask_rule_variants.md;;
    mask_choice_validation) printf '%s\n' "$PACT_OUT/manuscript/macros_mask_validation.tex" \
                              "$PACT_OUT/results/mask_choice_validation.json" \
                              "$PACT_OUT/results/mask_choice_validation.md";;
    actmask) printf '%s\n' "$PACT_OUT/manuscript/macros_actmask.tex" \
               "$PACT_OUT/manuscript/tables/actmask_table.tex" \
               "$PACT_OUT/results/active_mask_arm.json" "$PACT_OUT/results/active_mask_arm.md";;
    scale_sensitivity) printf '%s\n' manuscript/macros_scalesens.tex results/scale_sensitivity.json \
                         results/scale_sensitivity.md;;
    five_classes) printf '%s\n' manuscript/macros_fiveclass.tex results/five_classes.json \
                    results/five_classes.md;;
    project_mix) printf '%s\n' manuscript/macros_projectmix.tex results/project_mix.json \
                   results/project_mix.md;;
    rc_trained) printf '%s\n' manuscript/macros_rctrained.tex results/rc_trained_policy.json \
                  results/rc_trained_policy.md;;
    divisor) printf '%s\n' manuscript/macros_divisor.tex results/divisor_probe.json \
               results/divisor_probe.md;;
    deploy) printf '%s\n' "$PACT_OUT/paper/macros_deploy.tex" "$PACT_OUT/paper/tables/deploy_table.tex";;
    rule_change_cost) printf '%s\n' "$PACT_OUT/manuscript/macros_rulechange_cost.tex" \
                        "$PACT_OUT/results/rule_change_cost.json" "$PACT_OUT/results/rule_change_cost.md" \
                        "$PACT_OUT/results/rule_change_count.json";;
    model_facts) printf '%s\n' "$PACT_OUT/manuscript/macros_modelfacts.tex" \
                   "$PACT_OUT/results/model_facts.json";;
    figure_timed) local v
                  for v in A B B_supplement; do
                    printf '%s\n' "manuscript/sections_new/figures/fq3_time_budget_$v.pdf" \
                      "manuscript/sections_new/figures/fq3_time_budget_$v.png"
                  done;;
    figure_limits) printf '%s\n' manuscript/sections_new/figures/fq4_limits_A.pdf \
                     manuscript/sections_new/figures/fq4_limits_A.png;;
  esac
}

copies_of() {         # "source destination [reported]" lines, copied after the step succeeded
  case $1 in
    family) echo "$PACT_OUT/paper/macros_family.tex manuscript/macros_family.tex";;
    seed_extension) echo "$PACT_OUT/manuscript/macros_seedext.tex manuscript/macros_seedext.tex";;
    instance_counts) echo "$PACT_OUT/manuscript/macros_counts.tex manuscript/macros_counts.tex";;
    long_specialists) echo "$PACT_OUT/manuscript/macros_longspec.tex manuscript/macros_longspec.tex";;
    budget_specialists) echo "$PACT_OUT/manuscript/macros_budgetspec.tex manuscript/macros_budgetspec.tex";;
    heldout_splits) echo "$PACT_OUT/manuscript/macros_heldout.tex manuscript/macros_heldout.tex";;
    per_floor) echo "$PACT_OUT/manuscript/macros_perfloor.tex manuscript/macros_perfloor.tex";;
    production_specialists) echo "$PACT_OUT/manuscript/macros_prodspec.tex manuscript/macros_prodspec.tex";;
    ood) echo "$PACT_OUT/paper/macros_ood.tex manuscript/macros_ood.tex";;
    asym_setups) echo "$PACT_OUT/manuscript/macros_asym.tex manuscript/macros_asym.tex";;
    wrong_descriptor) echo "$PACT_OUT/manuscript/macros_wrongdesc.tex manuscript/macros_wrongdesc.tex";;
    timed_instance_sets) echo "$PACT_OUT/manuscript/macros_timedsets.tex manuscript/macros_timedsets.tex";;
    memetic_parallel) echo "$PACT_OUT/manuscript/macros_memetic_parallel.tex manuscript/macros_memetic_parallel.tex";;
    memetic_one_minute) echo "$PACT_OUT/manuscript/macros_memetic_minute.tex manuscript/macros_memetic_minute.tex";;
    memetic_five_minute) echo "$PACT_OUT/manuscript/macros_memetic_five.tex manuscript/macros_memetic_five.tex";;
    decision_rows) echo "$PACT_OUT/manuscript/macros_decision_rows.tex manuscript/macros_decision_rows.tex";;
    lower_bound_reference) echo "$PACT_OUT/manuscript/macros_lbref.tex manuscript/macros_lbref.tex"
                           echo "$PACT_OUT/manuscript/tables/lbref_table.tex manuscript/tables/lbref_table.tex reported";;
    randomized_rules) echo "$PACT_OUT/manuscript/macros_randomized_rules.tex manuscript/macros_randomized_rules.tex";;
    memetic_public) echo "$PACT_OUT/manuscript/macros_memetic_public.tex manuscript/macros_memetic_public.tex";;
    best_of_seven) echo "$PACT_OUT/manuscript/macros_bestseven.tex manuscript/macros_bestseven.tex";;
    public_seed_extension) echo "$PACT_OUT/manuscript/macros_pubseedext.tex manuscript/macros_pubseedext.tex";;
    seed_spread) echo "$PACT_OUT/manuscript/macros_seedspread.tex manuscript/macros_seedspread.tex";;
    convergence) echo "$PACT_OUT/manuscript/macros_convergence.tex manuscript/macros_convergence.tex";;
    release_dates) echo "$PACT_OUT/manuscript/macros_release.tex manuscript/macros_release.tex"
                   echo "$PACT_OUT/manuscript/tables/release_table.tex manuscript/tables/release_table.tex";;
    instance_bound) echo "$PACT_OUT/manuscript/macros_ibound.tex manuscript/macros_ibound.tex";;
    cross_eval) echo "$PACT_OUT/manuscript/macros_crosseval.tex manuscript/macros_crosseval.tex";;
    mask_choice_validation) echo "$PACT_OUT/manuscript/macros_mask_validation.tex manuscript/macros_mask_validation.tex";;
    actmask) echo "$PACT_OUT/manuscript/macros_actmask.tex manuscript/macros_actmask.tex";;
    deploy) echo "$PACT_OUT/paper/macros_deploy.tex manuscript/macros_deploy.tex";;
    rule_change_cost) echo "$PACT_OUT/manuscript/macros_rulechange_cost.tex manuscript/macros_rulechange_cost.tex";;
    margin_and_pairs) echo "$PACT_OUT/manuscript/macros_margin_pairs.tex manuscript/macros_margin_pairs.tex";;
    model_facts) echo "$PACT_OUT/manuscript/macros_modelfacts.tex manuscript/macros_modelfacts.tex";;
  esac
}

# ---------------------------------------------------------------- run
RUN_START=$(date +%s.%N)
declare -A STATUS=()
echo "Analysis chain, $(date '+%Y-%m-%d %H:%M:%S')$([ "$DRY" = 1 ] && echo ', DRY RUN: nothing is pinned, started or written')"
# read-only: GIT_OPTIONAL_LOCKS=0 keeps git status from refreshing the index file
dirty=$(GIT_OPTIONAL_LOCKS=0 git status --porcelain --untracked-files=no 2>/dev/null | head -c1)
echo "repository commit $(git rev-parse --short HEAD 2>/dev/null || echo '-')${dirty:+ (tracked files with uncommitted changes)}"
echo "every command: PACT_ENV=$PACT_ENV PACT_SEEDS=$PACT_SEEDS PACT_OUT=$PACT_OUT OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1$([ -n "${RULECMP_K:-}" ] && echo " RULECMP_K=$RULECMP_K"), one core ($(pin)), interpreter $PY"
echo "steps selected: $(for s in "${STEPS[@]}"; do [ -n "${SELECTED[$s]:-}" ] && printf '%s ' "$s"; done)"

i=0
for s in "${STEPS[@]}"; do
  i=$((i + 1))
  [ -n "${SELECTED[$s]:-}" ] || continue
  echo
  echo "[$i/${#STEPS[@]}] $s: $(title_of "$s")"
  echo "    precondition:"
  pre_text "$s" | sed 's/^/      - /'
  dep_block=''
  for d in $(deps_of "$s"); do
    if [ -n "${SELECTED[$d]:-}" ] && [ "${STATUS[$d]:-}" != done ] && [ "$DRY" != 1 ]; then
      dep_block="$dep_block $d"
    fi
  done
  [ -n "$(deps_of "$s")" ] && echo "      - runs after: $(deps_of "$s") (skipped when one of them was selected and did not finish)"
  echo "    commands:"
  if [ "$DRY" = 1 ]; then
    cmds_of "$s"
  else
    DRY=1 cmds_of "$s"
  fi
  echo "    writes: $(outs_of "$s" | tr '\n' ' ')"
  while read -r src dst how; do
    [ -n "${src:-}" ] || continue
    if [ "${how:-}" = reported ]; then
      echo "    brings: $src -> $dst (when reported written, and then only when written by this run)"
    else
      echo "    brings: $src -> $dst (only when written by this run)"
    fi
  done < <(copies_of "$s")
  if [ -n "$dep_block" ]; then
    echo "    SKIP $s: dependency$dep_block selected in this run and not finished"
    STATUS[$s]=skipped
    continue
  fi
  echo "    precondition check:"
  check_out=$(check_of "$s" 2>&1)
  check_rc=$?
  [ -n "$check_out" ] && printf '%s\n' "$check_out"
  if [ "$DRY" = 1 ]; then
    if [ "$check_rc" = 0 ]; then echo "    holds today: YES"; STATUS[$s]=would-run
    else echo "    holds today: no"; STATUS[$s]=would-skip; fi
    continue
  fi
  if [ "$check_rc" != 0 ]; then
    echo "    SKIP $s: the precondition does not hold"
    STATUS[$s]=skipped
    continue
  fi
  mkdir -p "$LOGDIR"       # at the first step that runs: a run that skips every step writes nothing
  LOG=$LOGDIR/$s.log
  {
    echo "# step $s: $(title_of "$s")"
    echo "# started $(date '+%Y-%m-%d %H:%M:%S') on core $CORE, commit $(git rev-parse --short HEAD 2>/dev/null || echo '-')"
    echo "# PACT_ENV=$PACT_ENV PACT_SEEDS=$PACT_SEEDS PACT_OUT=$PACT_OUT"
    echo "# precondition check:"
    printf '%s\n' "$check_out" | sed 's/^/# /'
  } >"$LOG"
  marker=$LOGDIR/.$s.started
  : >"$marker"
  echo "    running, log $LOG"
  if ! cmds_of "$s"; then
    echo "    FAILED: $s (end of $LOG follows)"
    tail -n 40 "$LOG" | sed 's/^/    | /'
    exit 1
  fi
  stale=''
  while read -r f; do
    [ -n "$f" ] || continue
    written_since "$f" "$marker" || stale="$stale $f"
  done < <(outs_of "$s")
  if [ -n "$stale" ]; then
    echo "    FAILED: $s ended without writing:$stale (end of $LOG follows)"
    echo "# not written by this step:$stale" >>"$LOG"
    tail -n 40 "$LOG" | sed 's/^/    | /'
    exit 1
  fi
  while read -r src dst how; do
    [ -n "${src:-}" ] || continue
    if [ "${how:-}" = reported ] && ! reported_written "$LOG" "$src"; then
      echo "    left out: the step did not report $src as written; $dst left unchanged"
      echo "# not reported as written by this step: $src" >>"$LOG"
      continue
    fi
    if written_since "$src" "$marker"; then
      cp -f "$src" "$dst.tmp" && mv -f "$dst.tmp" "$dst" || { echo "    FAILED: copy $src -> $dst"; exit 1; }
      echo "    brought $src -> $dst ($(sha256sum "$dst" | cut -c1-12))"
      echo "# brought $src -> $dst" >>"$LOG"
    else
      echo "    FAILED: $src was not written by this step; $dst left unchanged"
      exit 1
    fi
  done < <(copies_of "$s")
  echo "# finished $(date '+%Y-%m-%d %H:%M:%S')" >>"$LOG"
  echo "    done: $s"
  STATUS[$s]=done
done

# ---------------------------------------------------------------- manual steps
cat <<EOF

MANUAL STEPS (timed measurements, never run by this script; each on a reserved core that
nothing else touches for its duration, launch note with the processes on that core):
  M1 deployment cost and re-planning time per shop size, the selected run:
      taskset -c <reserved core> env PACT_ENV=exact PACT_SEEDS=301-305 PACT_OUT=$PACT_OUT \\
        CUDA_VISIBLE_DEVICES= $PY scripts/p75_deploy_cost.py --seed select
      (one torch thread and one numpy thread by default; writes $DEPLOY_JSON and, when every
      size is valid, the macros; the step 'deploy' then regenerates them and brings them to manuscript/)
  M2 latency record of the selected run at the training size (JSON only, no macro):
      taskset -c <reserved core> env PACT_ENV=exact PACT_SEEDS=301-305 PACT_OUT=$PACT_OUT \\
        CUDA_VISIBLE_DEVICES= $PY scripts/p16_policy_latency.py --seed select
  M3 training cost: the policy and each specialist repeated with the GPU and their cores to
      themselves (the logs of the lane campaign are contended), one invocation of
      scripts/run_training_cost.sh per run (ARM, COMBO, SEED); records in
      train_log/training_cost, read by the step rule_change_cost.
  M4 time of the computed checks at a rule change:
      taskset -c <reserved core> env PACT_ENV=exact PACT_SEEDS=301-305 PACT_OUT=$PACT_OUT \\
        CUDA_VISIBLE_DEVICES= $PY scripts/p99_rule_change_cost.py checks --rule_sizes 10x25
      (writes $PACT_OUT/results/rule_change_checks.json, read by the step rule_change_cost)
RUNS THIS CHAIN WAITS FOR (not started here): scripts/run_eval_newseeds.sh (groups principal,
  secondary, n30, rules; PUBLIC=1 for public and pubsens; EVAL_GROUPS=hgnn), scripts/run_rule_comparison.sh
  (FAMILY=public for SD2 10x5 and each SET, FAMILY=factory), scripts/run_timed_campaign.sh (the
  campaign named in P120_CAMPAIGN_PROD, P120_CAMPAIGN_TRAIN and P120_CAMPAIGN_SOLVER_TRAIN, read
  by step timed on the instances of P94_CAMPAIGN_PROD and P94_CAMPAIGN_TRAIN and by
  step timed_instance_sets on those of P94_CAMPAIGN_STARTNINE and P94_CAMPAIGN_STARTALL), scripts/run_hgnn_exact_queue.sh (PHASE=panel2),
  the p87 evaluate parts printed by
  'scripts/p87_sensitivity.py plan ${SENS_ARGS[*]} --parts <K> --trace_counts' (after p87 build),
  and for the rule units, after the rule files of the earlier rule set are moved by
  'scripts/p87_sensitivity.py archive-rules --mask exact', the parts printed by
  'scripts/p87_sensitivity.py plan ${SENS_ARGS[*]} --parts <K> --trace_counts --only_rules',
  and the p72 campaign 'PACT_ENV=exact PACT_SEEDS=301-305 PACT_OUT=$PACT_OUT scripts/p72_aware_rules.py
  --cores <cores> --resume' (sizes ${AWARE_SIZES[*]}); the second five seeds for step
  seed_extension ('SEED_TIERS="306 307 308 309 310" scripts/run_exactmask_queue.sh' and
  scripts/run_seed_extension_eval.sh, both refuse while the extension rule does not apply); the evaluation group cross
  (EVAL_GROUPS=cross) for step cross_eval; the parts printed by 'scripts/p101_mask_controls.py
  plan --trace_arm policy --parts <K>' and '... plan --trace_arm polten --parts <K>' for step
  mask_controls; the evaluate parts printed by 'scripts/p69_release_dates.py plan --parts <K>'
  for step release_dates; the arm xmask-active of scripts/run_exactmask_extra_queue.sh and its jobs of
  the evaluation group secondary for step actmask; the arm pubx-act of
  'ARMS=act scripts/run_pubfamily_queue.sh' and the evaluation group pubact for step pubact;
  the arms actfam and acteq of scripts/run_pubfamily_queue.sh, the evaluation groups
  pubactfam, pubactsens and pubacteq and 'scripts/p109_public_active_rules.py run --sets <set>'
  for each public set for step public_act; 'MASK=active scripts/run_rule_comparison.sh'
  (FAMILY=public for SD2 10x5 and each SET) for step rulecmp_public_act; the second five
  seeds for step public_seed_extension ('ARMS=actfam SEED_TIERS="306 307 308 309 310"
  scripts/run_pubfamily_queue.sh' and scripts/run_public_seed_extension_eval.sh);
  'scripts/p102_scale_sensitivity.py plan --parts <K> --arms pact,spec,single'
  for step scale_sensitivity; 'scripts/p103_five_classes.py plan --parts <K> --arms pact,spec
  --default_policy $FIVE_DEFAULT_POLICY' for step five_classes; 'scripts/p129_project_mix.py
  build' and 'scripts/p129_project_mix.py plan --parts <K>' for step project_mix; 'PACT_ENV=exact
  scripts/p104_divisor_probe.py plan --parts <K>' for step divisor; the runs of 2000 updates
  and scripts/run_longspec_eval.sh for step long_specialists; the runs of 250 updates
  (scripts/run_budgetspec_queue.sh) and scripts/run_budgetspec_eval.sh for step
  budget_specialists; the split runs and
  scripts/run_heldout_splits_eval.sh for step heldout_splits; scripts/run_further_sets_eval.sh
  (SET=27x25 for step per_floor, the set 10x25_asymsetup for step asym_setups);
  scripts/run_production_specialists_eval.sh (LANE=<i>/<K> CORES=<core>) for step
  production_specialists; 'scripts/
  p121_wrong_descriptor.py rollout' for step wrong_descriptor; the seeds 2 to 8 of
  memetic_rule@600 on the instances 003 to 011 of the production batches (core sets
  $TIMED_SLOTS) for step memetic_parallel; the seeds 1 to 8 of memetic_rule@60 on the same
  cells (core set 16-23) and of memetic_policy@60 with the selected run (core sets
  $TIMED_SLOTS) for step memetic_one_minute; the seeds 1 to 8 of memetic_rule@300 on the same
  cells (core sets $MFIVE_SLOTS) and of memetic_policy@300 with the selected run (core sets
  $MFIVE_POL_SLOTS) for step memetic_five_minute.
NOT IN THIS CHAIN (earlier analysis or no script, see the report): the earlier summaries of
  the sensitivity study (p87 analyze) and of the three further rules (p76, earlier environment
  only), the public CP-SAT references (p91),
  training cost (p21) and the count of models and files (p47). The rule-change pairs, the LSW
  specialist, the larger shops and the 15-station cells are p88 sections F, G and H (step
  family); the two environments compared is step environments (p96); the sensitivity study
  under the protocol is step sensitivity (p87 infer) and the three further rules are step
  aware_rules (p72).
EOF

# ---------------------------------------------------------------- summary
echo
echo "STEP STATUS:"
for s in "${STEPS[@]}"; do
  [ -n "${SELECTED[$s]:-}" ] && printf '  %-26s %s\n' "$s" "${STATUS[$s]:-not reached}"
done

echo
echo "MACRO FILES OF manuscript/ AND DUPLICATE CHECK:"
if [ "$DRY" = 1 ]; then helper macro-table 0; else helper macro-table "$RUN_START"; fi
rc=$?
if [ "$rc" != 0 ]; then
  echo "DUPLICATE MACRO NAMES among the files the builds input (listed above)"
  exit 2
fi
exit 0
