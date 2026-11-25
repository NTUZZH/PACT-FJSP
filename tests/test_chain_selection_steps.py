"""The steps solver_selection (p98), memetic_selection (p106), reference_replay (p105 infer)
and mask_rules (p107 infer) of scripts/run_analysis.sh, and the steps seed_spread (p112),
release_dates (p69 analyze), instance_bound (p113), rule_change_cost (p99 count and infer) and
model_facts (p111).

  * the steps stand where their inputs exist: the two selections before the timed campaign
    whose configurations they select, the replay and the admissibility rules after the
    controls for the mask;
  * every case statement of the chain names each step, the outputs of a step are the outputs
    of its script, the readiness helpers are registered and the macro files are in the table
    of the chain's macro files;
  * the dry run of each step prints its command and its outputs and writes nothing;
  * the readiness helpers say no, with the reason, when the records are absent;
  * each of the five later steps is named by every case statement, writes the outputs of its
    script under $PACT_OUT and brings its macro file (and the table of release_dates) to
    manuscript/, and its dry run prints its commands and whether its precondition holds today,
    writing nothing;
  * the steps of the further analyses (p117 to p129, p131, p135, p136, p137 at 60 and 300 s, p138, p140, p141 and p33) and the
    figures fq2_reuse,
    fq2_heldout and fq5_rule_change of step figures: each step sits at its place in the order, every case
    statement names it, its command line, outputs, copies and dependencies are those of its
    script, and with an interpreter that says no to every check (a stand-in that runs nothing)
    its dry run prints the command and the precondition it checks and a real run skips it
    with one line and goes on; the two figures are drawn only when their records exist and
    their steps, when selected, finished;
  * the timed steps read the campaign as designed and step lower_bound_reference reads it as
    launched, with the chain's defaults; step figure_timed passes the record of the best of
    eight runs once its curves file exists, beside it the record of the one-minute budget
    once that exists, and beside both the record of the five-minute budget once that exists;
  * step timed_instance_sets, after step timed: the two tagged records of p94 on the start clock
    (the two searches started from the rule schedule on the nine instances of the design and on
    all 27 evidence instances, campaigns that an environment variable can replace) and p130 on
    both; every case statement names it, its outputs are those of the two scripts, it brings
    p130's macro file to manuscript/, its dry run prints the three commands and checks the
    evidence campaign with timed-ready, and a real run skips it while that check says no or
    while step timed was selected and did not finish;
  * step figure_limits draws fq4_limits after the single-shell mixes from the records the
    figure script reads, and no step draws fq4_boundary, which the manuscript does not input.

Run from the repository root:  python -m pytest tests/test_chain_selection_steps.py -q
CPU only; the dry runs read the repository and write nothing; the stand-in interpreter runs
no analysis script.
"""
import ast
import importlib
import os
import re
import shlex
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, ROOT)
CHAIN = os.path.join(ROOT, 'scripts', 'run_analysis.sh')

# step -> (script module, command line of the step, readiness helper)
STEPS = {
    'solver_selection': ('p98_solver_selection', 'scripts/p98_solver_selection.py', 'p98-ready'),
    'memetic_selection': ('p106_memetic_selection', 'scripts/p106_memetic_selection.py',
                          'p106-ready'),
    'reference_replay': ('p105_reference_replay', 'scripts/p105_reference_replay.py infer',
                         'p105-ready'),
    'mask_rules': ('p107_mask_rule_variants', 'scripts/p107_mask_rule_variants.py infer',
                   'p107-ready'),
}


def src():
    return open(CHAIN).read()


def chain_steps():
    """The step names, from the chain's refusal of an unknown step."""
    r = subprocess.run(['bash', CHAIN], cwd=ROOT, capture_output=True, text=True, timeout=120,
                       env=dict(os.environ, DRY='1', ONLY='-'))
    return re.search(r'\(steps: ([^)]*)\)', r.stdout + r.stderr).group(1).split()


def script_outputs(mod):
    """(macros, json, md) of a script, from its own table of outputs."""
    M = importlib.import_module(mod)
    if hasattr(M, 'OUT_FILES'):
        return [M.OUT_FILES['macros'], M.OUT_FILES['json'], M.OUT_FILES['md']]
    d = dict(M.OUTPUTS)
    return [d['macros'], d['json'], d['md']]


ORDER = ['family', 'seed_extension', 'instance_counts', 'long_specialists', 'budget_specialists',
         'heldout_splits', 'per_floor', 'production_specialists', 'ood', 'asym_setups',
         'wrong_descriptor', 'example_schedule', 'seed_spread',
         'convergence', 'release_dates', 'cross_eval', 'public', 'pubact', 'public_act',
         'rulecmp_public_act', 'public_seed_extension', 'rulecmp_public',
         'rulecmp_public_sdtwenty', 'rulecmp_public_brand', 'rulecmp_public_edata',
         'rulecmp_public_rdata', 'rulecmp_public_vdata', 'rulecmp_factory', 'solver_selection',
         'memetic_selection', 'timed', 'timed_instance_sets', 'memetic_parallel',
         'memetic_one_minute', 'memetic_five_minute', 'decision_rows', 'instance_bound', 'lower_bound_reference',
         'randomized_rules', 'memetic_public', 'secondary', 'margin_and_pairs', 'environments', 'sensitivity', 'aware_rules',
         'best_of_seven', 'mask_controls', 'reference_replay', 'mask_rules',
         'mask_choice_validation', 'actmask',
         'scale_sensitivity', 'five_classes', 'project_mix', 'figure_limits', 'rc_trained', 'divisor', 'deploy',
         'rule_change_cost', 'model_facts', 'figures', 'figure_timed']


def test_steps_stand_where_their_inputs_exist():
    st = chain_steps()
    assert st == ORDER
    assert len(st) == len(set(st)) == 62
    # the ten-seed comparison and the counts read family; the counts read seed_extension's rule
    assert st[:3] == ['family', 'seed_extension', 'instance_counts']
    for s in ('long_specialists', 'budget_specialists', 'heldout_splits', 'per_floor',
              'production_specialists', 'ood', 'wrong_descriptor', 'example_schedule'):
        assert st.index('family') < st.index(s) < st.index('seed_spread')
    # the specialists of 250 updates stand beside the specialists of 2000 updates
    assert st.index('budget_specialists') == st.index('long_specialists') + 1
    assert st.index('ood') < st.index('asym_setups')        # beside the out-of-range set
    # the production batches read the record of per_floor
    assert st.index('production_specialists') == st.index('per_floor') + 1
    assert st[12:15] == ['seed_spread', 'convergence', 'release_dates']
    i = st.index('timed')
    assert st[i - 2:i + 1] == ['solver_selection', 'memetic_selection', 'timed']
    # the two instance sets of the timed comparison follow the comparison of record; the
    # one-minute reading of the 60-s runs and the five-minute reading of the 300-s runs read the
    # record of memetic_parallel, and the paired differences of the rows of the decision table
    # read the records of the three
    assert st[i + 1:i + 8] == ['timed_instance_sets', 'memetic_parallel', 'memetic_one_minute',
                               'memetic_five_minute', 'decision_rows', 'instance_bound',
                               'lower_bound_reference']
    j = st.index('mask_controls')
    # the repetition on the validation instances reads the record of mask_rules
    assert st[j:j + 5] == ['mask_controls', 'reference_replay', 'mask_rules',
                           'mask_choice_validation', 'actmask']
    assert st[st.index('aware_rules') + 1] == 'best_of_seven'
    # the single-shell mixes stand beside the fifth setup class, after the specialists of
    # the production batches whose arrays they read
    assert st.index('project_mix') == st.index('five_classes') + 1
    assert st.index('production_specialists') < st.index('project_mix')
    # figures reads family, public_act, cross_eval, heldout_splits and secondary
    assert st.index('figures') > max(st.index(s) for s in ('family', 'public_act', 'cross_eval',
                                                           'heldout_splits', 'secondary'))
    # the margin and the two specialists read the records of these steps and the arrays of
    # cross_eval, and stand right after the last of them
    assert st.index('margin_and_pairs') == st.index('secondary') + 1
    assert st.index('margin_and_pairs') > max(st.index(s) for s in (
        'family', 'seed_extension', 'budget_specialists', 'production_specialists', 'cross_eval'))
    # the figure of the policy's limits reads family, per_floor, production_specialists and
    # sensitivity, and stands after the single-shell mixes
    assert st.index('figure_limits') == st.index('project_mix') + 1
    assert st.index('figure_limits') > max(st.index(s) for s in (
        'family', 'per_floor', 'production_specialists', 'sensitivity'))
    # the policy trained on reinforced-concrete instances compares its rule rows with the
    # record of the single-shell mixes
    assert st.index('rc_trained') > st.index('project_mix')
    k = st.index('deploy')
    assert st[k:] == ['deploy', 'rule_change_cost', 'model_facts', 'figures', 'figure_timed']


@pytest.mark.parametrize('step', list(STEPS))
def test_every_case_statement_names_the_step(step):
    text = src()
    for fn in ('title_of', 'pre_text', 'check_of', 'cmds_of', 'outs_of'):
        body = re.search('^' + fn + r'\(\) \{[^\n]*\n(.*?)\n\}\n', text, re.S | re.M).group(1)
        assert re.search(rf'^    {step}\)', body, re.M), fn
    mod, cmd, ready = STEPS[step]
    assert f'run_cmd {cmd} || return 1' in text
    assert re.search(rf'^    {step}\)\n      helper {ready} \|\| ok=1;;', text, re.M)
    assert f"'{ready}': {ready.replace('-', '_')}" in text
    outs = re.search(rf'^    {step}\) printf .%s\\n. ((?:[^;]|\n)*);;', text, re.M).group(1)
    assert outs.replace('\\', ' ').split() == script_outputs(mod)
    assert f"'{script_outputs(mod)[0]}'" in text           # in CHAIN of the macro table


@pytest.mark.parametrize('step', list(STEPS))
def test_dry_run_prints_the_command_and_the_outputs(step):
    mod, cmd, _ = STEPS[step]
    r = subprocess.run(['bash', CHAIN], cwd=ROOT, capture_output=True, text=True, timeout=300,
                       env=dict(os.environ, DRY='1', ONLY=step))
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    block = r.stdout.split(f'] {step}: ', 1)[1].split('\n\n', 1)[0]
    assert re.search(rf'^      taskset -c <CORE> \S+ {re.escape(cmd)}$', block, re.M), block
    writes = re.search(r'    writes: (.*)\n', block).group(1).split()
    assert writes == script_outputs(mod)
    assert re.search(r'^    holds today: (YES|no)$', block, re.M)


def chain_helper():
    """The Python helper of run_analysis.sh, its functions only."""
    code = re.search(r"read -r -d '' HELPER <<'PYEOF'\n(.*?)\nPYEOF", src(), re.S).group(1)
    code = code.rsplit('\nmain()', 1)[0]
    ns = {'__name__': 'figures_helper'}
    exec(compile(code, 'run_analysis.sh', 'exec'), ns)
    return ns


def test_memetic_readiness_says_no_without_records(tmp_path, monkeypatch, capsys):
    import p106_memetic_selection as M
    monkeypatch.setattr(M, 'DEFAULT_RECORDS', str(tmp_path / 'no_records'))
    assert chain_helper()['p106_ready']() is False
    assert 'p106: ' in capsys.readouterr().out


def test_case_report_counts_the_reasons(capsys):
    H = chain_helper()
    why = ['pub10x5/cpsat150/N/a: case not run', 'pub10x5/cpsat150/S/a: case not run',
           '10x25/memetic_rule600/N/b: reference pending (no record)']
    assert H['case_report']('p105 cases', 10, why) is False
    out = capsys.readouterr().out
    assert '7 of 10 planned cases count; 2 case not run, 1 reference pending' in out
    assert H['case_report']('p105 cases', 10, []) is True
    assert '10 of 10 planned cases count' in capsys.readouterr().out


# ------------------------------------------------------------------ the five later steps
def p99_outputs():
    import p99_rule_change_cost as P99
    return [P99.OUT_MACROS, P99.OUT_JSON, P99.OUT_MD, P99.COUNT_JSON]


def module_outputs(mod, keys):
    M = importlib.import_module(mod)
    return [M.OUT_FILES[k] for k in keys]


# step -> (commands, outputs under $PACT_OUT, files brought to manuscript/)
LATER = {
    'seed_spread': (['scripts/p112_seed_spread.py'],
                    lambda: module_outputs('p112_seed_spread', ('macros', 'json', 'md')),
                    ['manuscript/macros_seedspread.tex']),
    'release_dates': (['scripts/p69_release_dates.py analyze'],
                      lambda: ['manuscript/macros_release.tex', 'manuscript/tables/release_table.tex',
                               'results/release_dates.json', 'results/release_dates.md'],
                      ['manuscript/macros_release.tex', 'manuscript/tables/release_table.tex']),
    'instance_bound': (['scripts/p113_instance_bound.py'],
                       lambda: module_outputs('p113_instance_bound', ('macros', 'json', 'md')),
                       ['manuscript/macros_ibound.tex']),
    'rule_change_cost': (['scripts/p99_rule_change_cost.py count',
                          'scripts/p99_rule_change_cost.py infer --require all'],
                         p99_outputs, ['manuscript/macros_rulechange_cost.tex']),
    'model_facts': (['scripts/p111_model_facts.py'],
                    lambda: module_outputs('p111_model_facts', ('macros', 'json')),
                    ['manuscript/macros_modelfacts.tex']),
}


@pytest.mark.parametrize('step', list(LATER))
def test_later_step_is_declared(step):
    text = src()
    for fn in ('title_of', 'pre_text', 'check_of', 'cmds_of', 'outs_of', 'copies_of'):
        body = re.search('^' + fn + r'\(\) \{[^\n]*\n(.*?)\n\}\n', text, re.S | re.M).group(1)
        assert re.search(rf'^    {step}\)', body, re.M), fn
    cmds, outs, brings = LATER[step]
    for c in cmds:
        assert f'run_cmd {c} || return 1' in text
    got = re.search(rf'^    {step}\) printf .%s\\n. ((?:[^;]|\n)*);;', text, re.M).group(1)
    assert got.replace('\\', ' ').split() == ['"$PACT_OUT/' + p + '"' for p in outs()]
    for b in brings:
        assert f'"$PACT_OUT/{b} {b}"' in text.replace('echo "', '"')
        if b.endswith('.tex') and os.path.basename(b).startswith('macros'):
            assert f"'{b}'" in text                       # in CHAIN of the macro table
    deps = re.search(r'^deps_of\(\) \{[^\n]*\n(.*?)\n\}\n', text, re.S | re.M).group(1)
    if step == 'seed_spread':
        assert re.search(r'^    seed_spread\) echo "family";;', deps, re.M)
    else:
        assert not re.search(rf'^    {step}\)', deps, re.M)


@pytest.mark.parametrize('step', list(LATER))
def test_later_step_dry_run_skips_cleanly_and_writes_nothing(step, tmp_path):
    cmds, outs, brings = LATER[step]
    out = tmp_path / 'out'
    r = subprocess.run(['bash', CHAIN], cwd=ROOT, capture_output=True, text=True, timeout=300,
                       env=dict(os.environ, DRY='1', ONLY=step, PACT_OUT=str(out)))
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    block = r.stdout.split(f'] {step}: ', 1)[1].split('\n\n', 1)[0]
    for c in cmds:
        assert re.search(rf'^      taskset -c <CORE> \S+ {re.escape(c)}$', block, re.M), block
    writes = re.search(r'    writes: (.*)\n', block).group(1).split()
    assert writes == [f'{out}/{p}' for p in outs()]
    for b in brings:
        assert f'    brings: {out}/{b} -> {b} (only when written by this run)' in block
    assert re.search(r'^    holds today: (YES|no)$', block, re.M), block
    assert not out.exists()                                  # the checks write nothing


def test_rule_change_readiness_says_no_without_records(tmp_path, monkeypatch, capsys):
    import p99_rule_change_cost as P99
    monkeypatch.setattr(P99.TCE, 'RECORD_DIR', str(tmp_path / 'no_records'))
    monkeypatch.setenv('PACT_ENV', 'exact')
    monkeypatch.setenv('PACT_OUT', str(tmp_path / 'out'))
    assert chain_helper()['p99_ready']() is False
    text = capsys.readouterr().out
    assert 'p99: training-cost records under' in text and '0 of 9 arms' in text
    assert 'rule_change_checks.json absent' in text


# ------------------------------------------------------------------ the further analyses
def source_constant(script, name):
    """A literal constant of a script, read from its source (nothing is imported or run)."""
    path = os.path.join(ROOT, 'scripts', script)
    for node in ast.parse(open(path).read(), path).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name
                                                for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f'{script} defines no {name}')


def chain_default(var):
    return re.search(rf'^{var}=\$\{{{var}:-"?([^"}}]*)"?\}}$', src(), re.M).group(1)


PROD, TRAIN = chain_default('P120_CAMPAIGN_PROD'), chain_default('P120_CAMPAIGN_TRAIN')
SOLVER_TRAIN = '10x25 = cpsat_rule@600:0-9'
CAMPAIGNS = ['--campaign', PROD, '--campaign', TRAIN, '--campaign', SOLVER_TRAIN]
# the timed campaign as designed (steps timed and figure_timed) and as launched (step
# lower_bound_reference, the descriptions of the launches that p120 read before the design
# was written into the tokens of p94)
DESIGNED_PROD = ('50x25n30 80x25n30 = memetic_rule@600:3-11 memetic_policy@600:3-11 '
                 'cpsat_rule@600:3-11 cpsat_policy@600:3-11 policy_sample@120:3-11')
DESIGNED_TRAIN = '10x25 = memetic_rule@600 cpsat_rule@600:0-9 policy_sample@60'
LAUNCHED_PROD = ('50x25n30 80x25n30 = memetic_rule@600 memetic_policy@600 cpsat_rule@600 '
                 'cpsat_policy@600 policy_sample@120')
LAUNCHED_TRAIN = '10x25 = memetic_rule@600 policy_sample@60'


def step(script, args=(), check=('--check',), src_args='', helper=None, deps=None,
         outs=None, brings=None, reported=()):
    """The declaration of one step. outs: paths under $PACT_OUT (the script's OUT_FILES unless
    given; a path given with a leading '/' is a literal path of the repository, without it);
    brings: (path under $PACT_OUT, destination) pairs, the macro file by default; reported:
    the brought files copied only when the step reports them written."""
    if outs is None:
        of = source_constant(script, 'OUT_FILES')
        outs = [of['macros'], of['json'], of['md']]
    if brings is None:
        brings = [(outs[0], 'manuscript/' + os.path.basename(outs[0]))]
    return dict(script=f'scripts/{script}', args=list(args), check=list(check) if check else None,
                src_args=src_args, helper=helper, deps=deps, outs=outs, brings=brings,
                reported=list(reported))


P123_OUTS = ['/' + source_constant('p123_example_schedule.py', 'OUT_JSON')] + \
    ['/' + source_constant('p123_example_schedule.py', 'OUT_FIG') + e for e in ('.pdf', '.png')] + \
    ['/' + source_constant('p123_example_schedule.py', 'OUT_TEX')]
P120_FILES = source_constant('p120_lower_bound_reference.py', 'OUT_FILES')
NEW = {
    'instance_counts': step('p125_instance_counts.py', deps='family'),
    'long_specialists': step('p117_long_specialists.py', ['--pact-seeds', '301-305'],
                             ['--check', '--pact-seeds', '301-305'],
                             src_args=' --pact-seeds "$LONGSPEC_SEEDS"'),
    'budget_specialists': step('p127_budget_specialists.py', ['--seeds', '301-310'],
                               ['--check', '--seeds', '301-310'],
                               src_args=' --seeds "$BUDGETSPEC_SEEDS"'),
    'heldout_splits': step('p122_heldout_splits.py', ['--pact-seeds', '301-303'],
                           ['--check', '--pact-seeds', '301-303'],
                           src_args=' --pact-seeds "$HELDOUT_SEEDS"'),
    'per_floor': step('p118_per_floor.py', deps='family'),
    'production_specialists': step('p128_production_specialists.py', deps='family per_floor'),
    'ood': step('p33_ood_table.py', deps='family'),
    'asym_setups': step('p119_asym_setups.py', deps='ood'),
    'wrong_descriptor': step('p121_wrong_descriptor.py'),
    'example_schedule': step('p123_example_schedule.py', helper='p123-ready', outs=P123_OUTS,
                             brings=[]),
    'memetic_parallel': step('p124_memetic_parallel.py', ['--slots', '16-23', '8-15'],
                             ['--check', '--slots', '16-23', '8-15'],
                             src_args=' --slots "${TIMED_SLOT_LIST[@]}"'),
    'memetic_one_minute': step('p137_memetic_one_minute.py',
                               ['--ref-slots', '16-23', '8-15', '--pol-slots', '16-23', '8-15'],
                               ['--check', '--ref-slots', '16-23', '8-15', '--pol-slots', '16-23',
                                '8-15'],
                               src_args=' --ref-slots "${TIMED_SLOT_LIST[@]}" --pol-slots '
                                        '"${TIMED_SLOT_LIST[@]}"',
                               deps='memetic_parallel'),
    'memetic_five_minute': step(
        'p137_memetic_one_minute.py',
        ['--budget', '300', '--slots', '16-23', '--pol-slots', '8-15', '--ref-slots', '16-23',
         '8-15'],
        ['--budget', '300', '--check', '--slots', '16-23', '--pol-slots', '8-15', '--ref-slots',
         '16-23', '8-15'],
        src_args=' --budget 300 --slots "${MFIVE_SLOT_LIST[@]}" --pol-slots '
                 '"${MFIVE_POL_SLOT_LIST[@]}" --ref-slots "${TIMED_SLOT_LIST[@]}"',
        deps='memetic_parallel',
        outs=[source_constant('p137_memetic_one_minute.py', 'FIVE_OUT_FILES')[k]
              for k in ('macros', 'json', 'md')]),
    'decision_rows': step('p138_decision_rows.py',
                          ['--ref-slots', '16-23', '8-15', '--pol-slots', '16-23', '8-15',
                           '--five-slots', '16-23', '--five-pol-slots', '8-15'],
                          ['--check', '--ref-slots', '16-23', '8-15', '--pol-slots', '16-23',
                           '8-15', '--five-slots', '16-23', '--five-pol-slots', '8-15'],
                          src_args=' --ref-slots "${TIMED_SLOT_LIST[@]}" --pol-slots '
                                   '"${TIMED_SLOT_LIST[@]}" --five-slots "${MFIVE_SLOT_LIST[@]}" '
                                   '--five-pol-slots "${MFIVE_POL_SLOT_LIST[@]}"',
                          deps='memetic_parallel memetic_one_minute memetic_five_minute'),
    'lower_bound_reference': step(
        'p120_lower_bound_reference.py', CAMPAIGNS, ['--check'] + CAMPAIGNS,
        src_args=' --campaign "$P120_CAMPAIGN_PROD" \\\n              --campaign '
                 '"$P120_CAMPAIGN_TRAIN" --campaign "$P120_CAMPAIGN_SOLVER_TRAIN"',
        outs=[P120_FILES['macros'], P120_FILES['json'], P120_FILES['md']],
        brings=[(P120_FILES['macros'], 'manuscript/macros_lbref.tex'),
                (P120_FILES['table'], 'manuscript/tables/lbref_table.tex')],
        reported=['manuscript/tables/lbref_table.tex']),
    'best_of_seven': step('p126_best_of_seven.py', deps='aware_rules'),
    'margin_and_pairs': step(
        'p140_margin_and_pairs.py',
        deps='family seed_extension budget_specialists production_specialists secondary',
        outs=[source_constant('p140_margin_and_pairs.py', 'OUT_FILES')[k]
              for k in ('macros', 'margin_json', 'margin_md', 'pairs_json', 'pairs_md')]),
    'mask_choice_validation': step('p141_mask_choice_validation.py', deps='mask_rules'),
    'project_mix': step('p129_project_mix.py', ['infer', '--seeds', '301-305'], None,
                        src_args=' infer --seeds "$PACT_SEEDS"', helper='p129-ready',
                        deps='family sensitivity production_specialists',
                        outs=['/' + source_constant('p129_project_mix.py', 'OUT_FILES')[k]
                              for k in ('macros', 'json', 'md')], brings=[]),
    'randomized_rules': step('p135_randomized_rules.py', ['infer'], None, src_args=' infer',
                             helper='p135-ready'),
    'memetic_public': step('p136_memetic_public_check.py', ['infer'], None, src_args=' infer',
                           helper='p136-ready'),
    'rc_trained': step('p131_rc_trained_policy.py', ['infer'], ['check'], src_args=' infer',
                       deps='project_mix',
                       outs=['/' + source_constant('p131_rc_trained_policy.py', 'OUT_FILES')[k]
                             for k in ('macros', 'json', 'md')], brings=[]),
}


def body_of(fn, text=None):
    return re.search('^' + fn + r'\(\) \{[^\n]*\n(.*?)\n\}\n', text or src(), re.S | re.M).group(1)


def declared_outs(s, out):
    """The outputs as the chain prints them under the output root out."""
    return [p[1:] if p.startswith('/') else f'{out}/{p}' for p in NEW[s]['outs']]


def test_the_scripts_name_the_outputs_the_steps_declare():
    assert source_constant('p33_ood_table.py', 'OUT_FILES') == {
        'macros': 'paper/macros_ood.tex', 'json': 'results/ood.json', 'md': 'results/ood.md'}
    assert NEW['ood']['outs'] == ['paper/macros_ood.tex', 'results/ood.json', 'results/ood.md']
    assert NEW['ood']['brings'] == [('paper/macros_ood.tex', 'manuscript/macros_ood.tex')]
    p33 = open(os.path.join(ROOT, 'scripts', 'p33_ood_table.py')).read()
    assert "emit(outs['macros'], lines)" in p33 and 'return {k: REG.out(v) for k, v in OUT_FILES' in p33
    # p119 reads the record of p33 that step ood writes
    assert source_constant('p119_asym_setups.py', 'OUT_FILES')['macros'] == 'manuscript/macros_asym.tex'
    assert "P33_RECORD = P33.OUT_FILES['json']" in open(
        os.path.join(ROOT, 'scripts', 'p119_asym_setups.py')).read()
    assert P123_OUTS == ['/out/exact/results/example_schedule.json',
                         '/manuscript/sections_new/figures/fq0_example_schedule.pdf',
                         '/manuscript/sections_new/figures/fq0_example_schedule.png',
                         '/manuscript/macros_example.tex']
    assert P120_FILES['table'] == 'manuscript/tables/lbref_table.tex'
    assert chain_default('P120_CAMPAIGN_SOLVER_TRAIN') == SOLVER_TRAIN
    assert (PROD, TRAIN) == (LAUNCHED_PROD, LAUNCHED_TRAIN)
    assert chain_default('TIMED_SLOTS') == '16-23 8-15'
    assert (chain_default('MFIVE_SLOTS'), chain_default('MFIVE_POL_SLOTS')) == ('16-23', '8-15')
    assert source_constant('p137_memetic_one_minute.py', 'FIVE_OUT_FILES') == {
        'macros': 'manuscript/macros_memetic_five.tex', 'json': 'results/memetic_five_minute.json',
        'md': 'results/memetic_five_minute.md'}
    assert chain_default('LONGSPEC_SEEDS') == '301-305'
    assert chain_default('HELDOUT_SEEDS') == '301-303'
    assert chain_default('BUDGETSPEC_SEEDS') == '301-310'      # the ten seeds of the principal comparison
    assert source_constant('p127_budget_specialists.py', 'OUT_FILES')['macros'] == \
        'manuscript/macros_budgetspec.tex'


@pytest.mark.parametrize('s', list(NEW))
def test_new_step_is_declared(s):
    text, spec = src(), NEW[s]
    for fn in ('title_of', 'pre_text', 'check_of', 'cmds_of', 'outs_of'):
        assert re.search(rf'^    {s}\)', body_of(fn, text), re.M), fn
    assert f'run_cmd {spec["script"]}{spec["src_args"]} || return 1' in text
    if spec['check']:
        first = '--check' if '--check' in spec['check'] else spec['check'][0]   # option or subcommand
        assert f'check_cmd {spec["script"]} {first}' in body_of('check_of', text)
    if spec['helper']:
        assert f'helper {spec["helper"]} || ok=1' in body_of('check_of', text)
        assert f"'{spec['helper']}': {spec['helper'].replace('-', '_')}" in text
    got = re.search(rf'^    {s}\) printf .%s\\n. ((?:[^;]|\n)*);;', text, re.M).group(1)
    want = [p[1:] if p.startswith('/') else f'"$PACT_OUT/{p}"' for p in spec['outs']]
    assert got.replace('\\', ' ').split() == want
    copies = body_of('copies_of', text)
    if spec['brings']:
        for p, dst in spec['brings']:
            flag = ' reported' if dst in spec['reported'] else ''
            assert f'echo "$PACT_OUT/{p} {dst}{flag}"' in copies
            if os.path.basename(dst).startswith('macros'):
                assert f"'{dst}'" in text                    # in CHAIN of the macro table
    else:
        assert not re.search(rf'^    {s}\)', copies, re.M)
    deps = body_of('deps_of', text)
    if spec['deps']:
        assert re.search(rf'^    {s}\) echo "{spec["deps"]}";;', deps, re.M)
    else:
        assert not re.search(rf'^    {s}\)', deps, re.M)


# A stand-in interpreter: it records its arguments (separated by \x1f, one call per line; for
# the chain's helper the word 'helper' and the helper's arguments), says no to every check and
# to every command, and runs nothing; the macro table of the chain's summary gets a yes.
STUB = r'''#!/usr/bin/env bash
if [ "$1" = -c ]; then
  { printf 'helper'; printf '\x1f%s' "${@:3}"; printf '\n'; } >>"$STUB_LOG"
  [ "$3" = macro-table ] && exit 0
  exit 1
fi
{ printf '%s\x1f' "$@"; printf '\n'; } >>"$STUB_LOG"
echo "stand-in interpreter: nothing run"
exit 1
'''


def run_with_stub(tmp_path, **env):
    py = tmp_path / 'python'
    py.write_text(STUB)
    py.chmod(0o755)
    log = tmp_path / 'stub.log'
    full = dict(os.environ, PY=str(py), STUB_LOG=str(log))
    for k in ('DRY', 'ONLY', 'CORE'):
        full.pop(k, None)
    full.update({k: str(v) for k, v in env.items()})
    r = subprocess.run(['bash', CHAIN], cwd=ROOT, capture_output=True, text=True, timeout=300,
                       env=full)
    calls = [x.rstrip('\x1f').split('\x1f') for x in log.read_text().splitlines()] \
        if log.exists() else []
    return r, calls, str(py)


def block_of(stdout, s):
    return stdout.split(f'] {s}: ', 1)[1].split('\n\n', 1)[0]


def printed_commands(block, py):
    """The commands a block lists, each as the argument list after the interpreter."""
    out = []
    for line in block.splitlines():
        if line.startswith('      taskset -c '):
            w = shlex.split(line)
            assert w[:3] == ['taskset', '-c', '<CORE>'] and w[3] == py, line
            out.append(w[4:])
    return out


@pytest.mark.parametrize('s', list(NEW))
def test_new_step_dry_run_prints_the_command_and_its_check(s, tmp_path):
    spec, out = NEW[s], tmp_path / 'out'
    r, calls, py = run_with_stub(tmp_path, DRY=1, ONLY=s, PACT_OUT=out)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    block = block_of(r.stdout, s)
    assert printed_commands(block, py) == [[spec['script']] + spec['args']]
    assert re.search(r'    writes: (.*)\n', block).group(1).split() == declared_outs(s, out)
    for p, dst in spec['brings']:
        how = ('when reported written, and then only when written by this run'
               if dst in spec['reported'] else 'only when written by this run')
        assert f'    brings: {out}/{p} -> {dst} ({how})' in block
    if not spec['brings']:
        assert '    brings:' not in block
    if spec['check']:
        assert [spec['script']] + spec['check'] in calls
    if spec['helper']:
        assert ['helper', spec['helper']] in calls
    checks = [[spec['script']] + spec['check']] if spec['check'] else []
    assert not any(c[:1] == [spec['script']] and '--check' not in c and c not in checks
                   for c in calls)
    assert re.search(r'^    holds today: no$', block, re.M), block
    assert not out.exists()                                  # the checks write nothing


@pytest.mark.parametrize('s', list(NEW))
def test_new_step_is_skipped_in_a_real_run_while_its_check_says_no(s, tmp_path):
    spec, out = NEW[s], tmp_path / 'out'
    log = os.path.join(ROOT, 'train_log', 'analysis_chain', f'{s}.log')
    before = os.stat(log).st_mtime_ns if os.path.exists(log) else None
    core = min(os.sched_getaffinity(0))
    r, calls, _ = run_with_stub(tmp_path, ONLY=s, CORE=core, PACT_OUT=out)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert f'    SKIP {s}: the precondition does not hold' in r.stdout
    assert re.search(rf'^  {s}\s+skipped$', r.stdout, re.M)
    assert [spec['script']] + spec['args'] not in calls       # the step's command never ran
    assert (os.stat(log).st_mtime_ns if os.path.exists(log) else None) == before
    assert not out.exists()


def planned(desc):
    """{(arm, size): instances} of the planner's segments of one description; the token of the
    policy records (read by p94 itself, never by the planner) is left out."""
    import timed_campaign_plan as P
    sizes, _, toks = desc.partition('=')
    keep = ' '.join(t for t in toks.split() if not t.startswith('policy_sample'))
    a = P.build_parser().parse_args(['table', '--arms', keep, '--sizes', *sizes.split()])
    return {(s.arm, size): insts for s in P.Campaign(a).segments
            for size, insts in s.by_size.items()}


def test_timed_reads_the_campaign_as_designed_and_p120_as_launched():
    """The defaults of the chain: steps timed and figure_timed read the campaign as designed
    (every production token on instances 3 to 11, the policy records too; at 10x25 CP-SAT on
    0 to 9 in the one description of that size, since p94 refuses a size named twice), and
    step lower_bound_reference reads the rule-started searches on the instances it read
    before (3 to 29 at the production batches, 0 to 29 and CP-SAT 0 to 9 at 10x25)."""
    assert chain_default('P94_CAMPAIGN_PROD') == DESIGNED_PROD
    assert chain_default('P94_CAMPAIGN_TRAIN') == DESIGNED_TRAIN
    nine = list(range(3, 12))
    assert planned(DESIGNED_PROD) == {(arm, size): nine for size in ('50x25n30', '80x25n30')
                                      for arm in ('memetic_rule', 'memetic_policy', 'cpsat_rule',
                                                  'cpsat_policy')}
    assert 'policy_sample@120:3-11' in DESIGNED_PROD.split()
    assert planned(DESIGNED_TRAIN) == {('memetic_rule', '10x25'): list(range(30)),
                                       ('cpsat_rule', '10x25'): list(range(10))}
    sizes = [s for d in (DESIGNED_PROD, DESIGNED_TRAIN) for s in d.partition('=')[0].split()]
    assert len(sizes) == len(set(sizes))
    read = {}
    for d in (PROD, TRAIN, SOLVER_TRAIN):
        got = {k: v for k, v in planned(d).items() if k[0] in ('memetic_rule', 'cpsat_rule')}
        assert not set(got) & set(read)                 # no cell in two descriptions
        read.update(got)
    assert read == {(arm, size): list(range(3, 30)) for size in ('50x25n30', '80x25n30')
                    for arm in ('memetic_rule', 'cpsat_rule')} | {
                        ('memetic_rule', '10x25'): list(range(30)),
                        ('cpsat_rule', '10x25'): list(range(10))}


@pytest.mark.parametrize('s', ['timed', 'figure_timed'])
def test_timed_steps_run_the_final_analysis_with_the_defaults(s, tmp_path, monkeypatch):
    """With no campaign variable set, the dry run of step timed prints p94 on the campaign as
    designed and checks it with timed-ready; step figure_timed draws fq3 from the outputs of
    that run."""
    for v in ('P94_CAMPAIGN_PROD', 'P94_CAMPAIGN_TRAIN', 'P120_CAMPAIGN_PROD',
              'P120_CAMPAIGN_TRAIN', 'P120_CAMPAIGN_SOLVER_TRAIN'):
        monkeypatch.delenv(v, raising=False)
    out = tmp_path / 'out'
    r, calls, py = run_with_stub(tmp_path, DRY=1, ONLY=s, PACT_OUT=out)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    block = block_of(r.stdout, s)
    if s == 'timed':
        assert printed_commands(block, py) == [['scripts/p94_timed_inference.py', '--campaign',
                                                DESIGNED_PROD, '--campaign', DESIGNED_TRAIN,
                                                '--primary', '600']]
        assert ['helper', 'timed-ready', DESIGNED_PROD, DESIGNED_TRAIN] in calls
        assert f"'{DESIGNED_PROD}' and '{DESIGNED_TRAIN}'" in block
    else:
        assert printed_commands(block, py) == [['scripts/make_figures.py', '--only', 'fq3',
                                                '--policy-time',
                                                f'{out}/results/deploy_cost_cpu_1thread.json']]
        assert re.search(r'^      - runs after: timed ', block, re.M)
    assert not out.exists()


def test_figure_timed_reads_the_best_of_eight_once_its_curves_exist(tmp_path, monkeypatch):
    """With the curves file of the eight-run analysis under PACT_OUT, step figure_timed passes
    that analysis's record to the figure; without the file it draws the single run (the test
    above)."""
    for v in ('P94_CAMPAIGN_PROD', 'P94_CAMPAIGN_TRAIN', 'P120_CAMPAIGN_PROD',
              'P120_CAMPAIGN_TRAIN', 'P120_CAMPAIGN_SOLVER_TRAIN'):
        monkeypatch.delenv(v, raising=False)
    out = tmp_path / 'out'
    (out / 'results').mkdir(parents=True)
    (out / 'results' / 'memetic_parallel_curves.csv').write_text('run\n')
    r, calls, py = run_with_stub(tmp_path, DRY=1, ONLY='figure_timed', PACT_OUT=out)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    block = block_of(r.stdout, 'figure_timed')
    assert printed_commands(block, py) == [['scripts/make_figures.py', '--only', 'fq3',
                                            '--policy-time',
                                            f'{out}/results/deploy_cost_cpu_1thread.json',
                                            '--memetic-parallel',
                                            f'{out}/results/memetic_parallel.json']]
    assert 'memetic curves are the single run' not in block


@pytest.mark.parametrize('curves', [True, False], ids=['with the best of eight', 'single run'])
def test_figure_timed_adds_the_one_minute_record_beside_the_best_of_eight(curves, tmp_path,
                                                                          monkeypatch):
    """With the record of step memetic_one_minute under PACT_OUT, step figure_timed passes it to
    the figure beside the best of eight (--memetic-minute after --memetic-parallel); without the
    curves file of the best of eight the figure draws the single run and gets no one-minute
    record, which the figure script refuses without --memetic-parallel."""
    for v in ('P94_CAMPAIGN_PROD', 'P94_CAMPAIGN_TRAIN', 'P120_CAMPAIGN_PROD',
              'P120_CAMPAIGN_TRAIN', 'P120_CAMPAIGN_SOLVER_TRAIN'):
        monkeypatch.delenv(v, raising=False)
    out = tmp_path / 'out'
    (out / 'results').mkdir(parents=True)
    (out / 'results' / 'memetic_one_minute.json').write_text('{}\n')
    if curves:
        (out / 'results' / 'memetic_parallel_curves.csv').write_text('run\n')
    r, calls, py = run_with_stub(tmp_path, DRY=1, ONLY='figure_timed', PACT_OUT=out)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    block = block_of(r.stdout, 'figure_timed')
    cmd = ['scripts/make_figures.py', '--only', 'fq3', '--policy-time',
           f'{out}/results/deploy_cost_cpu_1thread.json']
    if curves:
        cmd += ['--memetic-parallel', f'{out}/results/memetic_parallel.json',
                '--memetic-minute', f'{out}/results/memetic_one_minute.json']
    assert printed_commands(block, py) == [cmd]
    assert 'no points of the one-minute budget' not in block


@pytest.mark.parametrize('have', [('curves', 'minute', 'five'), ('curves', 'five'), ('five',)],
                         ids=['all three', 'without the one-minute record', 'five alone'])
def test_figure_timed_adds_the_five_minute_record_beside_the_one_minute_record(have, tmp_path,
                                                                               monkeypatch):
    """With the record of step memetic_five_minute under PACT_OUT, step figure_timed passes it
    to the figure after the one-minute record (--memetic-five after --memetic-minute); without
    the one-minute record, or without the best of eight, the figure gets no five-minute record,
    which the figure script refuses without --memetic-minute."""
    for v in ('P94_CAMPAIGN_PROD', 'P94_CAMPAIGN_TRAIN', 'P120_CAMPAIGN_PROD',
              'P120_CAMPAIGN_TRAIN', 'P120_CAMPAIGN_SOLVER_TRAIN'):
        monkeypatch.delenv(v, raising=False)
    out = tmp_path / 'out'
    (out / 'results').mkdir(parents=True)
    files = {'curves': 'memetic_parallel_curves.csv', 'minute': 'memetic_one_minute.json',
             'five': 'memetic_five_minute.json'}
    for k in have:
        (out / 'results' / files[k]).write_text('{}\n')
    r, calls, py = run_with_stub(tmp_path, DRY=1, ONLY='figure_timed', PACT_OUT=out)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    block = block_of(r.stdout, 'figure_timed')
    cmd = ['scripts/make_figures.py', '--only', 'fq3', '--policy-time',
           f'{out}/results/deploy_cost_cpu_1thread.json']
    if 'curves' in have:
        cmd += ['--memetic-parallel', f'{out}/results/memetic_parallel.json']
    if 'minute' in have:
        cmd += ['--memetic-minute', f'{out}/results/memetic_one_minute.json',
                '--memetic-five', f'{out}/results/memetic_five_minute.json']
    assert printed_commands(block, py) == [cmd]
    assert 'no points of the five-minute budget' not in block


def test_figure_timed_names_a_missing_five_minute_record(tmp_path, monkeypatch):
    """Without the record of step memetic_five_minute the dry run of step figure_timed says that
    the figure gets no points of the five-minute budget."""
    for v in ('P94_CAMPAIGN_PROD', 'P94_CAMPAIGN_TRAIN', 'P120_CAMPAIGN_PROD',
              'P120_CAMPAIGN_TRAIN', 'P120_CAMPAIGN_SOLVER_TRAIN'):
        monkeypatch.delenv(v, raising=False)
    out = tmp_path / 'out'
    r, calls, py = run_with_stub(tmp_path, DRY=1, ONLY='figure_timed', PACT_OUT=out)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert (f'{out}/results/memetic_five_minute.json not found, the figure gets no points of '
            'the five-minute budget') in block_of(r.stdout, 'figure_timed')


# ------------------------------------------------------------------ step timed_instance_sets
# the two tagged records of p94 that p130 sets side by side: the two searches started from the
# rule schedule on the nine instances of the design and on every evidence instance
TIS = 'timed_instance_sets'
STARTNINE = '50x25n30 80x25n30 = memetic_rule@600:3-11 cpsat_rule@600:3-11'
STARTALL = '50x25n30 80x25n30 = memetic_rule@600:3-29 cpsat_rule@600:3-29'
P94 = 'scripts/p94_timed_inference.py'
P130 = 'scripts/p130_timed_instance_sets.py'
TAGGED = ['--origin', 'start', '--budgets', '60', '300', '600', '--primary', '600']
TIS_CMDS = [[P94, *TAGGED, '--tag', 'startnine', '--campaign', STARTNINE],
            [P94, *TAGGED, '--tag', 'startall', '--campaign', STARTALL],
            [P130]]
TIS_VARS = ('P94_CAMPAIGN_STARTNINE', 'P94_CAMPAIGN_STARTALL')


def tis_outputs():
    """The six tagged files of the two p94 runs (repository root) and the three outputs of p130
    (under $PACT_OUT), from the two scripts' own tables."""
    tagged = source_constant('p94_timed_inference.py', 'TAG_OUTPUTS')
    of = source_constant('p130_timed_instance_sets.py', 'OUT_FILES')
    return ([p.format(tag=t) for t in ('startnine', 'startall') for _, p in tagged],
            [of[k] for k in ('macros', 'json', 'md')])


def tis_calls(calls):
    """The calls of the two scripts of the step (a check of the chain's helper excluded)."""
    return [c for c in calls if c[:1] in ([P94], [P130])]


def test_timed_instance_sets_reads_the_campaigns_and_tags_of_p130():
    """The chain's two campaigns are the defaults of p130: the two rule-started searches on
    instances 3 to 11 (design, tag startnine) and 3 to 29 (evidence, tag startall) at both
    production batches, the evidence cells those of the campaign as launched; the budgets of the
    two p94 runs are p130's."""
    assert tuple(chain_default(v) for v in TIS_VARS) == (STARTNINE, STARTALL)
    sets = source_constant('p130_timed_instance_sets.py', 'SETS')
    assert [(s[0], s[2]) for s in sets] == [('design', 'startnine'), ('evidence', 'startall')]
    assert source_constant('p130_timed_instance_sets.py', 'BUDGETS') == (60, 300, 600)
    assert source_constant('p130_timed_instance_sets.py', 'ARMS') == ('memetic_rule', 'cpsat_rule')
    doc = ast.get_docstring(ast.parse(open(os.path.join(ROOT, P130)).read()))
    assert f'--campaign "{STARTNINE}"' in doc
    arms = ('memetic_rule', 'cpsat_rule')
    for desc, insts in ((STARTNINE, range(3, 12)), (STARTALL, range(3, 30))):
        assert planned(desc) == {(a, size): list(insts) for a in arms
                                 for size in ('50x25n30', '80x25n30')}
    assert planned(STARTALL) == {k: v for k, v in planned(LAUNCHED_PROD).items() if k[0] in arms}


def test_timed_instance_sets_is_declared():
    text = src()
    for fn in ('title_of', 'deps_of', 'pre_text', 'check_of', 'cmds_of', 'outs_of', 'copies_of'):
        assert re.search(rf'^    {TIS}\)', body_of(fn, text), re.M), fn
    assert re.search(rf'^    {TIS}\) echo "timed";;', body_of('deps_of', text), re.M)
    assert re.search(rf'^    {TIS}\)\n      helper timed-ready "\$P94_CAMPAIGN_STARTALL" \|\| ok=1;;',
                     body_of('check_of', text), re.M)
    cmds = re.search(rf'^    {TIS}\)\n((?:      .*\n)+?)(?=    \S)', body_of('cmds_of', text) + '\n    .',
                     re.M).group(1)
    for tag, var in (('startnine', TIS_VARS[0]), ('startall', TIS_VARS[1])):
        assert (f'run_cmd {P94} {" ".join(TAGGED)} \\\n              --tag {tag} --campaign '
                f'"${var}" || return 1') in cmds
    assert f'run_cmd {P130} || return 1;;' in cmds
    tagged, p130 = tis_outputs()
    got = re.search(rf'^    {TIS}\) printf .%s\\n. ((?:[^;]|\n)*);;', text, re.M).group(1)
    assert got.replace('\\', ' ').split() == tagged + [f'"$PACT_OUT/{p}"' for p in p130]
    copies = re.findall(rf'^    {TIS}\) (.*)$', body_of('copies_of', text), re.M)
    assert copies == [f'echo "$PACT_OUT/{p130[0]} manuscript/macros_timedsets.tex";;']
    assert p130[0] == 'manuscript/macros_timedsets.tex'
    assert "'manuscript/macros_timedsets.tex'" in text      # in CHAIN of the macro table
    # the header lists the step between timed and memetic_parallel, documents its two
    # variables, and the list of the runs the chain waits for names it
    head = text.split('\nset -uo pipefail', 1)[0]
    i, j, k = (head.index(f'\n#   {s} ') for s in ('timed', TIS, 'memetic_parallel'))
    assert i < j < k
    assert '#   P94_CAMPAIGN_STARTNINE, P94_CAMPAIGN_STARTALL ' in head
    assert f'step {TIS}' in text.split('RUNS THIS CHAIN WAITS FOR', 1)[1].split('NOT IN THIS CHAIN')[0]


def test_timed_instance_sets_dry_run_prints_its_commands_and_its_check(tmp_path, monkeypatch):
    for v in TIS_VARS:
        monkeypatch.delenv(v, raising=False)
    out = tmp_path / 'out'
    r, calls, py = run_with_stub(tmp_path, DRY=1, ONLY=TIS, PACT_OUT=out)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert f'steps selected: {TIS} \n' in r.stdout                  # selected by ONLY alone
    assert f'[{ORDER.index(TIS) + 1}/{len(ORDER)}] {TIS}: ' in r.stdout
    block = block_of(r.stdout, TIS)
    assert printed_commands(block, py) == TIS_CMDS
    tagged, p130 = tis_outputs()
    writes = re.search(r'    writes: (.*)\n', block).group(1).split()
    assert writes == tagged + [f'{out}/{p}' for p in p130]
    assert re.findall(r'^    brings: .*$', block, re.M) == [
        f'    brings: {out}/manuscript/macros_timedsets.tex -> manuscript/macros_timedsets.tex '
        f'(only when written by this run)']
    assert re.search(r'^      - runs after: timed \(', block, re.M)
    assert f"'{STARTALL}'" in block and f"'{STARTNINE}'" in block   # the precondition text
    assert [c for c in calls if c[:2] != ['helper', 'macro-table']] == [
        ['helper', 'timed-ready', STARTALL]]
    assert re.search(r'^    holds today: no$', block, re.M), block
    assert not out.exists()                                  # the checks write nothing


def test_timed_instance_sets_reads_its_campaigns_from_the_environment(tmp_path):
    nine = '50x25n30 = memetic_rule@600:3-5 cpsat_rule@600:3-5'
    every = '50x25n30 = memetic_rule@600:3-8 cpsat_rule@600:3-8'
    out = tmp_path / 'out'
    r, calls, py = run_with_stub(tmp_path, DRY=1, ONLY=TIS, PACT_OUT=out,
                                 P94_CAMPAIGN_STARTNINE=nine, P94_CAMPAIGN_STARTALL=every)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    block = block_of(r.stdout, TIS)
    assert printed_commands(block, py) == [TIS_CMDS[0][:-1] + [nine], TIS_CMDS[1][:-1] + [every],
                                           TIS_CMDS[2]]
    assert ['helper', 'timed-ready', every] in calls and f"'{nine}'" in block
    assert not out.exists()


def tagged_mtimes():
    tagged, _ = tis_outputs()
    return {p: os.stat(os.path.join(ROOT, p)).st_mtime_ns
            for p in tagged if os.path.exists(os.path.join(ROOT, p))}


def test_timed_instance_sets_is_skipped_in_a_real_run_while_its_check_says_no(tmp_path,
                                                                             monkeypatch):
    for v in TIS_VARS:
        monkeypatch.delenv(v, raising=False)
    out = tmp_path / 'out'
    logs = [os.path.join(ROOT, 'train_log', 'analysis_chain', f) for f in (f'{TIS}.log',
                                                                          f'.{TIS}.started')]
    before = [os.stat(x).st_mtime_ns if os.path.exists(x) else None for x in logs]
    tagged = tagged_mtimes()
    r, calls, _ = run_with_stub(tmp_path, ONLY=TIS, CORE=min(os.sched_getaffinity(0)), PACT_OUT=out)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert f'    SKIP {TIS}: the precondition does not hold' in r.stdout
    assert re.search(rf'^  {TIS}\s+skipped$', r.stdout, re.M)
    assert ['helper', 'timed-ready', STARTALL] in calls and not tis_calls(calls)
    assert [os.stat(x).st_mtime_ns if os.path.exists(x) else None for x in logs] == before
    assert tagged_mtimes() == tagged
    assert not out.exists()


def test_timed_instance_sets_waits_for_step_timed(tmp_path, monkeypatch):
    """Selected with step timed in a real run, the step is skipped without its check when step
    timed did not finish (here its check says no)."""
    for v in TIS_VARS + ('P94_CAMPAIGN_PROD', 'P94_CAMPAIGN_TRAIN'):
        monkeypatch.delenv(v, raising=False)
    out = tmp_path / 'out'
    tagged = tagged_mtimes()
    r, calls, _ = run_with_stub(tmp_path, ONLY=f'timed,{TIS}', CORE=min(os.sched_getaffinity(0)),
                                PACT_OUT=out)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert 'steps selected: timed timed_instance_sets \n' in r.stdout
    assert '    SKIP timed: the precondition does not hold' in r.stdout
    assert f'    SKIP {TIS}: dependency timed selected in this run and not finished' in r.stdout
    assert ['helper', 'timed-ready', DESIGNED_PROD, DESIGNED_TRAIN] in calls
    assert ['helper', 'timed-ready', STARTALL] not in calls and not tis_calls(calls)
    assert re.search(r'^  timed\s+skipped$', r.stdout, re.M)
    assert re.search(rf'^  {TIS}\s+skipped$', r.stdout, re.M)
    assert tagged_mtimes() == tagged
    assert not out.exists()


# the records fq4_limits reads in full mode (make_figures.fq4_limits through
# arm_registry.out, and the record of p87 at the repository root)
LIMITS_IN = ['{out}/results/family_inference.json', '{out}/paper/macros_family.tex',
             '{out}/results/per_floor.json', '{out}/manuscript/macros_perfloor.tex',
             '{out}/results/production_specialists.json', '{out}/manuscript/macros_prodspec.tex',
             'results/sensitivity_inference.json', 'manuscript/macros_sensitivity.tex']


def test_figure_limits_is_declared():
    """Step figure_limits: every case statement names it, it waits for the four steps whose
    records it reads, it draws variant A of fq4_limits in full mode in the setting of the run,
    writes its two files, brings nothing, and checks the records the figure script reads."""
    import make_figures as F
    text = src()
    for fn in ('title_of', 'deps_of', 'pre_text', 'check_of', 'cmds_of', 'outs_of'):
        assert re.search(r'^    figure_limits\)', body_of(fn, text), re.M), fn
    assert not re.search(r'^    figure_limits\)', body_of('copies_of', text), re.M)
    assert re.search(r'^    figure_limits\) echo "family per_floor production_specialists '
                     r'sensitivity";;', body_of('deps_of', text), re.M)
    assert ('run_cmd scripts/make_figures.py --only fq4_limits --fq4-limits-variant A \\\n'
            '              --pact-env "$PACT_ENV" --pact-seeds "$PACT_SEEDS" --pact-out "$PACT_OUT"'
            ' || return 1') in text
    check = body_of('check_of', text)
    for f in LIMITS_IN:
        assert f.format(out='$PACT_OUT') in check, f
    # the same records as the figure script's defaults
    assert (F.FQL_PERFLOOR_JSON, F.FQL_PERFLOOR_MACROS, F.FQL_PROD_JSON, F.FQL_PROD_MACROS,
            F.fql_sens_json(F.FQL_SENS_CSV), F.FQL_SENS_MACROS) == tuple(
                f.replace('{out}/', '') for f in LIMITS_IN[2:])
    fig = open(os.path.join(ROOT, 'scripts', 'make_figures.py')).read()
    assert "REG.out('results/family_inference.json')" in fig
    assert "REG.out('paper/macros_family.tex')" in fig


def test_figure_limits_dry_run_lists_its_command_and_missing_records(tmp_path):
    out = tmp_path / 'out'
    r, calls, py = run_with_stub(tmp_path, DRY=1, ONLY='figure_limits', PACT_OUT=out)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    block = block_of(r.stdout, 'figure_limits')
    assert printed_commands(block, py) == [['scripts/make_figures.py', '--only', 'fq4_limits',
                                            '--fq4-limits-variant', 'A', '--pact-env', 'exact',
                                            '--pact-seeds', '301-305', '--pact-out', str(out)]]
    assert re.search(r'    writes: (.*)\n', block).group(1).split() == [
        'manuscript/sections_new/figures/fq4_limits_A.pdf',
        'manuscript/sections_new/figures/fq4_limits_A.png']
    for f in LIMITS_IN[:6]:
        assert f'      missing {f.format(out=out)}' in block
    assert re.search(r'^      - runs after: family per_floor production_specialists sensitivity ',
                     block, re.M)
    assert re.search(r'^    holds today: no$', block, re.M), block
    assert not any(c[:1] == ['scripts/make_figures.py'] for c in calls)
    assert not out.exists()


def test_figure_limits_is_skipped_in_a_real_run_while_a_record_is_missing(tmp_path):
    out = tmp_path / 'out'
    log = os.path.join(ROOT, 'train_log', 'analysis_chain', 'figure_limits.log')
    before = os.stat(log).st_mtime_ns if os.path.exists(log) else None
    r, calls, _ = run_with_stub(tmp_path, ONLY='figure_limits',
                                CORE=min(os.sched_getaffinity(0)), PACT_OUT=out)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert '    SKIP figure_limits: the precondition does not hold' in r.stdout
    assert not any(c[:1] == ['scripts/make_figures.py'] for c in calls)
    assert (os.stat(log).st_mtime_ns if os.path.exists(log) else None) == before
    assert not out.exists()


@pytest.mark.skipif(
    not os.path.isfile(os.path.join(ROOT, 'manuscript', 'sections_new', 'fig_limits.tex')),
    reason='needs the sources of the manuscript under manuscript/sections_new and '
           'manuscript/supp_new (fig_limits.tex among them), which the release does not hold')
def test_no_step_draws_the_boundary_figure_the_manuscript_does_not_input():
    """fq4_boundary is input by no file of the manuscript or its supplement, so no step of the
    chain draws it; fq4_limits, which the manuscript inputs, has its step."""
    tex = []
    for d in ('manuscript/sections_new', 'manuscript/supp_new'):
        for base, _, files in os.walk(os.path.join(ROOT, d)):
            tex += [open(os.path.join(base, f)).read() for f in files if f.endswith('.tex')]
    assert tex and not any('fq4_boundary' in t for t in tex)
    assert any('fq4_limits_A' in t for t in tex)
    text = src()
    assert 'figure_boundary' not in text and '--only fq4 ' not in text and \
        '--only fq4 ||' not in text
    assert 'figure_limits' in chain_steps()


def test_timed_slots_must_be_core_sets(tmp_path):
    r, _, _ = run_with_stub(tmp_path, DRY=1, ONLY='memetic_parallel', TIMED_SLOTS='16-23 x')
    assert r.returncode == 3 and 'REFUSING: TIMED_SLOTS=16-23 x' in r.stderr


def test_p33_check_says_no_without_arrays(tmp_path):
    """p33 has its own check (--check); the chain's former helper p33-ready is gone."""
    assert 'p33_ready' not in src() and 'p33-ready' not in src()
    env = dict(os.environ, PACT_ENV='exact', PACT_SEEDS='301-305', CUDA_VISIBLE_DEVICES='',
               PACT_RESULTS=str(tmp_path / 'no_results'), PACT_OUT=str(tmp_path / 'out'))
    r = subprocess.run([sys.executable, '-B', os.path.join(ROOT, 'scripts', 'p33_ood_table.py'),
                        '--check'], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 1, r.stdout + r.stderr
    for f in ('10x25_xmask', '10x25_oodsetup_xmask', '10x25_oodwindow_xmask'):
        assert f'{tmp_path}/no_results/{f}' in r.stdout       # p33's three folders
    assert 'the record of step family' in r.stdout and 'missing' in r.stdout
    assert not (tmp_path / 'out').exists() or not any((tmp_path / 'out').rglob('*.*'))


def test_p123_readiness_compares_its_run_with_the_selected_run(monkeypatch, capsys):
    import arm_registry as REG
    import p123_example_schedule as P123
    monkeypatch.setenv('PACT_ENV', 'exact')
    seed = int(P123.MODEL.rsplit('-s', 1)[1])
    H = chain_helper()
    monkeypatch.setattr(REG, 'select_seed', lambda *_: (seed, {}))
    assert H['p123_ready']() is True
    other = 303 if seed != 303 else 302
    monkeypatch.setattr(REG, 'select_seed', lambda *_: (other, {}))
    assert H['p123_ready']() is False
    assert 'they differ' in capsys.readouterr().out


def bash_functions(*names):
    text = src()
    return '\n'.join(re.search('^' + n + r'\(\) \{.*?\n\}\n', text, re.S | re.M).group(0)
                     for n in names)


def test_a_reported_file_is_named_by_the_last_line_of_the_log(tmp_path):
    log = tmp_path / 'step.log'
    log.write_text('# precondition check:\n#       9 macros; wrote x/table.tex\n'
                   '3 macros; wrote out/exact/manuscript/macros_lbref.tex, '
                   'out/exact/results/lower_bound_reference.json\n')
    script = bash_functions('reported_written') + (
        f'\nfor f in out/exact/results/lower_bound_reference.json out//exact/manuscript/macros_lbref.tex '
        f'out/exact/manuscript/tables/lbref_table.tex x/table.tex; do\n'
        f'  reported_written {shlex.quote(str(log))} "$f" && echo "$f yes" || echo "$f no"\ndone\n'
        f'reported_written {shlex.quote(str(tmp_path / "absent.log"))} x && echo absent-yes || echo absent-no\n')
    r = subprocess.run(['bash', '-c', script], capture_output=True, text=True, timeout=60)
    assert r.stdout.split('\n')[:5] == ['out/exact/results/lower_bound_reference.json yes',
                                        'out//exact/manuscript/macros_lbref.tex yes',
                                        'out/exact/manuscript/tables/lbref_table.tex no',
                                        'x/table.tex no', 'absent-no']
    # the copy loop reads the third field and leaves an unreported file out with a note
    text = src()
    assert text.count('while read -r src dst how; do') == 2
    assert '[ "${how:-}" = reported ] && ! reported_written "$LOG" "$src"' in text


# ------------------------------------------------------------------ step figures
FQ2_REUSE = ['scripts/make_figures.py', '--only', 'fq2_reuse', '--crosseval', '{out}/results/cross_eval.json',
             '--crosseval-macros', '{out}/manuscript/macros_crosseval.tex']
FQ2_HELDOUT = ['scripts/make_figures.py', '--only', 'fq2_heldout', '--factory-seeds', 'auto',
               '--heldout-splits', '{out}/results/heldout_splits.json',
               '--heldout-splits-macros', '{out}/manuscript/macros_heldout.tex',
               '--secondary', '{sec}/results/secondary_inference.json',
               '--secondary-macros', '{sec}/manuscript/macros_secondary.tex']
FQ5_RULE_CHANGE = ['scripts/make_figures.py', '--only', 'fq5_rule_change',
                   '--rule-change', '{out}/results/rule_change_cost.json',
                   '--rule-change-macros', '{out}/manuscript/macros_rulechange_cost.tex',
                   '--deploy', '{out}/results/deploy_cost_cpu_1thread.json',
                   '--deploy-macros', '{out}/paper/macros_deploy.tex']
FQ1 = ['scripts/make_figures.py', '--only', 'fq1', '--factory-seeds', 'auto']
FQ2_KEYS = ['cross', 'crossmac', 'split', 'splitmac', 'sec', 'secmac']
FQ5_KEYS = ['rcc', 'rccmac', 'dep', 'depmac']


def fill(args, **kv):
    return [a.format(**kv) for a in args]


def figures_harness(tmp_path, have, real=False, selected=(), done=()):
    """cmds_of figures and outs_of figures from the chain's source, in a shell of their own
    whose record files lie under tmp_path (the ones named in have exist)."""
    out, sec = tmp_path / 'out', tmp_path / 'repo'
    files = {'cross': f'{out}/results/cross_eval.json', 'crossmac': f'{out}/manuscript/macros_crosseval.tex',
             'split': f'{out}/results/heldout_splits.json', 'splitmac': f'{out}/manuscript/macros_heldout.tex',
             'sec': f'{sec}/results/secondary_inference.json', 'secmac': f'{sec}/manuscript/macros_secondary.tex',
             'rcc': f'{out}/results/rule_change_cost.json',
             'rccmac': f'{out}/manuscript/macros_rulechange_cost.tex',
             'dep': f'{out}/results/deploy_cost_cpu_1thread.json', 'depmac': f'{out}/paper/macros_deploy.tex'}
    for k in have:
        os.makedirs(os.path.dirname(files[k]), exist_ok=True)
        open(files[k], 'w').close()
    script = '\n'.join([
        'set -uo pipefail', 'DRY=1', f'REAL_RUN={int(real)}', 'PY=python', 'CORE=', 'LOG=/dev/null',
        f'PACT_OUT={shlex.quote(str(out))}', 'declare -A SELECTED=() STATUS=()',
        *[f'SELECTED[{x}]=1' for x in selected], *[f'STATUS[{x}]=done' for x in done],
        f'FQ2_REUSE_IN=({files["cross"]} {files["crossmac"]})',
        f'FQ2_HELDOUT_IN=({files["split"]} {files["splitmac"]} {files["sec"]} {files["secmac"]})',
        f'FQ5_RULE_CHANGE_IN=({files["rcc"]} {files["rccmac"]} {files["dep"]} {files["depmac"]})',
        bash_functions('quote', 'pin', 'run_cmd', 'note', 'missing_of', 'finished_or_unselected',
                       'figure_left_out', 'cmds_of', 'outs_of'),
        'cmds_of figures; echo "rc=$?"', 'echo "--- outs"', 'outs_of figures'])
    r = subprocess.run(['bash', '-c', script], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    cmds_text, outs_text = r.stdout.split('--- outs\n')
    assert cmds_text.endswith('rc=0\n'), cmds_text
    cmds = [shlex.split(x)[4:] for x in cmds_text.splitlines() if x.startswith('      taskset')]
    notes = [x.strip() for x in cmds_text.splitlines()
             if x.startswith('      ') and not x.startswith('      taskset')]
    return cmds, notes, outs_text.split(), dict(out=out, sec=sec)


FIG = 'manuscript/sections_new/figures/'
FQ1_OUTS = [FIG + f'fq1_specialists_{v}.{e}' for v in 'AB' for e in ('pdf', 'png')]


def test_figures_draws_every_further_figure_when_its_records_exist(tmp_path):
    cmds, notes, outs, kv = figures_harness(tmp_path, FQ2_KEYS + FQ5_KEYS)
    assert cmds == [FQ1, fill(FQ2_REUSE, **kv), fill(FQ2_HELDOUT, **kv), fill(FQ5_RULE_CHANGE, **kv)]
    assert '--heldout-partial' not in sum(cmds, [])        # the full mode only
    assert notes == []
    assert outs == FQ1_OUTS + [FIG + f'{f}.{e}' for f in ('fq2_reuse', 'fq2_heldout', 'fq5_rule_change')
                               for e in ('pdf', 'png')]


@pytest.mark.parametrize('missing', ['crossmac', 'split', 'sec', 'secmac', 'rcc', 'rccmac', 'dep', 'depmac'])
def test_figures_leaves_a_figure_out_while_a_record_is_missing(tmp_path, missing):
    have = [k for k in FQ2_KEYS + FQ5_KEYS if k != missing]
    cmds, notes, outs, kv = figures_harness(tmp_path, have)
    fig = 'fq2_reuse' if missing == 'crossmac' else 'fq5_rule_change' if missing in FQ5_KEYS \
        else 'fq2_heldout'
    calls = {'fq2_reuse': FQ2_REUSE, 'fq2_heldout': FQ2_HELDOUT, 'fq5_rule_change': FQ5_RULE_CHANGE}
    assert cmds == [FQ1] + [fill(c, **kv) for f, c in calls.items() if f != fig]
    assert len(notes) == 1 and notes[0].startswith(f'{fig} left out: ') and 'missing' in notes[0]
    assert FIG + f'{fig}.pdf' not in outs and FQ1_OUTS == outs[:4]


def test_figures_waits_for_its_steps_when_they_were_selected(tmp_path):
    every = FQ2_KEYS + FQ5_KEYS
    # a real run in which heldout_splits was selected and skipped, cross_eval finished
    cmds, notes, outs, kv = figures_harness(tmp_path, every, real=True,
                                            selected=('cross_eval', 'heldout_splits'),
                                            done=('cross_eval',))
    assert cmds == [FQ1, fill(FQ2_REUSE, **kv), fill(FQ5_RULE_CHANGE, **kv)]
    assert notes == ['fq2_heldout left out: step heldout_splits or secondary was selected in this '
                     'run and did not finish']
    # secondary selected and finished, heldout_splits not selected: both are drawn
    cmds, notes, _, kv = figures_harness(tmp_path, every, real=True, selected=('secondary',),
                                         done=('secondary',))
    assert cmds == [FQ1, fill(FQ2_REUSE, **kv), fill(FQ2_HELDOUT, **kv),
                    fill(FQ5_RULE_CHANGE, **kv)] and not notes
    # deploy selected and skipped (its measurement is absent), rule_change_cost finished
    cmds, notes, _, kv = figures_harness(tmp_path, every, real=True,
                                         selected=('rule_change_cost', 'deploy'),
                                         done=('rule_change_cost',))
    assert cmds == [FQ1, fill(FQ2_REUSE, **kv), fill(FQ2_HELDOUT, **kv)]
    assert notes == ['fq5_rule_change left out: step rule_change_cost or deploy was selected in '
                     'this run and did not finish']


def test_figures_names_the_record_files_of_the_figure_script():
    text = src()
    assert 'FQ2_REUSE_IN=("$PACT_OUT/results/cross_eval.json" "$PACT_OUT/manuscript/macros_crosseval.tex")' in text
    assert ('FQ2_HELDOUT_IN=("$PACT_OUT/results/heldout_splits.json" "$PACT_OUT/manuscript/macros_heldout.tex"\n'
            '                results/secondary_inference.json manuscript/macros_secondary.tex)') in text
    assert ('FQ5_RULE_CHANGE_IN=("$PACT_OUT/results/rule_change_cost.json" '
            '"$PACT_OUT/manuscript/macros_rulechange_cost.tex"\n'
            '                    "$DEPLOY_JSON" "$PACT_OUT/paper/macros_deploy.tex")') in text
    assert re.search(r'^DEPLOY_JSON=\$\{DEPLOY_JSON:-\$PACT_OUT/results/deploy_cost_cpu_1thread\.json\}',
                     text, re.M)
    fig = open(os.path.join(ROOT, 'scripts', 'make_figures.py')).read()
    # the figure's own defaults: p122 under the output root, p95 at the repository root
    for k, v in (('FQ2_JSON', 'results/cross_eval.json'), ('FQ2_MACROS', 'manuscript/macros_crosseval.tex'),
                 ('FQH_SPLITS_JSON', 'results/heldout_splits.json'),
                 ('FQH_SPLITS_MACROS', 'manuscript/macros_heldout.tex'),
                 ('FQH_SEC_JSON', 'results/secondary_inference.json'),
                 ('FQH_SEC_MACROS', 'manuscript/macros_secondary.tex'),
                 ('FQ5_JSON', 'results/rule_change_cost.json'),
                 ('FQ5_MACROS', 'manuscript/macros_rulechange_cost.tex'),
                 ('FQ5_DEPLOY_JSON', 'results/deploy_cost_cpu_1thread.json'),
                 ('FQ5_DEPLOY_MACROS', 'paper/macros_deploy.tex')):
        assert re.search(rf"^{k} = '{re.escape(v)}'", fig, re.M), k
    assert "path(given['secondary'], FQH_SEC_JSON)" in fig and 'REG.out(FQH_SPLITS_JSON)' in fig
    deps = body_of('deps_of', text)
    assert re.search(r'^    figures\) echo "family public_act";;', deps, re.M)


def test_figures_dry_run_lists_its_calls(tmp_path):
    out = tmp_path / 'out'
    for p in ('results/cross_eval.json', 'manuscript/macros_crosseval.tex'):
        os.makedirs(os.path.dirname(out / p), exist_ok=True)
        open(out / p, 'w').close()
    r, calls, py = run_with_stub(tmp_path, DRY=1, ONLY='figures', PACT_OUT=out)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    block = block_of(r.stdout, 'figures')
    kv = dict(out=out, sec=ROOT)
    assert printed_commands(block, py) == [FQ1, fill(FQ2_REUSE, **kv)]
    assert re.search(r'^      fq2_heldout left out: a record of step heldout_splits or secondary is '
                     rf'missing: {re.escape(str(out))}/results/heldout_splits\.json ', block, re.M)
    assert re.search(r'^      fq5_rule_change left out: a record of step rule_change_cost or deploy is '
                     rf'missing: {re.escape(str(out))}/results/rule_change_cost\.json ', block, re.M)
    writes = re.search(r'    writes: (.*)\n', block).group(1).split()
    assert writes == FQ1_OUTS + [FIG + 'fq2_reuse.pdf', FIG + 'fq2_reuse.png']
    assert not calls or all(c[0] == 'helper' for c in calls)  # no figure script started


# ------------------------------------------------------------------ step convergence
CONV_CMDS = [['scripts/p116_convergence.py'],
             ['scripts/make_figures.py', '--only', 'fs_convergence']]


def step_case(fn, step, text=None):
    """The lines of one step in a case statement, up to its ;;."""
    return re.search(rf'^    {step}\)\n?(.*?);;', body_of(fn, text), re.M | re.S).group(1)


def test_convergence_holds_only_with_the_specialists_of_2000_updates():
    import p116_convergence as P116
    text = src()
    pre = step_case('pre_text', 'convergence', text)
    assert 'every specialist trained for 2000 updates, eight combinations and seeds $PACT_SEEDS ' \
           '(40 runs' in pre
    assert 'scripts/longspec_runs.py' in pre and 'the step holds only when all 40 have finished' in pre
    assert 'p116 --check' in pre
    assert step_case('check_of', 'convergence', text).strip().splitlines()[0] == \
        'check_cmd scripts/p116_convergence.py --check | sed \'s/^/      /\''
    cmds = [ln.strip() for ln in step_case('cmds_of', 'convergence', text).strip().splitlines()]
    assert cmds == [f'run_cmd {" ".join(c)} || return 1' for c in CONV_CMDS]
    assert re.search(r'^    convergence\) echo "";;', body_of('deps_of', text), re.M)
    # the check of p116 reads the eight specialists of 2000 updates of every seed the chain sets
    assert re.search(r'^export PACT_SEEDS=301-305$', text, re.M)
    long = [c for a, c in P116.arm_combos() if a == 'long']
    assert long == list(P116.R8) and len(long) * 5 == 40


def test_convergence_dry_run_prints_both_commands_and_its_check(tmp_path):
    out = tmp_path / 'out'
    r, calls, py = run_with_stub(tmp_path, DRY=1, ONLY='convergence', PACT_OUT=out)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    block = block_of(r.stdout, 'convergence')
    assert printed_commands(block, py) == CONV_CMDS
    assert 'every specialist trained for 2000 updates' in block
    assert ['scripts/p116_convergence.py', '--check'] in calls
    assert not any(c[:1] == ['scripts/p116_convergence.py'] and '--check' not in c for c in calls)
    assert not any(c[:1] == ['scripts/make_figures.py'] for c in calls)
    assert re.search(r'^    holds today: no$', block, re.M), block
    assert not out.exists()


def test_convergence_is_skipped_in_a_real_run_while_its_check_says_no(tmp_path):
    out = tmp_path / 'out'
    log = os.path.join(ROOT, 'train_log', 'analysis_chain', 'convergence.log')
    before = os.stat(log).st_mtime_ns if os.path.exists(log) else None
    core = min(os.sched_getaffinity(0))
    r, calls, _ = run_with_stub(tmp_path, ONLY='convergence', CORE=core, PACT_OUT=out)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert '    SKIP convergence: the precondition does not hold' in r.stdout
    assert re.search(r'^  convergence\s+skipped$', r.stdout, re.M)
    assert all(c not in calls for c in CONV_CMDS)             # neither command ran
    assert (os.stat(log).st_mtime_ns if os.path.exists(log) else None) == before
    assert not out.exists()
