"""Tests for section 9 of scripts/p89_public_family.py, the gap to the references of the
public benchmark (records of scripts/p91_public_cpsat_refs.py), on synthetic trees: the
public family of tests/test_public_family_inference.py (six instances per set) with
synthetic reference records.

Run from the repository root:  python -m pytest tests/test_public_references.py -q
CPU only; every file is written under a temporary folder.
"""
import csv
import json
import os
import re
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import arm_registry as REG  # noqa: E402
import p89_public_family as P  # noqa: E402
import stats_family as SF  # noqa: E402
import test_public_family_inference as T  # noqa: E402
from test_public_family_inference import small_sets  # noqa: E402,F401  (autouse: six instances)

R8 = list(REG.COMBOS)
ST = P.REF_SET
B = T.B
N = T.N_INST
# synthetic references: N equals the best-known value (0.7 x base); every other combination
# 0.75 x base x (1 + 0.1 index), proved; LSW instance 0 stopped at the limit at 1.6 x base
# with bound 1.2 x base (25% gap), above two runs of the policy (seeds 303 and 305)
LIMIT = {('LSW', 0): dict(objective=1.6, bound=1.2, status='FEASIBLE')}


def bases(root):
    """The base makespans of the tree: the N specialist of seed 301 (factor 1.0)."""
    p = os.path.join(root, REG.PUB_RESULTS, ST + P.XS,
                     f'Result_greedy+{REG.PUB_PREFIX}{REG.pub_model("specialist", "N", 301)}_N.npy')
    return np.load(p)[:, 0]


def ref_value(r, base):
    return base * 0.7 if r == 'N' else base * 0.75 * (1 + 0.1 * R8.index(r))


def write_refs(root, combos=R8, count=N, edit=None, limit=LIMIT):
    """Records as p91 writes them for the first `count` instances of each combination;
    edit[(r, i)] overrides fields (values in units of the base where objective, bound and
    warm_makespan are floats below 5)."""
    base = bases(root)
    folder = os.path.join(root, P.Inputs(root).ref_folder())
    names = T.names_of(ST)
    for r in combos:
        os.makedirs(os.path.join(folder, r), exist_ok=True)
        for i in range(count):
            obj = bnd = ref_value(r, base)[i]
            status = 'OPTIMAL'
            if (r, i) in limit:
                x = limit[(r, i)]
                obj, bnd, status = x['objective'] * base[i], x['bound'] * base[i], x['status']
            warm = base[i] * (1.2 + 0.1 * R8.index(r))           # FIFO, the best rule
            rec = dict(set=ST, instance=names[i], index=i, regime=r, model='compact',
                       objective=float(obj), bound=float(bnd), status=status,
                       proved_optimal=status == 'OPTIMAL', gap_to_bound=(obj - bnd) / obj,
                       solve_time_s=3.0 if status == 'OPTIMAL' else 150.3, budget_s=150.0,
                       workers=2, seed=1, warm_makespan=float(warm), desc_seed=1000 + i,
                       rules_version=P.RULES_VERSION, cpu_share=0.99, cores=[0, 1],
                       affinity=[0, 1], validated=True, development_run=False, valid=True,
                       valid_reason='test')
            for k, v in ((edit or {}).get((r, i)) or {}).items():
                rec[k] = float(v * base[i]) if k in ('objective', 'bound', 'warm_makespan') \
                    and isinstance(v, float) and v < 5 else v
            with open(os.path.join(folder, r, f'{names[i]}.json'), 'w') as f:
                json.dump(rec, f)
    return folder


def run(root, out):
    return P.main(['--root', str(root), '--out', str(out), '--boot', str(B)])


def macros_of(out):
    return (out / 'manuscript' / 'macros_public.tex').read_text()


@pytest.fixture
def tree(tmp_path):
    return T.build_tree(str(tmp_path / 'tree'))


def test_skipped_while_partial_and_the_existing_part_unchanged(tree, tmp_path):
    res0 = run(tree, tmp_path / 'out0')                       # no reference record at all
    text0 = macros_of(tmp_path / 'out0')
    assert res0['references']['state']['total']['present'] == 0
    write_refs(tree, combos=('N', 'L'))
    write_refs(tree, combos=('W',), count=4)
    res = run(tree, tmp_path / 'out1')
    X = res['references']
    assert not X['computed'] and 'rows' not in X
    assert X['state']['complete'] == ['N', 'L'] and X['state']['per']['W']['valid'] == 4
    want = (f'{6 + 6 + 4} of {8 * N} references of {ST} are valid (16 present, 0 not counted); '
            f'valid per combination N 6, L 6, S 0, W 4, LS 0, LW 0, SW 0, LSW 0 of {N}; '
            f'the section needs all {8 * N}')
    assert X['state']['reason'] == want
    text = macros_of(tmp_path / 'out1')
    head0 = text0.rsplit('% Section 9', 1)[0]          # everything before the one skip line
    assert text.startswith(head0) and text.rsplit('% Section 9', 1)[0] == head0
    assert text.count('% Section 9') == 1 and 'PubRef' not in text
    assert text.rstrip('\n').endswith(f'skipped: {want}.')
    md = (tmp_path / 'out1' / 'results' / 'public_family_inference.md').read_text()
    assert md.rstrip('\n').endswith(f'Skipped: {want}.')
    # every number of sections 1 to 8 is the same with or without the records
    assert {k: v for k, v in res.items() if k not in ('references', 'n_macros_references')} == \
        {k: v for k, v in res0.items() if k not in ('references', 'n_macros_references')}
    assert X['best_known_N']['equal'] == N and X['best_known_N']['below'] == 0


def test_complete_section_sign_intervals_counts_and_macros(tree, tmp_path):
    write_refs(tree, edit={('W', 2): dict(warm_makespan=0.5)})   # a reference above its warm start
    res = run(tree, tmp_path / 'out')
    X = res['references']
    assert X['computed'] and X['combinations'] == R8
    base = bases(tree)
    I = P.Inputs(tree)
    I.check_records(ST)
    I.load_set(ST)
    D = {}
    for r in R8:
        ref = ref_value(r, base).copy()
        if r == 'LSW':
            ref[0] = 1.6 * base[0]
        d = np.stack([P.rel(I.ms[(ST, 'released', None, s, r)], ref) for s in P.SEEDS])
        D[r] = d
        x = X['rows'][r]['arms']['policy']
        hb = SF.hier_boot(d, B=B, seed=11)
        assert (x['mean'], x['lo'], x['hi']) == (hb['mean'], hb['lo'], hb['hi'])
        assert x['direction'] == SF.direction(hb['lo'], hb['hi'])
        assert set(x) >= {'equal', 'longer', 'shorter'} and not any(k.startswith('v') for k in x)
        dq = P.rel(np.min([I.rule[(ST, q, r)] for q in P.RULES], axis=0), ref)
        ib = SF.instance_boot(dq, B=B, seed=7)
        y = X['rows'][r]['rules']['best']
        assert (y['mean'], y['lo'], y['hi']) == (ib['mean'], ib['lo'], ib['hi'])
    # sign: method minus reference over the reference; the policy is longer everywhere on N
    want = 100 * (np.mean(list(T.SEED_FACTOR.values())) * 1.004 / 0.7 - 1)
    assert X['rows']['N']['arms']['policy']['mean'] == pytest.approx(want, rel=1e-9)
    assert X['rows']['N']['arms']['policy']['direction'] == 'behind'
    # pooled: one hierarchical bootstrap of [k, n, 8], a base instance with its combinations
    hb = SF.hier_boot(np.stack([D[r] for r in R8], axis=2), B=B, seed=11)
    p = X['pooled']['policy']
    assert (p['mean'], p['lo'], p['hi'], p['c']) == (hb['mean'], hb['lo'], hb['hi'], 8)
    # counts: 47 proved, one at the limit with a 25% gap; the policy shorter on two runs there
    assert X['proved'] == 8 * N - 1 and X['at_limit'] == 1
    row = X['rows']['LSW']
    assert (row['proved'], row['at_limit']) == (N - 1, 1)
    assert row['at_limit_gap_mean'] == pytest.approx(25.0)
    assert row['status'] == {'FEASIBLE': 1, 'OPTIMAL': N - 1}
    cases = X['policy_shorter']
    assert [(c['combination'], c['instance'], c['seed']) for c in cases] == \
        [('LSW', 'inst00', 303), ('LSW', 'inst00', 305)]
    assert row['arms']['policy']['shorter'] == pytest.approx(100 * 2 / 5 / N)
    assert X['unexpected']['worse_than_warm_start'][0]['instance'] == 'inst02'
    assert len(X['unexpected']['worse_than_warm_start']) == 1
    assert X['best_known_N'] == dict(X['best_known_N'], counted=N, below=0, equal=N, above=0)
    # macros: PubRef, letters only, unique, after every other macro, no verdict
    text = macros_of(tmp_path / 'out')
    names = re.findall(r'\\newcommand\{\\([^}]*)\}', text)
    assert len(names) == len(set(names))
    k = [i for i, n in enumerate(names) if n.startswith('PubRef')]
    assert k == list(range(len(names) - len(k), len(names))) == list(range(k[0], len(names)))
    assert len(k) == res['n_macros_references']
    assert all(re.fullmatch(r'PubRef[A-Za-z]+', names[i]) for i in k)
    assert not any(re.search(r'(VHalf|VOne|Verdict)$', names[i]) for i in k)
    mac = dict(re.findall(r'\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}', text))
    assert mac['PubRefN'] == str(N) and mac['PubRefBudget'] == '150' and mac['PubRefLimitN'] == '1'
    assert mac['PubRefLSWLimit'] == '1' and mac['PubRefLSWLimitGap'] == '25.00'
    assert mac['PubRefNLimitGap'] == '--' and mac['PubRefNProved'] == str(N)
    assert mac['PubRefPolShortN'] == '2' and mac['PubRefBkEqualN'] == str(N)
    assert mac['PubRefPolN'] == P.f2(X['rows']['N']['arms']['policy']['mean'])
    assert mac['PubRefPolNDir'] == 'behind' and mac['PubRefPolNEq'] == '0.0'
    assert {'PubRefSpecLSWHi', 'PubRefBestPoolDir', 'PubRefMwkrSWLo', 'PubRefCtlPool'} <= set(mac)
    md = (tmp_path / 'out' / 'results' / 'public_family_inference.md').read_text()
    assert '## 9. Gap to the references' in md and '| pooled | 48 | 47 | 1 |' in md


def set_table_proved(root, i):
    path = os.path.join(root, REG.PUB_SETS[ST])
    rows = list(csv.DictReader(open(path)))
    rows[i]['optimal'] = 'yes'
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_refuses_a_reference_below_a_proved_best_known_value(tree, tmp_path):
    out = tmp_path / 'out'
    set_table_proved(tree, 3)
    write_refs(tree, combos=('N',), edit={('N', 3): dict(objective=0.69, bound=0.69)})
    for p in [out / x for x in P.OUT_FILES]:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('stale')
    with pytest.raises(SystemExit) as e:
        run(tree, out)
    assert 'section 9' in str(e.value) and 'inst03 N: reference' in str(e.value)
    assert 'marks as proved optimal' in str(e.value)
    assert not any((out / x).exists() for x in P.OUT_FILES)
    # the same reference below a value the table does not mark proved: counted as below
    tree2 = T.build_tree(str(tmp_path / 'tree2'))
    write_refs(tree2, combos=('N',), edit={('N', 3): dict(objective=0.69, bound=0.69)})
    X = run(tree2, tmp_path / 'out2')['references']
    assert (X['best_known_N']['below'], X['best_known_N']['equal']) == (1, N - 1)
    assert X['best_known_N']['below_cases'][0]['instance'] == 'inst03'


def test_refuses_a_bound_above_a_best_known_or_a_stored_makespan(tree, tmp_path):
    write_refs(tree, combos=('N',), edit={('N', 1): dict(objective=0.7, bound=0.72,
                                                          status='FEASIBLE')})
    with pytest.raises(SystemExit) as e:
        run(tree, tmp_path / 'out')
    assert 'inst01 N: reference bound' in str(e.value) and 'above the best-known makespan' in str(e.value)
    tree2 = T.build_tree(str(tmp_path / 'tree2'))
    # a W bound above the stored makespan of the seed-303 runs (0.8 x 1.3 x base x ...)
    write_refs(tree2, combos=('N', 'W'), edit={('W', 4): dict(objective=1.25, bound=1.2,
                                                               status='FEASIBLE')})
    with pytest.raises(SystemExit) as e:
        run(tree2, tmp_path / 'out2')
    assert 'inst04 W: stored makespan' in str(e.value) and 'below the reference bound' in str(e.value)


@pytest.mark.parametrize('field,value,why', [
    ('budget_s', 300.0, 'budget 300.0 s, want 150'),
    ('valid', False, 'not marked valid'),
    ('development_run', True, 'development run'),
    ('desc_seed', 7, 'desc_seed 7, want 1005'),
    ('cpu_share', 0.5, 'record_validity says')])
def test_a_record_outside_the_campaign_or_not_valid_is_not_counted(tree, tmp_path, field, value, why):
    edit = {('SW', 5): {field: value}}
    if field == 'cpu_share':
        edit[('SW', 5)].update(status='FEASIBLE', objective=1.3, bound=1.0)
    write_refs(tree, edit=edit)
    X = run(tree, tmp_path / 'out')['references']
    assert not X['computed'] and X['state']['per']['SW'] == dict(present=N, valid=N - 1,
                                                                  proved=N - 1)
    assert X['not_counted_n'] == 1 and X['not_counted'][0].startswith('SW/inst05: ')
    assert why in X['not_counted'][0]
    assert '(48 present, 1 not counted)' in X['state']['reason']


def test_partial_reading_writes_only_its_two_files_outside_paper_and_results(tree, tmp_path):
    write_refs(tree, combos=('N', 'L'))
    write_refs(tree, combos=('W',), count=3)
    where = tmp_path / 'look'
    X = P.main(['--root', tree, '--partial-references', str(where), '--boot', str(B)])
    assert sorted(os.listdir(where)) == sorted(P.REF_PARTIAL_FILES)
    assert X['partial'] and X['combinations'] == ['N', 'L'] and set(X['rows']) == {'N', 'L'}
    assert X['pooled']['policy']['c'] == 2 and not X['computed']
    js = json.load(open(where / P.REF_PARTIAL_FILES[0]))
    assert js['state']['per']['W']['valid'] == 3
    md = (where / P.REF_PARTIAL_FILES[1]).read_text()
    assert 'PARTIAL READING: complete combinations N, L; incomplete S, W, LS, LW, SW, LSW' in md
    assert not any(os.path.exists(os.path.join(tree, d)) for d in ('manuscript', 'results'))
    for bad in (os.path.join(tree, 'results'), os.path.join(tree, 'manuscript', 'x'), tree,
                os.path.join(ROOT, 'paper'), os.path.join(ROOT, 'results', 'look')):
        with pytest.raises(SystemExit) as e:
            P.main(['--root', tree, '--partial-references', bad, '--boot', str(B)])
        assert 'never written into the repository root' in str(e.value)
    assert not os.path.exists(os.path.join(ROOT, 'results', 'look'))
