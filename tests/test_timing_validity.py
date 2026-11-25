"""Validity rule of time-budgeted solver cells (scripts/timing_validity.py)."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
import timing_validity as T  # noqa: E402

SLOT_A, SLOT_B = list(range(8, 16)), list(range(16, 24))


def _write(root, folder, regime, name, **kw):
    d = dict(hint='complete', validated=True, status='FEASIBLE', cpu_share=0.95,
             affinity=SLOT_A)
    d.update(kw)
    path = os.path.join(root, folder, regime)
    os.makedirs(path, exist_ok=True)
    p = os.path.join(path, name + '.json')
    json.dump(d, open(p, 'w'))
    return p


def test_default_mask_is_the_sixteen_core_set(tmp_path):
    root = str(tmp_path)
    ok = _write(root, 'armA', 'N', 'i0', affinity=list(range(16)))
    bad = _write(root, 'armA', 'N', 'i1', affinity=SLOT_A)
    out = T.classify_folders(root, ['armA'], thresh=os.path.join(root, 'none.json'))
    assert out[ok] == (True, 'share')
    assert out[bad][0] is False and 'affinity' in out[bad][1]


def test_campaign_slots_are_accepted_and_others_refused(tmp_path):
    root = str(tmp_path)
    a = _write(root, 'armA', 'S', 'i0', affinity=SLOT_A)
    b = _write(root, 'armA', 'S', 'i1', affinity=SLOT_B)
    c = _write(root, 'armA', 'S', 'i2', affinity=list(range(10, 16)))
    out = T.classify_folders(root, ['armA'], slots=[SLOT_A, SLOT_B],
                             thresh=os.path.join(root, 'none.json'))
    assert out[a] == (True, 'share') and out[b] == (True, 'share')
    assert out[c][0] is False


def test_low_share_is_pending_without_a_bar_and_judged_with_one(tmp_path):
    root = str(tmp_path)
    low = _write(root, 'armA', 'S', 'i0', cpu_share=0.80)
    opt = _write(root, 'armA', 'S', 'i1', cpu_share=0.40, status='OPTIMAL')
    unval = _write(root, 'armA', 'S', 'i2', validated=False)
    none = os.path.join(root, 'none.json')
    out = T.classify_folders(root, ['armA'], slots=[SLOT_A], thresh=none)
    assert out[low][0] is None
    assert out[opt] == (True, 'optimal')
    assert out[unval] == (False, 'not validated')
    bar = os.path.join(root, 'bars.json')
    json.dump({T._key(low): 0.84}, open(bar, 'w'))
    assert T.classify([low], slots=[SLOT_A], thresh=bar)[low] == (True, 'type')
    json.dump({T._key(low): 0.90}, open(bar, 'w'))
    assert T.classify([low], slots=[SLOT_A], thresh=bar)[low][0] is False


def test_single_thread_runs_need_one_core_of_a_slot(tmp_path):
    root = str(tmp_path)
    base = dict(validated=True, budget_clock='wall', threads=1, cpu_share=0.99)
    ok = _write(root, 'm', 'S', 'i0_seed1', affinity=[9], **base)
    two = _write(root, 'm', 'S', 'i1_seed1', affinity=[9, 10], **base)
    out_ = _write(root, 'm', 'S', 'i2_seed1', affinity=[3], **base)
    low = _write(root, 'm', 'S', 'i3_seed1', affinity=[17], **dict(base, cpu_share=0.85))
    cpu = _write(root, 'm', 'S', 'i4_seed1', affinity=[17], **dict(base, budget_clock='cpu'))
    out = T.classify_single_thread([ok, two, out_, low, cpu], slots=[SLOT_A, SLOT_B])
    assert out[ok] == (True, 'share')
    assert out[two][0] is False and out[out_][0] is False
    assert out[low][0] is False and 'share' in out[low][1]
    assert out[cpu][0] is False and 'clock' in out[cpu][1]


def test_cparms_key_separates_output_folder_and_size(tmp_path):
    a = os.path.join(str(tmp_path), '50x25n30_cparms', 'pilot', 'lns', 'S', 'instance_000.json')
    b = os.path.join(str(tmp_path), '80x25n30_cparms', 'evidence_rule1800s', 'lns', 'N', 'x.json')
    assert T.key_cparms(a) == 'pilot|lns|50x25n30|setup'
    assert T.key_cparms(b) == 'evidence_rule1800s|lns|80x25n30|nosetup'
    for p in (a, b):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        json.dump(dict(hint='complete', validated=True, status='FEASIBLE', cpu_share=0.8,
                       affinity=SLOT_A), open(p, 'w'))
    bar = os.path.join(str(tmp_path), 'bars.json')
    json.dump({T.key_cparms(a): 0.82}, open(bar, 'w'))
    out = T.classify([a, b], slots=[SLOT_A], thresh=bar, key=T.key_cparms)
    assert out[a] == (True, 'type') and out[b][0] is None
    assert T.type_bars([a, b], key=T.key_cparms) == {T.key_cparms(a): 0.8, T.key_cparms(b): 0.8}
