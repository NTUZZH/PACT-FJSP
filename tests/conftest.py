"""Settings shared by the tests.

Every test file runs as it runs alone, in whatever order pytest collects the files: a file
that needs a run setting (PACT_ENV, PACT_SEEDS, PACT_OUT, ...) declares it with
module_environment(), and each file imports the modules of scripts/ and tests/ afresh, under
its own setting (see the section "each file as if run alone" below).

    from conftest import module_environment  # noqa: E402
    MODULE_ENVIRONMENT = module_environment(PACT_ENV='exact', PACT_SEEDS='301-305')

The instance files of the public benchmark (data/PUBLIC/<set>/*.fjs) are not part of the
repository; scripts/fetch_public_instances.py downloads them and checks every file
against the SHA256SUMS of its set folder. A test that reads these files carries the mark
of needs_public_instances() and skips, with one message that names the command, while
a file of its sets is absent.

    from conftest import needs_public_instances

    @needs_public_instances()                  # every set of data/PUBLIC/sources.json
    def test_reads_an_instance(): ...

    @needs_public_instances('Fattahi_setup')   # only the named set folders
    def test_reads_one_set(): ...
"""
import functools
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUBLIC = os.path.join(ROOT, 'data', 'PUBLIC')
FETCH_COMMAND = 'python scripts/fetch_public_instances.py'
SKIP_MESSAGE = ('needs the public instance files (data/PUBLIC/<set>/*.fjs); '
                f'run `{FETCH_COMMAND}` from the repository root')


@functools.lru_cache(maxsize=None)
def manifest_sets():
    """The set names of data/PUBLIC/sources.json, in the order of the file; an empty
    tuple when the file is absent."""
    path = os.path.join(PUBLIC, 'sources.json')
    if not os.path.isfile(path):
        return ()
    with open(path, encoding='utf-8') as f:
        return tuple(s['set'] for s in json.load(f)['sets'])


@functools.lru_cache(maxsize=None)
def set_present(name):
    """True when the folder data/PUBLIC/<name> holds every file its SHA256SUMS names;
    False when the folder, its SHA256SUMS or one of these files is absent."""
    sums = os.path.join(PUBLIC, name, 'SHA256SUMS')
    if not os.path.isfile(sums):
        return False
    with open(sums, encoding='utf-8') as f:
        names = [ln.split(None, 1)[1].strip() for ln in f if ln.strip()]
    return bool(names) and all(os.path.isfile(os.path.join(PUBLIC, name, n)) for n in names)


def instances_present(*sets):
    """True when every file of the named sets is present; with no name, every set of
    data/PUBLIC/sources.json."""
    names = sets or manifest_sets()
    return bool(names) and all(set_present(s) for s in names)


def skip_message(*sets):
    """The skip message: the command, and for a named set that sources.json does not
    list, the remark that the command does not download it."""
    extra = [s for s in sets if s not in manifest_sets()]
    if not extra:
        return SKIP_MESSAGE
    return (f'{SKIP_MESSAGE}; {", ".join(extra)} is not among the sets of '
            'data/PUBLIC/sources.json and is not downloaded by it')


def needs_public_instances(*sets):
    """A skip mark for a test, or for a pytest.param, that reads instance files of the
    named sets (default: every set of data/PUBLIC/sources.json)."""
    return pytest.mark.skipif(not instances_present(*sets), reason=skip_message(*sets))


# ----------------------------------------------------------------- each file as if run alone
# pytest imports every test file at collection, before the first test runs, and Python keeps a
# module after its first import. A file that sets PACT_ENV or PACT_SEEDS at import therefore
# changed the setting under which every later file was imported and every earlier file's tests
# ran, and a script that reads the setting when it is imported (p72_aware_rules sets ENV,
# p88_family_inference sets SEEDS, K, WORDS and SOLVER_ROWS from arm_registry; p33_ood_table,
# p50_actmask_analysis, p82_best3, p95_secondary_inference and others do the same) kept the
# setting of the first file that imported it. A module setting that a test changes without
# restoring it (p88_family_inference.configure sets B) also stayed for every later file. The
# two hooks below make every file run as it runs alone, in any order:
#
#   * a file that needs a run setting declares it once, before its own imports, with
#     module_environment() below; None removes a variable:
#
#         from conftest import module_environment  # noqa: E402
#         MODULE_ENVIRONMENT = module_environment(PACT_ENV='exact', PACT_SEEDS='301-305',
#                                                 PACT_OUT=None, PACT_RESULTS=None)
#
#     The call applies the setting at once, for the file's own imports. When the file has been
#     collected, every PACT_ variable and every declared name is put back as it was, so the
#     setting reaches no other file's import; while the file's tests run, the fixture
#     file_as_if_alone applies the declaration again with monkeypatch (to os.environ, which the
#     subprocesses of the tests inherit) and undoes it after the file's last test; it then puts
#     every PACT_ variable and every declared name back as before the file's first test, so a
#     change a test makes without monkeypatch stays inside its file. A test body that changes a
#     variable uses monkeypatch.setenv or monkeypatch.delenv.
#   * every file is imported with no module of scripts/ or tests/ (conftest excepted) in
#     sys.modules, so the scripts it imports are imported under its own setting. These modules
#     are kept with the file, put back into sys.modules while its tests run (an import inside a
#     test finds them, and a script the file did not import is imported under the file's
#     setting) and taken out after its last test. Modules outside scripts/ and tests/
#     (constraint_family, params, torch, ...) are shared: none of them reads these variables
#     when it is imported (constraint_family.fjsp_env_family reads PACT_FIRST_MASK and
#     PACT_MASK_TOL when called).
#
# The tests run on the CPU. The CUDA driver reads CUDA_VISIBLE_DEVICES once, at the first query of
# the device count in the process, and keeps that answer; `import train_family` sets the variable
# to configs.device_id ('0') and its setup_seed then asks torch.cuda.is_available(). In a file
# order where that query came first, the GPU stayed visible to every later test, and
# p101_mask_controls._cpu_torch refused ("CUDA is visible"). The suite therefore sets the variable
# to '' and asks the device count once, here, before any test file is imported, so no test in
# the process sees a GPU, whatever it sets the variable to later; the variable is also put back
# with the PACT_ variables after each file (subprocesses inherit it).
ENV_FIXED = {'CUDA_VISIBLE_DEVICES': ''}
os.environ.update(ENV_FIXED)
try:
    import torch
except ImportError:                    # nothing to hide from where torch is not installed
    torch = None
if torch is not None:
    torch.cuda.is_available()          # the driver fixes its device list now: none visible

ISOLATED = (os.path.join(ROOT, 'scripts') + os.sep, os.path.join(ROOT, 'tests') + os.sep)
CONFTEST = os.path.abspath(__file__)
DECLARED = set()                       # every name a module_environment call has set or removed
OWN_MODULES = pytest.StashKey()        # the modules of scripts/ and tests/ a file imported


def module_environment(**setting):
    """Apply the run setting of a test file now (value None: remove the variable) and return
    it; the file keeps it as MODULE_ENVIRONMENT, which file_as_if_alone applies to its tests."""
    for name, value in setting.items():
        DECLARED.add(name)
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = str(value)
    return dict(setting)


def _repository_module(module):
    path = getattr(module, '__file__', None)
    if not isinstance(path, str):
        return False
    path = os.path.abspath(path)
    return path != CONFTEST and path.startswith(ISOLATED)


def take_repository_modules():
    """Remove every module of scripts/ and tests/ (conftest excepted) from sys.modules and
    return them as {name: module}."""
    names = [n for n, m in list(sys.modules.items()) if _repository_module(m)]
    return {n: sys.modules.pop(n) for n in names}


def _managed(name):
    return name.startswith('PACT_') or name in DECLARED or name in ENV_FIXED


def put_back_managed(before):
    """Every PACT_ variable, every declared name and CUDA_VISIBLE_DEVICES as in the mapping
    before (an os.environ copy): removed when it was absent, set to its former value otherwise."""
    for name in [n for n in os.environ if _managed(n) and n not in before]:
        del os.environ[name]
    for name, value in before.items():
        if _managed(name) and os.environ.get(name) != value:
            os.environ[name] = value


@pytest.hookimpl(wrapper=True)
def pytest_make_collect_report(collector):
    """Import a test file with no module of scripts/ or tests/ loaded, keep the modules it
    imported with it, and put back every PACT_ variable and every declared name afterwards."""
    if not isinstance(collector, pytest.Module):
        return (yield)
    before = dict(os.environ)
    outer = take_repository_modules()
    try:
        return (yield)
    finally:
        collector.stash[OWN_MODULES] = take_repository_modules()
        sys.modules.update(outer)
        put_back_managed(before)


@pytest.fixture(scope='module', autouse=True)
def file_as_if_alone(request):
    """While the tests of a file run: the modules it imported in sys.modules, and its declared
    setting (MODULE_ENVIRONMENT) in os.environ; both undone after its last test, and every
    PACT_ variable and declared name put back as before its first test."""
    before = dict(os.environ)
    outer = take_repository_modules()
    sys.modules.update(request.node.getparent(pytest.Module).stash.get(OWN_MODULES, {}))
    try:
        with pytest.MonkeyPatch.context() as mp:
            for name, value in (getattr(request.module, 'MODULE_ENVIRONMENT', None) or {}).items():
                if value is None:
                    mp.delenv(name, raising=False)
                else:
                    mp.setenv(name, str(value))
            yield
    finally:
        take_repository_modules()
        sys.modules.update(outer)
        put_back_managed(before)
