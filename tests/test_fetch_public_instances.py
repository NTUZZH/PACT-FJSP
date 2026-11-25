"""Tests for the download script of the public instance files
(scripts/fetch_public_instances.py) and its manifest data/PUBLIC/sources.json.

  * on a small archive and manifest built in a temporary folder (read through
    file:// URLs, so no network is used): only the members the manifest names
    are extracted, bytes unchanged (CRLF line ends survive); an archive whose
    SHA-256 differs from the pin is refused before it is opened, downloaded or
    given with --archive; a member named with '..', an absolute path or a link
    is refused and nothing of that archive is written; a member whose bytes
    differ from SHA256SUMS is not written; a local file with one altered byte
    gives a non-zero exit code and is never overwritten; files already verified
    are left alone and no source is contacted; --sets selects; the zip of a
    repository commit is read below its top folder; a network failure gives one
    message naming the URL and --archive, without a traceback;
  * --sd2 generate writes SD2/test/10x5+mix byte for byte as in its SHA256SUMS;
  * --verify-only passes on the real data/PUBLIC for every set of the manifest
    (skipped when the instance files are absent);
  * the manifest names exactly the base folders of data/PUBLIC/family/*, and
    every such folder has a SHA256SUMS with the manifest's file count;
  * no absolute path and no user or machine name in the manifest, in the
    READMEs of data/PUBLIC that describe the download, or in the messages.

Run from the repository root:  python3 -m pytest tests/test_fetch_public_instances.py -q
Offline, CPU only.
"""
import getpass
import glob
import hashlib
import json
import os
import re
import socket
import stat
import subprocess
import sys
import tempfile
import zipfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import fetch_public_instances as F  # noqa: E402

SCRIPT = os.path.join(ROOT, 'scripts', 'fetch_public_instances.py')
PUB = os.path.join(ROOT, 'data', 'PUBLIC')
MANIFEST = os.path.join(PUB, 'sources.json')
READMES = ['README.md', 'Brandimarte/README.md', 'Hurink_edata/README.md',
           'Hurink_rdata/README.md', 'Hurink_vdata/README.md', 'SD2/README.md']

ONE = {'Mk01.fjs': b'2 2 1\r\n2 1 1 5 1 2 3\r\n1 2 1 4 2 6\r\n',
       'Mk02.fjs': b'1 2 2\r\n1 2 1 7 2 9\r\n'}
TWO = {'la01.fjs': b'1 1 1\r\n1 1 1 8\r\n'}
COMMIT = '0123456789abcdef0123456789abcdef01234567'


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def _sums_text(files):
    return ''.join(f'{_sha(b)}  {n}\n' for n, b in sorted(files.items()))


def _zip(path, members):
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
        for name, data in members:
            z.writestr(name, data)
    return path


def _library(tmp_path, sets, archive, url=None, pin=None):
    """A manifest folder: sources.json plus <set>/SHA256SUMS.

    sets: {set name: (path inside the archive, {file name: bytes})}."""
    lib = tmp_path / 'lib'
    recs = []
    for name, (path, files) in sets.items():
        d = lib.joinpath(*name.split('/'))
        d.mkdir(parents=True, exist_ok=True)
        (d / 'SHA256SUMS').write_text(_sums_text(files))
        recs.append({'set': name, 'source': url or archive.as_uri(),
                     'archive_sha256': pin or _sha(archive.read_bytes()),
                     'path': path, 'files': len(files), 'citation': 'test'})
    (lib / 'sources.json').write_text(json.dumps({'sets': recs}))
    return lib / 'sources.json'


def _standard(tmp_path, url=None, pin=None, one=ONE):
    members = [(f'Lib/Text/one/{n}', b) for n, b in one.items()]
    members += [(f'Lib/Text/two/{n}', b) for n, b in TWO.items()]
    members += [('Lib/Text/one/extra.fjs', b'1 1 1\r\n1 1 1 1\r\n'), ('Lib/read.me', b'x')]
    archive = _zip(tmp_path / 'TextData.zip', members)
    man = _library(tmp_path, {'S1': ('Lib/Text/one', ONE), 'S2': ('Lib/Text/two', TWO)},
                   archive, url=url, pin=pin)
    return man, archive


def _run(capsys, *args):
    code = F.main([str(a) for a in args])
    out = capsys.readouterr()
    return code, out.out + out.err


def _files(folder):
    return sorted(os.path.relpath(os.path.join(r, f), folder).replace(os.sep, '/')
                  for r, _, fs in os.walk(folder) for f in fs)


def _closed_port_url():
    s = socket.socket()
    s.bind(('127.0.0.1', 0))
    port = s.getsockname()[1]
    s.close()
    return f'http://127.0.0.1:{port}/TextData.zip'


# ----------------------------------------------------------------------
# small archive and manifest
# ----------------------------------------------------------------------

def test_extracts_only_named_members_bytes_unchanged(tmp_path, capsys):
    man, _ = _standard(tmp_path)
    dest = tmp_path / 'dest'
    code, out = _run(capsys, '--manifest', man, '--dest', dest)
    assert code == 0, out
    assert _files(dest) == ['S1/Mk01.fjs', 'S1/Mk02.fjs', 'S2/la01.fjs']
    for n, b in ONE.items():
        got = (dest / 'S1' / n).read_bytes()
        assert got == b and b'\r\n' in got                  # CRLF survives
    assert (dest / 'S2' / 'la01.fjs').read_bytes() == TWO['la01.fjs']
    lines = out.strip().splitlines()
    assert lines[-2:] == ['S1: expected 2, present 2, verified 2, missing 0, differing 0',
                          'S2: expected 1, present 1, verified 1, missing 0, differing 0']


def test_local_archive_is_used_without_download(tmp_path, capsys):
    man, archive = _standard(tmp_path, url=_closed_port_url())
    dest = tmp_path / 'dest'
    code, out = _run(capsys, '--manifest', man, '--dest', dest, '--archive', archive)
    assert code == 0, out
    assert 'downloading' not in out
    assert _files(dest) == ['S1/Mk01.fjs', 'S1/Mk02.fjs', 'S2/la01.fjs']


def test_wrong_archive_checksum_refuses_before_opening(tmp_path, capsys, monkeypatch):
    man, archive = _standard(tmp_path, pin='0' * 64)
    dest = tmp_path / 'dest'

    def never_opened(*a, **k):
        raise AssertionError('the archive was opened')
    monkeypatch.setattr(F.zipfile, 'ZipFile', never_opened)
    code, out = _run(capsys, '--manifest', man, '--dest', dest)
    assert code == 1
    assert 'SHA-256' in out and 'not opened' in out
    code, out = _run(capsys, '--manifest', man, '--dest', dest, '--archive', archive)
    assert code == 1
    assert 'SHA-256' in out and 'not opened' in out
    assert not dest.exists() or _files(dest) == []
    assert 'S1: expected 2, present 0, verified 0, missing 2, differing 0' in out


def test_altered_local_byte_is_reported_and_never_overwritten(tmp_path, capsys):
    man, _ = _standard(tmp_path)
    dest = tmp_path / 'dest'
    assert _run(capsys, '--manifest', man, '--dest', dest)[0] == 0
    p = dest / 'S1' / 'Mk01.fjs'
    altered = bytearray(p.read_bytes())
    altered[0] = ord('3')
    p.write_bytes(bytes(altered))
    before = p.stat().st_mtime_ns
    (dest / 'S1' / 'Mk02.fjs').unlink()

    code, out = _run(capsys, '--manifest', man, '--dest', dest, '--verify-only')
    assert code == 1
    assert 'S1: expected 2, present 1, verified 0, missing 1, differing 1' in out
    assert 'differs: <dest>/S1/Mk01.fjs' in out

    code, out = _run(capsys, '--manifest', man, '--dest', dest)
    assert code == 1
    assert p.read_bytes() == bytes(altered) and p.stat().st_mtime_ns == before
    assert (dest / 'S1' / 'Mk02.fjs').read_bytes() == ONE['Mk02.fjs']
    assert 'S1: expected 2, present 2, verified 1, missing 0, differing 1' in out
    assert 'S2: expected 1, present 1, verified 1, missing 0, differing 0' in out


def test_source_bytes_differing_from_sums_are_not_written(tmp_path, capsys):
    bad = dict(ONE, **{'Mk02.fjs': ONE['Mk02.fjs'].replace(b'9', b'8')})
    man, _ = _standard(tmp_path, one=bad)
    dest = tmp_path / 'dest'
    code, out = _run(capsys, '--manifest', man, '--dest', dest)
    assert code == 1
    assert not (dest / 'S1' / 'Mk02.fjs').exists()
    assert (dest / 'S1' / 'Mk01.fjs').read_bytes() == ONE['Mk01.fjs']
    assert 'S1: expected 2, present 1, verified 1, missing 1, differing 0' in out


def _link_info(name):
    info = zipfile.ZipInfo(name)
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    return info


@pytest.mark.parametrize('path,member', [('Lib/..', 'Lib/../evil.fjs'),
                                         ('/abs', '/abs/evil.fjs'),
                                         ('Lib', 'Lib/evil.fjs')],
                         ids=['dotdot', 'absolute', 'link'])
def test_refuses_dotdot_absolute_and_link_members(tmp_path, capsys, path, member):
    evil, good = b'1 1 1\n1 1 1 1\n', b'1 1 1\n1 1 1 2\n'
    info = _link_info(member) if path == 'Lib' else member
    archive = _zip(tmp_path / 'TextData.zip', [('Good/good.fjs', good), (info, evil)])
    # S2 is planned first and is legitimate: a refusal in S1 still writes nothing
    man = _library(tmp_path, {'S2': ('Good', {'good.fjs': good}),
                              'S1': (path, {'evil.fjs': evil})}, archive)
    dest = tmp_path / 'dest'
    code, out = _run(capsys, '--manifest', man, '--dest', dest)
    assert code == 1
    assert f'member {member} is refused' in out and 'nothing was extracted' in out
    assert not dest.exists() or _files(dest) == []
    assert not (tmp_path / 'evil.fjs').exists()


def test_sets_selects(tmp_path, capsys):
    man, _ = _standard(tmp_path)
    dest = tmp_path / 'dest'
    code, out = _run(capsys, '--manifest', man, '--dest', dest, '--sets', 'S2')
    assert code == 0, out
    assert _files(dest) == ['S2/la01.fjs']
    assert 'S1:' not in out
    with pytest.raises(SystemExit) as e:
        F.main(['--manifest', str(man), '--sets', 'S3'])
    assert e.value.code == 2


def test_verified_files_left_alone_without_network(tmp_path, capsys):
    man, _ = _standard(tmp_path)
    dest = tmp_path / 'dest'
    assert _run(capsys, '--manifest', man, '--dest', dest)[0] == 0
    stamps = {f: os.stat(dest / f).st_mtime_ns for f in _files(dest)}
    m = json.loads(man.read_text())
    for r in m['sets']:
        r['source'] = _closed_port_url()        # a download would fail
    man.write_text(json.dumps(m))
    code, out = _run(capsys, '--manifest', man, '--dest', dest)
    assert code == 0, out
    assert 'downloading' not in out and 'S1: no file missing' in out
    assert {f: os.stat(dest / f).st_mtime_ns for f in _files(dest)} == stamps


def test_repository_commit_archive(tmp_path, capsys):
    root = f'Repo-{COMMIT}'
    members = [(f'{root}/data/X/{n}', b) for n, b in TWO.items()]
    members += [(f'{root}/README.md', b'readme')]
    archive = _zip(tmp_path / 'commit.zip', members)
    lib = tmp_path / 'lib'
    (lib / 'R').mkdir(parents=True)
    (lib / 'R' / 'SHA256SUMS').write_text(_sums_text(TWO))
    rec = {'set': 'R', 'source': 'https://example.org/Repo', 'commit': COMMIT,
           'archive_url': archive.as_uri(), 'archive_root': root, 'path': 'data/X',
           'files': 1, 'citation': 'test'}
    man = lib / 'sources.json'
    man.write_text(json.dumps({'sets': [rec]}))
    code, out = _run(capsys, '--manifest', man, '--dest', tmp_path / 'd1')
    assert code == 0, out
    assert (tmp_path / 'd1' / 'R' / 'la01.fjs').read_bytes() == TWO['la01.fjs']

    rec['archive_url'] = _closed_port_url()
    man.write_text(json.dumps({'sets': [rec]}))
    code, out = _run(capsys, '--manifest', man, '--dest', tmp_path / 'd2', '--archive', archive)
    assert code == 0, out
    assert _files(tmp_path / 'd2') == ['R/la01.fjs']

    other = _zip(tmp_path / 'other.zip', [(f'Repo-{"f" * 40}/data/X/la01.fjs', TWO['la01.fjs'])])
    code, out = _run(capsys, '--manifest', man, '--dest', tmp_path / 'd3', '--archive', other)
    assert code == 1 and 'matches no source' in out
    assert not (tmp_path / 'd3').exists()


def test_network_failure_one_message_no_traceback(tmp_path):
    url = _closed_port_url()
    man, _ = _standard(tmp_path, url=url)
    r = subprocess.run([sys.executable, '-B', SCRIPT, '--manifest', str(man),
                        '--dest', str(tmp_path / 'dest')],
                       capture_output=True, text=True, timeout=120)
    out = r.stdout + r.stderr
    assert r.returncode == 1
    assert 'Traceback' not in out
    errors = [ln for ln in out.splitlines() if ln.startswith('error:')]
    assert len(errors) == 1 and url in errors[0] and '--archive' in errors[0]
    assert 'S1: expected 2, present 0, verified 0, missing 2, differing 0' in out


# ----------------------------------------------------------------------
# the real manifest and data/PUBLIC
# ----------------------------------------------------------------------

def _manifest():
    with open(MANIFEST) as f:
        return json.load(f)


def _instance_files_present():
    for r in _manifest()['sets']:
        d = os.path.join(PUB, *r['set'].split('/'))
        for name in F.read_sums(os.path.join(d, 'SHA256SUMS'), r['set']):
            if not os.path.isfile(os.path.join(d, name)):
                return False
    return True


def test_generate_route_matches_sums(tmp_path):
    pytest.importorskip('numpy')
    pytest.importorskip('torch')
    dest = tmp_path / 'dest'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    r = subprocess.run([sys.executable, '-B', SCRIPT, '--sd2', 'generate',
                        '--sets', 'SD2/test/10x5+mix', '--dest', str(dest)],
                       capture_output=True, text=True, timeout=300, env=env, cwd=ROOT)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    assert 'SD2/test/10x5+mix: expected 100, present 100, verified 100, missing 0, ' \
           'differing 0' in out
    assert len(_files(dest)) == 100


def test_verify_only_on_data_public(capsys):
    if not _instance_files_present():
        pytest.skip('instance files absent; run scripts/fetch_public_instances.py first')
    code, out = _run(capsys, '--verify-only')
    assert code == 0, out
    for r in _manifest()['sets']:
        n = r['files']
        assert f'{r["set"]}: expected {n}, present {n}, verified {n}, missing 0, ' \
               f'differing 0' in out


def test_manifest_matches_family_base_dirs():
    recs = {r['set']: r for r in _manifest()['sets']}
    metas = sorted(glob.glob(os.path.join(PUB, 'family', '*', 'dataset_meta.json')))
    assert metas
    base = {}
    for p in metas:
        with open(p) as f:
            m = json.load(f)
        assert m['base_dir'].startswith('data/PUBLIC/'), p
        base.setdefault(m['base_dir'][len('data/PUBLIC/'):], []).append(m)
    assert set(recs) == set(base)
    for name, r in recs.items():
        sums = F.read_sums(os.path.join(PUB, *name.split('/'), 'SHA256SUMS'), name)
        assert len(sums) == r['files'], name
        for m in base[name]:
            assert m['n_instances'] + len(m['excluded']) == r['files'], name
            assert {f'{i}.fjs' for i in m['instances']} <= set(sums), name
        assert ('archive_sha256' in r) != ('commit' in r), name
        if 'commit' in r:
            assert r['archive_root'].endswith(r['commit']), name
            assert r['commit'] in r['archive_url'], name


def _forbidden(text, extra=()):
    """Absolute paths and user or machine names found in `text`."""
    hits = re.findall(r'(?:^|[\s`(\'"=])(/(?:home|Users|tmp|root|mnt|media|var)/\S*)', text)
    hits += re.findall(r'\b[A-Za-z]:\\\S*', text)
    names = set()
    for fn in (getpass.getuser, socket.gethostname):
        try:
            names.add(fn())
        except Exception:
            pass
    for n in names:
        if len(n) >= 5 and re.search(rf'(?<![\w-]){re.escape(n)}(?![\w-])', text):
            hits.append(n)
    for s in (ROOT, os.path.expanduser('~'), tempfile.gettempdir()) + tuple(extra):
        if s and len(s) > 1 and s in text:
            hits.append(s)
    return hits


def test_no_absolute_path_or_user_name(tmp_path, capsys):
    with open(MANIFEST) as f:
        assert _forbidden(f.read()) == []
    for rel in READMES:
        with open(os.path.join(PUB, rel)) as f:
            assert _forbidden(f.read()) == [], rel
    man, archive = _standard(tmp_path, url=_closed_port_url())
    dest = tmp_path / 'dest'
    outs = [_run(capsys, '--manifest', man, '--dest', dest, '--archive', archive)[1]]
    (dest / 'S1' / 'Mk01.fjs').write_bytes(b'0')
    outs.append(_run(capsys, '--manifest', man, '--dest', dest, '--verify-only')[1])
    outs.append(_run(capsys, '--manifest', man, '--dest', tmp_path / 'other',
                     '--archive', tmp_path / 'lib' / 'sources.json')[1])
    outs.append(_run(capsys, '--verify-only', '--sets', 'Brandimarte')[1])
    for out in outs:
        assert out.strip()
        assert _forbidden(out, extra=(str(tmp_path),)) == [], out
