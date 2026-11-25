#!/usr/bin/env python3
"""Download the public flexible job-shop instances and check them against SHA256SUMS.

The instance files of data/PUBLIC (Brandimarte Mk01 to Mk10; Hurink edata, rdata
and vdata; the SD2 test and validation sets at 10x5 and 20x10) are not part of the
repository. data/PUBLIC/sources.json names, for every set, its source, the pin of the
source, the folder of the files inside the source and their number:

  * Brandimarte and Hurink: the FJSPLIB archive TextData.zip, pinned by its SHA-256;
  * SD2: the repository github.com/wrqccc/FJSP-DRL, pinned by a commit and read from
    the zip archive of that commit.

The checksum of every file is in the SHA256SUMS of its set folder, next to
sources.json.

Usage, from the repository root:

    python scripts/fetch_public_instances.py                 # every set of the manifest
    python scripts/fetch_public_instances.py --sets Brandimarte SD2/test/10x5+mix
    python scripts/fetch_public_instances.py --verify-only   # no network: local files against SHA256SUMS
    python scripts/fetch_public_instances.py --archive TextData.zip   # no download of the archive
    python scripts/fetch_public_instances.py --dest <folder> # default data/PUBLIC
    python scripts/fetch_public_instances.py --sd2 generate  # SD2 from the repository's generator, no network

--archive takes one or more local archive files: a copy of TextData.zip, recognised
by its pinned SHA-256, or the zip of the pinned FJSP-DRL commit
(github.com/wrqccc/FJSP-DRL/archive/<commit>.zip), recognised by its top folder
FJSP-DRL-<commit>.

Behaviour:

  * an archive is downloaded into a temporary folder (tempfile, honours TMPDIR); an
    archive pinned by a SHA-256 is checked before it is opened, and a mismatch
    stops that source with nothing extracted;
  * only the members the manifest names are read (<path>/<name> for every name of
    the set's SHA256SUMS); a member named with '..', an absolute path or a link is
    refused, and then nothing of that archive is written; a member is written by
    its base name into the set folder, bytes unchanged (the FJSPLIB files have
    CRLF line ends);
  * --sd2 generate writes the SD2 sets with data_utils.generate_data_to_files and
    the seeds of the manifest (200 for the test sets, 100 for the validation sets);
    the files pass the same checks as downloaded files;
  * a file that exists and equals its checksum is left alone; a file that exists
    and differs is never overwritten and is reported; every written file is read
    back and checked; a source is contacted only when a file it provides is missing;
  * the run ends with one line per set (files expected, present, verified, missing,
    differing); the exit code is 0 when every file of every selected set is present
    and verified, 1 otherwise, and 2 on a usage error;
  * messages name paths relative to the repository root (a destination outside the
    repository is shown as <dest>), and requests carry the generic client name
    below and no other identifying header.

The download path uses the Python standard library only; --sd2 generate imports
data_utils, and with it numpy and torch, from the repository.
"""
import argparse
import contextlib
import hashlib
import http.client
import io
import json
import os
import re
import stat
import sys
import tempfile
import types
import urllib.error
import urllib.parse
import urllib.request
import zipfile
import zlib

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MANIFEST = os.path.join(REPO_ROOT, 'data', 'PUBLIC', 'sources.json')
SUMS_NAME = 'SHA256SUMS'
USER_AGENT = 'fjsp-instance-fetch/1.0'
TIMEOUT_S = 60
CHUNK = 1 << 16
MAX_DOWNLOAD = 64 * 1024 * 1024      # bytes of one archive
MAX_MEMBER = 16 * 1024 * 1024        # bytes of one instance file
_HEX64 = re.compile(r'[0-9a-f]{64}')


class FetchError(Exception):
    """A failure reported as one message, without a traceback."""


# ----------------------------------------------------------------------
# manifest and checksums
# ----------------------------------------------------------------------

def _plain_name(name):
    return (bool(name) and name not in ('.', '..') and '/' not in name
            and '\\' not in name and '\0' not in name and ':' not in name)


def read_sums(path, label):
    """{file name: sha256} of one SHA256SUMS file (sha256sum text format)."""
    sums = {}
    try:
        with open(path, encoding='ascii') as f:
            lines = f.read().splitlines()
    except FileNotFoundError:
        raise FetchError(f'{label}: no {SUMS_NAME} file') from None
    except (OSError, UnicodeDecodeError) as e:
        raise FetchError(f'{label}: {SUMS_NAME} cannot be read ({e.__class__.__name__})') from None
    for k, line in enumerate(lines, 1):
        if not line.strip():
            continue
        digest, sep, name = line[:64].lower(), line[64:66], line[66:]
        if not _HEX64.fullmatch(digest) or sep not in ('  ', ' *'):
            raise FetchError(f'{label}: line {k} of {SUMS_NAME} is not "<sha256>  <name>"')
        if not _plain_name(name):
            raise FetchError(f'{label}: line {k} of {SUMS_NAME} names {name!r}, '
                             f'which is not a plain file name')
        if name in sums:
            raise FetchError(f'{label}: {name} is listed twice in {SUMS_NAME}')
        sums[name] = digest
    return sums


class SetRecord:
    """One set of the manifest with its checksums and its state in the destination."""

    def __init__(self, rec, sums_root, dest):
        self.rec = rec
        self.name = rec['set']
        parts = self.name.split('/')
        if not all(_plain_name(p) for p in parts):
            raise FetchError(f'manifest: set folder {self.name!r} is not a relative path')
        self.sums = read_sums(os.path.join(sums_root, *parts, SUMS_NAME), self.name)
        if len(self.sums) != int(rec['files']):
            raise FetchError(f'{self.name}: the manifest says {rec["files"]} files, '
                             f'{SUMS_NAME} lists {len(self.sums)}')
        self.folder = os.path.join(dest, *parts)
        if ('archive_sha256' in rec) == ('commit' in rec):
            raise FetchError(f'{self.name}: the manifest record needs exactly one of '
                             f'archive_sha256 and commit')
        self.verified, self.differing = set(), set()

    @property
    def missing(self):
        return [n for n in sorted(self.sums) if n not in self.verified
                and n not in self.differing]

    def present(self):
        return sum(os.path.lexists(os.path.join(self.folder, n)) for n in self.sums)


class Source:
    """One archive: the FJSPLIB archive or the zip of a pinned repository commit."""

    def __init__(self, rec):
        if 'archive_sha256' in rec:
            self.kind = 'archive'
            self.url = rec['source']
            self.sha256 = rec['archive_sha256'].lower()
            if not _HEX64.fullmatch(self.sha256):
                raise FetchError(f'{rec["set"]}: archive_sha256 is not a SHA-256')
            self.root = ''
            self.commit = None
        else:
            self.kind = 'repository'
            self.url = rec['archive_url']
            self.sha256 = None
            self.commit = rec['commit']
            self.root = rec['archive_root']
            if not _plain_name(self.root):
                raise FetchError(f'{rec["set"]}: archive_root is not a plain folder name')
        if self.kind == 'repository':
            self.file_name = self.root + '.zip'
        else:
            self.file_name = (os.path.basename(urllib.parse.urlparse(self.url).path)
                              or 'archive.zip')
        self.key = (self.kind, self.url, self.sha256, self.commit, self.root)
        self.sets = []


def load_manifest(path):
    try:
        with open(path, encoding='utf-8') as f:
            man = json.load(f)
    except FileNotFoundError:
        raise FetchError('the manifest sources.json was not found') from None
    except (OSError, ValueError) as e:
        raise FetchError(f'the manifest cannot be read ({e.__class__.__name__})') from None
    sets = man.get('sets')
    if not isinstance(sets, list) or not sets:
        raise FetchError('the manifest lists no sets')
    for k, r in enumerate(sets, 1):
        keys = (('set', 'path', 'files', 'source', 'archive_sha256')
                if isinstance(r, dict) and 'archive_sha256' in r else
                ('set', 'path', 'files', 'commit', 'archive_url', 'archive_root'))
        lacking = [key for key in keys if not isinstance(r, dict) or key not in r]
        if lacking:
            raise FetchError(f'the manifest record {k} lacks {", ".join(lacking)}')
    names = [r['set'] for r in sets]
    if len(set(names)) != len(names):
        raise FetchError('the manifest lists a set twice')
    return man


# ----------------------------------------------------------------------
# messages
# ----------------------------------------------------------------------

def _within(path, root):
    try:
        return os.path.commonpath([path, root]) == root
    except ValueError:
        return False


class Display:
    """Paths for messages: relative to the repository root, never absolute."""

    def __init__(self, dest):
        self.dest = dest

    def path(self, p):
        p = os.path.abspath(p)
        if _within(p, REPO_ROOT):
            return os.path.relpath(p, REPO_ROOT).replace(os.sep, '/')
        if _within(p, self.dest):
            rel = os.path.relpath(p, self.dest).replace(os.sep, '/')
            return '<dest>' if rel == '.' else '<dest>/' + rel
        return os.path.basename(p)


def say(msg):
    print(msg, flush=True)


# ----------------------------------------------------------------------
# local files
# ----------------------------------------------------------------------

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(CHUNK), b''):
            h.update(b)
    return h.hexdigest()


def scan(s, disp):
    """Classify the files of one set already in the destination."""
    for name in sorted(s.sums):
        p = os.path.join(s.folder, name)
        if not os.path.lexists(p):
            continue
        try:
            ok = os.path.isfile(p) and sha256_file(p) == s.sums[name]
        except OSError:
            ok = False
        if ok:
            s.verified.add(name)
        else:
            s.differing.add(name)
            say(f'differs: {disp.path(p)} does not match {SUMS_NAME}; it is left as it is')


def write_verified(s, name, data, disp, origin):
    """Write one file that does not exist yet, then read it back and check it."""
    p = os.path.join(s.folder, name)
    if hashlib.sha256(data).hexdigest() != s.sums[name]:
        say(f'not written: {origin} does not match {SUMS_NAME} of {s.name} for {name}')
        return
    os.makedirs(s.folder, exist_ok=True)
    try:
        f = open(p, 'xb')
    except FileExistsError:
        scan_one = os.path.isfile(p) and sha256_file(p) == s.sums[name]
        (s.verified if scan_one else s.differing).add(name)
        if not scan_one:
            say(f'differs: {disp.path(p)} does not match {SUMS_NAME}; it is left as it is')
        return
    try:
        with f:
            f.write(data)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(p)
        raise
    if sha256_file(p) == s.sums[name]:
        s.verified.add(name)
    else:
        s.differing.add(name)
        say(f'differs: {disp.path(p)} does not match {SUMS_NAME} after writing')


# ----------------------------------------------------------------------
# archives
# ----------------------------------------------------------------------

def _network_reason(e):
    if isinstance(e, urllib.error.HTTPError):
        return f'HTTP {e.code} {e.reason}'
    if isinstance(e, urllib.error.URLError):
        r = e.reason
        return getattr(r, 'strerror', None) or str(r) or r.__class__.__name__
    return getattr(e, 'strerror', None) or e.__class__.__name__


NETWORK_ERRORS = (OSError, http.client.HTTPException, ValueError)


def _network_error(src, e):
    hint = 'Download the file by other means and pass it with --archive <file>'
    if src.kind == 'repository' and any('generate' in s.rec for s in src.sets):
        hint += ', or write the SD2 sets with --sd2 generate'
    return FetchError(f'could not download {src.url} ({_network_reason(e)}). {hint}.')


def download(src, folder):
    """Download one archive into `folder`; returns (path, sha256)."""
    path = os.path.join(folder, src.file_name)
    req = urllib.request.Request(src.url, headers={'User-Agent': USER_AGENT})
    h, size = hashlib.sha256(), 0
    try:
        r = urllib.request.urlopen(req, timeout=TIMEOUT_S)
    except NETWORK_ERRORS as e:
        raise _network_error(src, e) from None
    with r, open(path, 'wb') as out:
        while True:
            try:
                b = r.read(CHUNK)
            except NETWORK_ERRORS as e:
                raise _network_error(src, e) from None
            if not b:
                break
            size += len(b)
            if size > MAX_DOWNLOAD:
                raise FetchError(f'{src.url} is larger than {MAX_DOWNLOAD} bytes; stopped')
            h.update(b)
            out.write(b)
    return path, h.hexdigest()


def _member_refusal(info):
    name = info.filename
    parts = name.replace('\\', '/').split('/')
    if name.startswith(('/', '\\')) or re.match(r'[A-Za-z]:', name):
        return 'an absolute path'
    if '..' in parts:
        return "a '..' component"
    if stat.S_ISLNK(info.external_attr >> 16):
        return 'a link'
    if info.is_dir():
        return 'a folder'
    if info.file_size > MAX_MEMBER:
        return f'a size above {MAX_MEMBER} bytes'
    return None


def zip_root(path):
    """The single top folder of a zip archive, or None."""
    try:
        with zipfile.ZipFile(path) as z:
            tops = {n.replace('\\', '/').split('/')[0] for n in z.namelist()}
    except (zipfile.BadZipFile, OSError):
        return None
    return tops.pop() if len(tops) == 1 else None


def install_from_zip(src, path, disp):
    """Write the missing files of every set of `src` from the archive at `path`."""
    origin = src.file_name
    try:
        z = zipfile.ZipFile(path)
    except (zipfile.BadZipFile, OSError):
        raise FetchError(f'{origin} is not a readable zip archive; nothing was extracted') from None
    with z:
        index = {}
        for info in z.infolist():
            index.setdefault(info.filename, []).append(info)
        plan = []
        for s in src.sets:
            prefix = '/'.join(p for p in (src.root, s.rec['path'].strip('/')) if p)
            if s.rec['path'].startswith('/'):
                prefix = '/' + prefix
            for name in s.missing:
                member = f'{prefix}/{name}'
                infos = index.get(member)
                if not infos:
                    say(f'not in {origin}: {member}')
                    continue
                if len(infos) > 1:
                    raise FetchError(f'{origin}: member {member} occurs {len(infos)} times; '
                                     f'nothing was extracted')
                why = _member_refusal(infos[0])
                if why:
                    raise FetchError(f'{origin}: member {member} is refused ({why}); '
                                     f'nothing was extracted')
                plan.append((s, name, infos[0]))
        data = []
        for s, name, info in plan:
            try:
                data.append(z.read(info))
            except (zipfile.BadZipFile, OSError, EOFError, zlib.error):
                raise FetchError(f'{origin}: member {info.filename} cannot be read; '
                                 f'nothing was extracted') from None
    for (s, name, info), b in zip(plan, data):
        write_verified(s, name, b, disp, f'{origin}:{info.filename}')


def match_local_archives(files, sources):
    """{source key: local path} for the files given with --archive."""
    by_key = {}
    for f in files:
        p = os.path.abspath(f)
        shown = os.path.basename(p)
        if not os.path.isfile(p):
            raise FetchError(f'--archive {shown}: no such file')
        digest = sha256_file(p)
        hit = [s for s in sources if s.kind == 'archive' and s.sha256 == digest]
        if not hit:
            named = [s for s in sources if s.kind == 'archive' and s.file_name == shown]
            if named:
                raise FetchError(f'--archive {shown}: its SHA-256 is {digest}, the manifest '
                                 f'pins {named[0].sha256}; the archive was not opened')
            root = zip_root(p)
            hit = [s for s in sources if s.kind == 'repository' and s.root == root]
        if not hit:
            raise FetchError(f'--archive {shown}: matches no source of the manifest '
                             f'(SHA-256 {digest}); nothing was extracted')
        for s in hit:
            by_key[s.key] = p
    return by_key


def fetch_source(src, local, tmp, disp):
    if local:
        path = local
        say(f'{", ".join(s.name for s in src.sets)}: reading {os.path.basename(local)}')
        if src.kind == 'archive' and sha256_file(path) != src.sha256:
            raise FetchError(f'{os.path.basename(local)} does not match the pinned SHA-256; '
                             f'the archive was not opened')
    else:
        say(f'{", ".join(s.name for s in src.sets)}: downloading {src.url}')
        path, digest = download(src, tmp)
        if src.kind == 'archive' and digest != src.sha256:
            raise FetchError(f'{src.url} has SHA-256 {digest}, the manifest pins '
                             f'{src.sha256}; the archive was not opened')
    if src.kind == 'repository':
        root = zip_root(path)
        if root != src.root:
            raise FetchError(f'{src.file_name}: the top folder is not {src.root}; '
                             f'nothing was extracted')
    install_from_zip(src, path, disp)


# ----------------------------------------------------------------------
# SD2 from the repository's generator
# ----------------------------------------------------------------------

def _import_data_utils():
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    argv = sys.argv
    sys.argv = argv[:1]          # params parses sys.argv when it is imported
    try:
        import data_utils
    except Exception as e:       # numpy, torch or the repository modules missing
        what = e.__class__.__name__ + (f' {e.name}' if getattr(e, 'name', None) else '')
        raise FetchError(f'--sd2 generate needs the repository environment '
                         f'(data_utils, numpy, torch): {what}') from None
    finally:
        sys.argv = argv
    return data_utils


def generate_sets(sets, tmp, disp):
    data_utils = _import_data_utils()
    for s in sets:
        g = s.rec['generate']
        say(f'{s.name}: generating with {g["writer"]}, seed {g["seed"]}')
        cfg = types.SimpleNamespace(
            n_j=int(g['n_j']), n_m=int(g['n_m']), data_source='SD2',
            data_size=int(g['data_size']), data_suffix=g['data_suffix'],
            low=int(g['low']), high=int(g['high']), op_per_job=g['op_per_job'],
            cover_data_flag=False)
        out = tempfile.mkdtemp(dir=tmp)
        with contextlib.redirect_stdout(io.StringIO()):
            data_utils.generate_data_to_files(int(g['seed']), out + os.sep, cfg)
        subs = [d for d in os.listdir(out) if os.path.isdir(os.path.join(out, d))]
        if len(subs) != 1:
            raise FetchError(f'{s.name}: the generator wrote {len(subs)} folders, expected 1')
        for name in s.missing:
            p = os.path.join(out, subs[0], name)
            if not os.path.isfile(p):
                say(f'not generated: {name} of {s.name}')
                continue
            with open(p, 'rb') as f:
                write_verified(s, name, f.read(), disp, f'generated {name}')


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def parse_args(argv, set_names):
    ap = argparse.ArgumentParser(
        description='Download the public instance files of data/PUBLIC and check them '
                    'against the SHA256SUMS of every set.')
    ap.add_argument('--sets', nargs='+', metavar='SET',
                    help='sets to fetch (default: every set of the manifest): '
                         + ', '.join(set_names))
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument('--verify-only', action='store_true',
                      help='no network: check the local files against SHA256SUMS')
    mode.add_argument('--archive', nargs='+', metavar='FILE', default=[],
                      help='local copies of the source archives, used instead of '
                           'downloading them')
    ap.add_argument('--dest', metavar='FOLDER',
                    help='destination folder of the set folders (default: data/PUBLIC)')
    ap.add_argument('--sd2', choices=('download', 'generate'), default='download',
                    help='download the SD2 sets (default) or write them with the '
                         "repository's generator, without network")
    ap.add_argument('--manifest', metavar='FILE', default=DEFAULT_MANIFEST,
                    help='the manifest; the SHA256SUMS files are read from its folder '
                         '(default: data/PUBLIC/sources.json)')
    a = ap.parse_args(argv)
    if a.sets:
        unknown = [s for s in a.sets if s not in set_names]
        if unknown:
            ap.error(f'unknown set(s) {", ".join(unknown)}; the manifest has '
                     f'{", ".join(set_names)}')
    return a


def _manifest_path(argv):
    """The --manifest value, read before the full parse so --sets can be checked."""
    argv = list(sys.argv[1:] if argv is None else argv)
    for k, a in enumerate(argv):
        if a == '--manifest' and k + 1 < len(argv):
            return argv[k + 1]
        if a.startswith('--manifest='):
            return a.split('=', 1)[1]
    return DEFAULT_MANIFEST


def fetch_missing(man, sets, a, disp, errors):
    """Obtain the missing files of `sets` from their sources or from the generator."""
    all_sources = {}
    for r in man['sets']:
        src = Source(r)
        all_sources.setdefault(src.key, src)
    local = match_local_archives(a.archive, list(all_sources.values()))
    need = [s for s in sets if s.missing]
    gen = [s for s in need if a.sd2 == 'generate' and 'generate' in s.rec]
    sources = {}
    for s in need:
        if s not in gen:
            src = Source(s.rec)
            sources.setdefault(src.key, src).sets.append(s)
    for s in sets:
        if not s.missing:
            say(f'{s.name}: no file missing')
    if not (sources or gen):
        return
    with tempfile.TemporaryDirectory(prefix='fetch-public-') as tmp:
        for key, src in sources.items():
            try:
                fetch_source(src, local.get(key), tmp, disp)
            except FetchError as e:
                errors.append(str(e))
                say(f'error: {e}')
        if gen:
            generate_sets(gen, tmp, disp)


def run(argv=None):
    man_path = os.path.abspath(_manifest_path(argv))
    man = load_manifest(man_path)
    names = [r['set'] for r in man['sets']]
    a = parse_args(argv, names)
    sums_root = os.path.dirname(man_path)
    dest = os.path.abspath(a.dest) if a.dest else sums_root
    disp = Display(dest)
    chosen = a.sets or names
    records = [r for r in man['sets'] if r['set'] in chosen]
    sets = [SetRecord(r, sums_root, dest) for r in records]

    for s in sets:
        scan(s, disp)

    errors = []
    if not a.verify_only:
        try:
            fetch_missing(man, sets, a, disp, errors)
        except FetchError as e:
            errors.append(str(e))
            say(f'error: {e}')

    bad = bool(errors)
    for s in sets:
        n_ver, n_diff = len(s.verified), len(s.differing)
        n_miss = len(s.sums) - n_ver - n_diff
        bad |= bool(n_miss or n_diff)
        say(f'{s.name}: expected {len(s.sums)}, present {s.present()}, verified {n_ver}, '
            f'missing {n_miss}, differing {n_diff}')
    return 1 if bad else 0


def main(argv=None):
    try:
        return run(argv)
    except FetchError as e:
        print(f'error: {e}', file=sys.stderr, flush=True)
        return 1
    except KeyboardInterrupt:
        print('interrupted', file=sys.stderr, flush=True)
        return 130


if __name__ == '__main__':
    sys.exit(main())
