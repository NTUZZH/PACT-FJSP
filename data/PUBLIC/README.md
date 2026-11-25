# Public FJSP instances and the public family

Public flexible job-shop instances in their original text format, their best-known
makespans with sources, and descriptors of the eight timing regimes (N, L, S, W, LS, LW,
SW, LSW) drawn on them by `constraint_family/public_rules.py`.

## Sets

### Obtaining the instance files

The instance files (`*.fjs`) are not part of the repository.
`scripts/fetch_public_instances.py` downloads them from their sources and verifies every
file against the `SHA256SUMS` of its folder. Run it once from the repository root:

    python scripts/fetch_public_instances.py                 # every set of sources.json
    python scripts/fetch_public_instances.py --sets Brandimarte SD2/test/10x5+mix
    python scripts/fetch_public_instances.py --verify-only   # no network: local files against SHA256SUMS
    python scripts/fetch_public_instances.py --archive <local TextData.zip>   # no download of the archive
    python scripts/fetch_public_instances.py --dest <folder> # default data/PUBLIC
    python scripts/fetch_public_instances.py --sd2 generate  # SD2 from the repository's generator, no network

`sources.json` lists, for each set, the source, its pin, the folder of the files inside
the source and the number of files.

- `Brandimarte`, `Hurink_edata`, `Hurink_rdata` and `Hurink_vdata` come from the FJSPLIB
  archive https://people.idsia.ch/~monaldo/fjspresults/TextData.zip, linked from
  https://people.idsia.ch/~monaldo/fjsp.html and pinned by its SHA-256
  `e6637b4f84e2ab188a0c90ce27062e8ee50f11f34334d12c07931227e474f62f`.
- The four `SD2` sets come from github.com/wrqccc/FJSP-DRL at commit
  `2cf81b13f5044451e78cf780f8fb3e7eeac054c1` and are read from the zip archive of that
  commit. `--archive` also accepts a local copy of that zip
  (github.com/wrqccc/FJSP-DRL/archive/2cf81b13f5044451e78cf780f8fb3e7eeac054c1.zip).
  `--sd2 generate` writes the same 400 files with `data_utils.generate_data_to_files`,
  with seed 200 for the test sets and seed 100 for the validation sets, and needs numpy
  and torch.

The script checks `TextData.zip` against its SHA-256 before it opens the archive. It
extracts only the files listed in `SHA256SUMS`, keeps their bytes unchanged and checks
every written file against its checksum. A file that is already present and matches its
checksum is left alone, and a file that differs is reported and never overwritten. The run
ends with one line per set (files expected, present, verified, missing, differing) and
exits with a non-zero code if any file is missing or differs.

### Folders

| folder | instances | sizes (J x M) | source | best-known table |
|---|---|---|---|---|
| `Brandimarte` | Mk01-Mk10 (10) | 10x6 to 20x15 | Brandimarte (1993), FJSPLIB archive | `best_known.csv` |
| `Hurink_edata` | mt06/10/20, la01-la40, abz5-9, car1-8, orb1-10 (66) | 6x6 to 30x10 | Hurink, Jurisch and Thole (1994), FJSPLIB archive | `best_known.csv` |
| `Hurink_rdata` | same 66 | same | same | `best_known.csv` |
| `Hurink_vdata` | same 66 | same | same | `best_known.csv` |
| `SD2/test/10x5+mix`, `SD2/test/20x10+mix` | 100 each | 10x5, 20x10 | Wang et al. (2023) release | `SD2/best_known_test_*.csv` (OR-Tools references) |
| `SD2/vali/10x5+mix`, `SD2/vali/20x10+mix` | 100 each | 10x5, 20x10 | Wang et al. (2023) release | none |

Each folder has a `README.md` with the source URL, the citation, the licence status, the
sizes and the notes on that set, and a `SHA256SUMS` of the instance files.

Not collected, because no public set matches the project's rules:

- minimum time lags in a flexible job shop: the job-shop time-lag benchmarks of Caumond,
  Lacomme and Tchernev (Computers and Operations Research 35:2331-2356, 2008) are built on
  classical job-shop instances (one machine per operation) and are used with maximum time
  lags; the project's LAG rule is a minimum lag only;
- machine unavailability: the flexible job-shop sets found with maintenance (for example
  Perroux, Arbaoui and Merghem-Boulahia, IFAC-PapersOnLine 56(2), 2023, not accessed)
  schedule maintenance tasks as decisions; the project's WINDOW rule fixes the outages in
  advance. No file set with fixed outage calendars was found.

## Best-known values

Every value in a `best_known.csv` names its publication and table in `source`.
`verified = yes` means the number was read from that table; the original papers credited
inside a compilation were not checked one by one. Values listed by the FJSPLib web page
(scheduleopt.github.io/benchmarks/fjsplib, accessed 2026-11-11) are given in separate
`fjsplib_*` columns and are not used as best-known values, because most of them are
credited to solver runs or to web pages, not to publications. A best-known value refers to
the instance as published, which is regime N of the public family.

## Instances the environment cannot hold

- `Hurink_*/orb7.fjs`: an operation with processing time 0 (the environment uses 0 for
  "machine not eligible"); left out of the public family.
- `Brandimarte` Mk06, Mk08, Mk10 declare machines that no operation can use (15/10, 10/9,
  15/11 declared/used); the environment runs with them (tested), and
  `PublicInstance.without_unused_machines()` removes them when wanted.

## Public family: `family/`

| folder | base instances | regimes | descriptor seeds |
|---|---|---|---|
| `family/pub_Brandimarte` | 10 | 8 | 20261101000 + index |
| `family/pub_Hurink_edata` | 65 | 8 | 20261102000 + index |
| `family/pub_Hurink_rdata` | 65 | 8 | 20261103000 + index |
| `family/pub_Hurink_vdata` | 65 | 8 | 20261104000 + index |
| `family/pub_SD2_test_10x5+mix` | 100 | 8 | 20261105000 + index |
| `family/pub_SD2_test_10x5+mix_lag050` | 100 | 8 | 20261105000 + index, lag durations halved (`lag_scale` 0.5) |
| `family/pub_SD2_test_20x10+mix` | 100 | 8 | 20261106000 + index |
| `family/pub_SD2_vali_10x5+mix` | 100 | 8 | 20261107000 + index |
| `family/pub_SD2_vali_20x10+mix` | 100 | 8 | 20261108000 + index |

The index is the position of the file in natural order (orb7 keeps its index and is
skipped). Layout of each folder, as for `data/FAMILY/<size>`:

- `desc/<REGIME>/<name>.desc.json`: `ConstraintDescriptor.to_json()` plus the keys
  `base_instance` (path stem of the `.fjs` file relative to the repository root),
  `base_format` (`fjs`: no `.meta.json` side-car), `regime`, `desc_seed` and
  `rules_version`;
- `rules/<name>.rules.json`: the member parameters shared by all regimes (true lags, job
  classes, setup matrix, windows, horizon, the generator parameters);
- `dataset_meta.json`: base folder, instance list, excluded files, regimes, seed, rule
  parameters.

All times are integers in the time unit of the base instance (not hours). Rebuild a folder
with, for example:

    python -m constraint_family.public_rules --set data/PUBLIC/Brandimarte \
        --out data/PUBLIC/family/pub_Brandimarte --name Brandimarte

The rules, their factory template and the seeds are documented in
`constraint_family/public_rules.py`; `tests/test_public_adapter.py` checks the stored
records against fresh draws.
