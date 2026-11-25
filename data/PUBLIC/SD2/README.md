# SD2 instances of the dual-attention network release (10x5 and 20x10)

## Source

- github.com/wrqccc/FJSP-DRL, branch `main`, fetched on 2026-11-11, equal to commit
  `2cf81b13f5044451e78cf780f8fb3e7eeac054c1` of 2023-09-12, the head of `main`:
  - `test/10x5+mix`, `test/20x10+mix`: files `data/SD2/10x5+mix/*.fjs` and
    `data/SD2/20x10+mix/*.fjs` (100 instances each), the published test sets;
  - `vali/10x5+mix`, `vali/20x10+mix`: files `data/data_train_vali/SD2/10x5+mix/*.fjs` and
    `data/data_train_vali/SD2/20x10+mix/*.fjs` (100 each), the validation sets of the
    release.
- The files are those of the repository, byte for byte, and `SHA256SUMS` in each folder
  lists their checksums.
- The generator of this repository regenerates all four sets exactly:
  `data_utils.SD2_instance_generator` with `data_suffix mix`, `op_per_job 0`, `low 1`,
  `high 99`, after `np.random.seed(s)` and `random.seed(s)` with s = 200 (test sets) and
  s = 100 (validation sets), instances drawn in file order (checked on 2026-11-11, 400 of
  400 identical). `data_utils.generate_data_to_files` with these settings writes all 400
  files byte for byte as listed in `SHA256SUMS`, and
  `scripts/fetch_public_instances.py --sd2 generate` uses this writer.

## Citation

R. Wang, G. Wang, J. Sun, F. Deng and J. Chen. Flexible job shop scheduling via dual
attention network-based reinforcement learning. IEEE Transactions on Neural Networks and
Learning Systems, 2023. doi:10.1109/TNNLS.2023.3306421.

## Licence

The repository states no licence. The files are not redistributed here.
`scripts/fetch_public_instances.py` downloads them from github.com/wrqccc/FJSP-DRL at
commit `2cf81b13f5044451e78cf780f8fb3e7eeac054c1` and checks every file against
`SHA256SUMS` (see "Obtaining the instance files" in `data/PUBLIC/README.md`).

## Distribution

J jobs and M machines (10x5 or 20x10); every job has M operations (N = 50 or 200); every
operation draws its number of eligible machines uniformly from 1..M and the machines
uniformly without replacement; every eligible processing time is drawn independently and
uniformly from 1..99. `constraint_family/public_adapter.sd2_instance` draws from the same
distribution with a numpy Generator, for training on the fly.

## Reference makespans: `best_known_test_10x5+mix.csv`, `best_known_test_20x10+mix.csv`

Per-instance OR-Tools CP-SAT results of the release (`or_solution/SD2/<set>/solution_<set>_<k>.npy`,
pairs of makespan and solver seconds with an 1800 s limit). The paper reports set
averages only, so `verified` reads "repository file"; the means of the files are 326.24
(10x5) and 464.16 (20x10). A solve that stopped before the limit is optimal: 96 of 100 on
10x5 and 1 of 100 on 20x10. On 20x10 the values are feasible solutions, not best-known
values in the literature sense.

## Public family

Descriptors for the eight regimes are under `data/PUBLIC/family/pub_SD2_test_10x5+mix`,
`pub_SD2_test_20x10+mix`, `pub_SD2_vali_10x5+mix` and `pub_SD2_vali_20x10+mix`.
