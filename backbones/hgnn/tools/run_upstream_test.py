"""
Run the upstream test.py, unmodified, on the CPU.

Usage:
    python tools/run_upstream_test.py <dir holding the upstream code>

The upstream script writes ./save/test_<timestamp>/ inside that directory, so
point it at a scratch copy of the upstream repository. The GPU is hidden
before torch is imported, and compat.install() supplies the missing gym and
pynvml modules. After the run, the per-instance greedy makespans recorded
from the upstream schedule() calls are summarised on stdout.
"""
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import compat  # noqa: E402

compat.install()


def main():
    upstream = os.path.abspath(sys.argv[1])
    os.chdir(upstream)
    sys.path.insert(0, upstream)
    import torch
    torch.set_num_threads(1)
    import test as upstream_test

    record = []
    original_schedule = upstream_test.schedule

    def recording_schedule(env, model, memories, flag_sample=False):
        makespan, spend = original_schedule(env, model, memories, flag_sample=flag_sample)
        record.append(float(makespan.min()))
        return makespan, spend

    upstream_test.schedule = recording_schedule
    upstream_test.main()
    # test.py averages num_average identical greedy rollouts per instance
    import json
    with open('./config.json') as f:
        test_paras = json.load(f)['test_paras']
    k = 1 if test_paras['sample'] else test_paras['num_average']
    per_ins = [sum(record[i:i + k]) / k for i in range(0, len(record), k)]
    print('instances: {0}  mean makespan: {1:.4f}'.format(len(per_ins), sum(per_ins) / len(per_ins)))


if __name__ == '__main__':
    main()
