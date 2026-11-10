import os
os.environ.setdefault("OMP_NUM_THREADS", "1")  # the runs are parallelized over processes instead
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import time
import warnings
import numpy as np
from multiprocessing import Pool

from io_utils import *
from environment import *

# Usage:
#   python simulate.py                     (one run of GTS-Unknown (LB) on the sample instance 0)
#   python simulate.py --algorithm STS-LB --instance_index 4 --runs 100 --jobs 8
#   python simulate.py --algorithm STS-LB ASTS --instance_index 4 5 --runs 100 --jobs 8
# The result file results/simulation/instance4_STS-LB.json keeps one entry per run (seed = run index),
# so running the same command again only runs the missing seeds. All the runs of a command share one pool.

SLOW_FIRST = ["STS-Plain", "MuSTS", "STS-LB", "STS-LB-Naive", "SGTS", "ASTS"]  # the order in which the runs are started


def single_run(job):
    index, algorithm, confidence, tracking, mode, seed, verbose = job
    warnings.filterwarnings("ignore")
    n, k, _, mus, A = read_instance(index)
    env = Environment(mus, A, n, k, seed=seed)
    return index, algorithm, env.run(confidence, algorithm, tracking, mode, verbose=verbose)


def print_summary(data):
    runs = data['runs']
    Ts = np.array([run['T'] for run in runs])
    times = np.array([run['time'] for run in runs])
    print(f"{data['algorithm']} on instance {data['instance']}: {len(runs)} runs, mean T {Ts.mean():.0f}, "
          f"median T {np.median(Ts):.0f}, error rate {1 - np.mean([run['correct'] for run in runs]):.3f} "
          f"(delta = {data['confidence']}), mean time per run {times.mean():.1f}s")


def simulate():
    args = get_arguments()
    mode = {'fast': not args.no_fast}

    files, jobs, size = {}, [], {}
    for index in args.instance_index:
        n, k, confidence, _, _ = read_instance(index)
        if args.delta is not None:
            confidence = args.delta
        size[index] = n * k
        for algorithm in args.algorithm:
            path = simulation_path(index, algorithm, args.out_dir)
            data = load_json(path, default={
                'instance': index,
                'algorithm': algorithm,
                'confidence': confidence,
                'mode': mode,
                'runs': [],
            })
            assert data['confidence'] == confidence, f"{path} was run with another confidence"
            files[(index, algorithm)] = (path, data)
            done = {run['seed'] for run in data['runs']}
            seeds = [seed for seed in range(args.runs) if seed not in done]
            print(f"{algorithm} on instance {index} (n={n}, k={k}, delta={confidence}): {len(done)} runs stored, {len(seeds)} to go")
            jobs += [(index, algorithm, confidence, args.tracking, mode, seed, args.verbose) for seed in seeds]
    jobs.sort(key=lambda job: (SLOW_FIRST.index(job[1]) if job[1] in SLOW_FIRST else len(SLOW_FIRST), -size[job[0]]))

    st_time = time.time()
    with Pool(args.jobs) as pool:
        for index, algorithm, result in pool.imap_unordered(single_run, jobs):
            path, data = files[(index, algorithm)]
            data['runs'].append(result)
            data['runs'].sort(key=lambda run: run['seed'])
            save_json(path, data)
            print(f"{algorithm} on instance {index}, seed {result['seed']}: T={result['T']}, correct={result['correct']}, "
                  f"{result['time']:.1f}s (total {time.time() - st_time:.0f}s)", flush=True)

    print("#"*60)
    for path, data in files.values():
        if data['runs']:
            print_summary(data)
    print("#"*60)


if __name__ == "__main__":
    simulate()
