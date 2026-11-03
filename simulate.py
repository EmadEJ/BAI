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
#   python simulate.py --algorithm STS-LB --instance_index 11 --runs 100 --jobs 8
# The result file results/simulation/instance11_STS-LB.json keeps one entry per run (seed = run index),
# so running the same command again only runs the missing seeds.


def single_run(job):
    index, algorithm, confidence, tracking, mode, seed, verbose = job
    warnings.filterwarnings("ignore")
    n, k, _, mus, A = read_instance(index)
    env = Environment(mus, A, n, k, seed=seed)
    return env.run(confidence, algorithm, tracking, mode, verbose=verbose)


def simulate():
    args = get_arguments()
    index = args.instance_index
    n, k, confidence, mus, A = read_instance(index)
    if args.delta is not None:
        confidence = args.delta
    mode = {'fast': not args.no_fast}

    path = simulation_path(index, args.algorithm, args.out_dir)
    data = load_json(path, default={
        'instance': index,
        'algorithm': args.algorithm,
        'confidence': confidence,
        'mode': mode,
        'runs': [],
    })
    assert data['confidence'] == confidence, f"{path} was run with another confidence"
    done = {run['seed'] for run in data['runs']}
    seeds = [seed for seed in range(args.runs) if seed not in done]
    print(f"{args.algorithm} on instance {index} (n={n}, k={k}, delta={confidence}): {len(done)} runs stored, {len(seeds)} to go")

    jobs = [(index, args.algorithm, confidence, args.tracking, mode, seed, args.verbose) for seed in seeds]
    st_time = time.time()
    with Pool(args.jobs) as pool:
        for result in pool.imap_unordered(single_run, jobs):
            data['runs'].append(result)
            data['runs'].sort(key=lambda run: run['seed'])
            save_json(path, data)
            print(f"seed {result['seed']}: T={result['T']}, correct={result['correct']}, {result['time']:.1f}s")

    runs = data['runs']
    if not runs:
        return
    Ts = np.array([run['T'] for run in runs])
    times = np.array([run['time'] for run in runs])
    print("#"*60)
    print(f"runs: {len(runs)}, mean T: {Ts.mean():.0f}, median T: {np.median(Ts):.0f}")
    print(f"error rate: {1 - np.mean([run['correct'] for run in runs]):.3f} (delta = {confidence})")
    print(f"mean time per run: {times.mean():.1f}s, per round: {np.sum(times) / np.sum(Ts) * 1e3:.2f}ms")
    print(f"wall time now: {time.time() - st_time:.1f}s")
    print("#"*60)


if __name__ == "__main__":
    simulate()
