import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import time
import warnings
import numpy as np
from multiprocessing import Pool

from io_utils import *
from environment import Environment
from algorithms.STS import STS
from optimization import optimize_GLR
from lowerbound import L_bound, L_naive, proven_factor

# Figure 2: the statistics along one run of GTS-Unknown (LB) on an instance
#   python glr_trace.py --instance_index 7 --log_period 10 --jobs 8
# The run is logged every log_period rounds, then at every logged round we compute the plain solver GLR,
# the reference (true) GLR, L, L_naive, the proven upper zone of L and both thresholds.
# Writes results/glr_trace/instance7.json

OUT_DIR = "results/glr_trace/"


def statistics(job):
    warnings.filterwarnings("ignore")
    np.random.seed(0)
    n, k, confidence, state = job
    alg = STS(n, k, confidence, "G")
    alg.load_state(state)
    mu_hat, A_hat, N_A, N_Z = alg.get_mu_hat(), alg.get_A_hat(), alg.N_A, alg.N_Z

    L = L_bound(mu_hat, A_hat, N_A, N_Z)
    factor = proven_factor(mu_hat, A_hat, N_A, N_Z)
    return {
        'T': state['T'],
        'plain': optimize_GLR(mu_hat, A_hat, N_A, N_Z, alg="plain")[0],
        'reference': optimize_GLR(mu_hat, A_hat, N_A, N_Z, alg="reference")[0],
        'L': np.min(L),
        'L_naive': np.min(L_naive(mu_hat, A_hat, N_A, N_Z)),
        'L_upper': np.min(factor * L),  # Lambda <= min_i factor_i L_i
        'beta': alg.beta_t(confidence),
        'naive_beta': alg.naive_beta_t(confidence),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_index", type=int, required=True)
    parser.add_argument("--log_period", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--delta", type=float, default=None)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()

    n, k, confidence, mus, A = read_instance(args.instance_index)
    if args.delta is not None:
        confidence = args.delta

    st = time.time()
    env = Environment(mus, A, n, k, seed=args.seed)
    run = env.run(confidence, "STS-LB", log_period=args.log_period)
    print(f"run stopped at T={run['T']} in {run['time']:.1f}s, logged {len(run['states'])} rounds")

    with Pool(args.jobs) as pool:
        stats = pool.map(statistics, [(n, k, confidence, state) for state in run['states']], chunksize=4)
    print(f"statistics computed in {time.time() - st:.0f}s")

    trace = {key: [s[key] for s in stats] for key in stats[0]}
    result = {
        'instance': args.instance_index,
        'n': n, 'k': k,
        'confidence': confidence,
        'seed': args.seed,
        'log_period': args.log_period,
        'T_stop': run['T'],
        'correct': run['correct'],
        'trace': trace,
    }
    save_json(OUT_DIR + f"instance{args.instance_index}.json", result)

    for key in ['plain', 'reference', 'L', 'L_naive']:
        crossed = [t for t, value, beta in zip(trace['T'], trace[key], trace['beta']) if value > beta]
        print(f"{key} crosses our beta at T={crossed[0] if crossed else None}")
