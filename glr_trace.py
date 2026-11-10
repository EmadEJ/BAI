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
from optimization import plain_GLR, reference_GLR
from lowerbound import L_bound, L_naive, proven_factor

# Figure 2: the statistics along one run of GTS-Unknown (LB) on an instance
#   python glr_trace.py --instance_index 1 --log_period 20 --jobs 8
# The sampling rule of GTS-Unknown (LB) keeps running after its stopping time, until L and L_naive have also crossed
# the naive threshold (plus a margin), so that every stopping time is on the plot. The run is logged every log_period
# rounds, then at every logged round we compute the plain solver GLR, the reference (true) GLR, L, L_naive, the proven
# upper zone of L and both thresholds. The plain solver of the plot is weaker than the one of GTS-Unknown (solver): it
# starts from B = A and --plain_starts random points (3 by default), while the algorithm also starts from the best B
# for lambda = mu. With k = 2 the weaker solver can stay above the true GLR.
# Writes results/glr_trace/instance1.json

OUT_DIR = "results/glr_trace/"


def statistics(job):
    warnings.filterwarnings("ignore")
    np.random.seed(0)
    n, k, confidence, plain_starts, reference_starts, state = job
    alg = STS(n, k, confidence, "G")
    alg.load_state(state)
    mu_hat, A_hat, N_A, N_Z = alg.get_mu_hat(), alg.get_A_hat(), alg.N_A, alg.N_Z

    L = L_bound(mu_hat, A_hat, N_A, N_Z)
    factor = proven_factor(mu_hat, A_hat, N_A, N_Z)
    return {
        'T': state['T'],
        'plain': plain_GLR(mu_hat, A_hat, N_A, N_Z, n_random=plain_starts, context_start=False)[0],
        'reference': reference_GLR(mu_hat, A_hat, N_A, N_Z, n_random=reference_starts)[0],
        'L': np.min(L),
        'L_naive': np.min(L_naive(mu_hat, A_hat, N_A, N_Z)),
        'L_upper': np.min(factor * L),  # Lambda <= min_i factor_i L_i
        'beta': alg.beta_t(confidence),
        'naive_beta': alg.naive_beta_t(confidence),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_index", type=int, default=0)
    parser.add_argument("--log_period", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--delta", type=float, default=None)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--margin", type=float, default=0.15, help="fraction of rounds run after the last crossing")
    parser.add_argument("--max_T", type=int, default=10**6)
    parser.add_argument("--plain_starts", type=int, default=3, help="random starting points of the plotted plain solver")
    parser.add_argument("--reference_starts", type=int, default=20, help="random starting points of the reference GLR")
    args = parser.parse_args()

    n, k, confidence, mus, A = read_instance(args.instance_index)
    if args.delta is not None:
        confidence = args.delta

    st = time.time()
    env = Environment(mus, A, n, k, seed=args.seed)
    alg = env.make_algorithm("STS-LB", confidence, "G", {'fast': True})
    states, T_end = [], args.max_T
    while env.T < T_end:
        action, init = alg.get_action()
        post_action, reward = env.take_action(action)
        alg.update(action, post_action, reward)
        if init or env.T % args.log_period != 0:
            continue
        states.append(alg.state())
        mu_hat, A_hat = alg.get_mu_hat(), alg.get_A_hat()
        lower = min(np.min(L_bound(mu_hat, A_hat, alg.N_A, alg.N_Z)), np.min(L_naive(mu_hat, A_hat, alg.N_A, alg.N_Z)))
        if T_end == args.max_T and lower > alg.naive_beta_t(confidence):
            T_end = int(env.T * (1 + args.margin))
    print(f"run until T={env.T} in {time.time() - st:.1f}s, logged {len(states)} rounds")

    with Pool(args.jobs) as pool:
        stats = pool.map(statistics, [(n, k, confidence, args.plain_starts, args.reference_starts, state) for state in states], chunksize=4)
    print(f"statistics computed in {time.time() - st:.0f}s")

    trace = {key: [s[key] for s in stats] for key in stats[0]}
    result = {
        'instance': args.instance_index,
        'n': n, 'k': k,
        'confidence': confidence,
        'seed': args.seed,
        'log_period': args.log_period,
        'plain_starts': args.plain_starts,
        'reference_starts': args.reference_starts,
        'trace': trace,
    }
    save_json(OUT_DIR + f"instance{args.instance_index}.json", result)

    for key in ['plain', 'reference', 'L', 'L_naive']:
        for beta in ['beta', 'naive_beta']:
            crossed = [t for t, value, b in zip(trace['T'], trace[key], trace[beta]) if value > b]
            print(f"{key} crosses {beta} at T={crossed[0] if crossed else None}")
