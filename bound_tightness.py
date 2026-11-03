import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import time
import warnings
import numpy as np
from multiprocessing import Pool

from io_utils import *
from optimization import reference_GLR_pair, optimize
from lowerbound import L_bound, L_naive, proven_factor, variance, optimal_w

# How far are the convex lower bounds from the true GLR Lambda_i (and from the proven factor)?
#   python bound_tightness.py --jobs 8
# Writes results/bound_tightness/pairs.json (one entry per pair (instance, i)) and
# results/bound_tightness/characteristic_times.json (T*, U and U_naive of the paper instances).

OUT_DIR = "results/bound_tightness/"
PAPER_INSTANCES = [6, 7, 10, 11, 12, 21, 22, 32, 33, 34, 62, 63, 64, 92, 93, 94, 122, 123]
HARD_EPS = [0.3, 0.1, 0.03, 0.01, 0.003, 0.001]
PROP5_EPS = [1e-2, 1e-4, 1e-6, 1e-9, 1e-12]


def make_cases(n_random, n_families, seed=0):
    # every case is an instance with a fixed allocation w and q = A^T w (the bounds and Lambda scale with w)
    rng = np.random.default_rng(seed)
    cases = []

    for index in PAPER_INSTANCES:
        n, k, _, mu, A = read_instance(index)
        cases.append({'family': 'paper', 'instance': index, 'mu': mu, 'A': A, 'w': np.ones(n) / n})

    for r in range(n_random):
        n, k = rng.integers(2, 6), rng.integers(2, 6)
        alpha = rng.choice([0.3, 1.0, 3.0])
        A = np.maximum(rng.dirichlet(alpha * np.ones(k), size=n), 1e-6)
        A = A / A.sum(axis=1, keepdims=True)
        cases.append({'family': 'random', 'id': r, 'alpha': float(alpha), 'mu': rng.uniform(0, 1, k), 'A': A,
                      'w': rng.dirichlet(np.ones(n))})

    for f in range(n_families):
        # hard instances: A_i = (1 - eps) A_star + eps A0_i, harder as eps -> 0
        k = [3, 4, 5][f % 3]
        mu = rng.uniform(0, 1, k)
        A0 = rng.dirichlet(np.ones(k), size=3)
        A0 = A0[np.argsort(-(A0 @ mu))]  # arm 0 is the best one
        for eps in HARD_EPS:
            A = np.vstack([A0[0], (1 - eps) * A0[0] + eps * A0[1], (1 - eps) * A0[0] + eps * A0[2]])
            cases.append({'family': 'hard', 'id': f, 'eps': eps, 'mu': mu, 'A': A, 'w': np.ones(3) / 3})

    for eps in PROP5_EPS:
        # the family of Proposition 5, no closed form of this type can be within a constant factor
        A = np.array([[1 - eps, eps], [0.5, 0.5]])
        cases.append({'family': 'prop5', 'eps': eps, 'mu': np.array([1.0, 0.0]), 'A': A, 'w': np.array([1.0, eps**2])})
    return cases


def evaluate_case(case):
    warnings.filterwarnings("ignore")
    np.random.seed(0)
    mu, A, w = case['mu'], case['A'], case['w']
    q = A.T @ w
    i_star = np.argmax(A @ mu)

    L, L_n = L_bound(mu, A, w, q), L_naive(mu, A, w, q)
    factor, factor_w = proven_factor(mu, A), proven_factor(mu, A, w, q)
    pairs = []
    for i in range(A.shape[0]):
        if i == i_star:
            continue
        Lambda_i = reference_GLR_pair(mu, A, w, q, i)[0]
        delta = (A[i_star] - A[i]) @ mu
        pairs.append({
            'i': i,
            'Lambda': Lambda_i,
            'L': L[i],
            'L_naive': L_n[i],
            'factor': factor[i],  # allocation-free, Theorems 2-3
            'factor_w': factor_w[i],  # Theorem 3 at this allocation
            'delta': delta,
            'gamma_bar': delta / min(variance(A[i], mu), variance(A[i_star], mu)),
            'p_min': float(np.min(np.minimum(A[i], A[i_star]))),
        })
    info = {key: (value.tolist() if isinstance(value, np.ndarray) else value) for key, value in case.items()}
    info.update({'n': A.shape[0], 'k': A.shape[1], 'pairs': pairs})
    return info


def characteristic_times(index):
    # T*^-1 = sup_w F(w) (plain solver, best of a few starts), U^-1 and U_naive^-1 (convex programs)
    warnings.filterwarnings("ignore")
    np.random.seed(0)
    n, k, _, mu, A = read_instance(index)
    st = time.time()
    U_inv, w_U = optimal_w(mu, A, bound="L")
    U_naive_inv, w_U_naive = optimal_w(mu, A, bound="naive")
    best, w_best = -np.inf, None
    for w0 in [np.ones(n) / n, w_U, None, None]:
        value, w = optimize(mu, A, w0=w0, inner_alg="reference")
        if value > best:
            best, w_best = value, w
    return {'instance': index, 'n': n, 'k': k, 'T_star': 1 / best, 'U': 1 / U_inv, 'U_naive': 1 / U_naive_inv,
            'w_star': w_best.tolist(), 'w_U': w_U.tolist(), 'w_U_naive': w_U_naive.tolist(), 'time': time.time() - st}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--n_random", type=int, default=300)
    parser.add_argument("--n_families", type=int, default=12)
    parser.add_argument("--skip_times", action="store_true", help="do not compute T* and U of the paper instances")
    args = parser.parse_args()

    cases = make_cases(args.n_random, args.n_families)
    st = time.time()
    with Pool(args.jobs) as pool:
        results = pool.map(evaluate_case, cases, chunksize=1)
    save_json(OUT_DIR + "pairs.json", results)
    n_pairs = sum(len(r['pairs']) for r in results)
    print(f"{len(results)} cases, {n_pairs} pairs in {time.time() - st:.0f}s")

    bad = [(r['family'], p['i']) for r in results for p in r['pairs'] if not (p['L'] <= p['Lambda'] * (1 + 1e-6))]
    print("pairs with L > Lambda:", bad)
    bad = [(r['family'], p['i']) for r in results for p in r['pairs'] if not (p['Lambda'] <= p['factor'] * p['L'] * (1 + 1e-6))]
    print("pairs with Lambda > factor * L:", bad)

    if not args.skip_times:
        st = time.time()
        with Pool(args.jobs) as pool:
            times = pool.map(characteristic_times, PAPER_INSTANCES, chunksize=1)
        save_json(OUT_DIR + "characteristic_times.json", times)
        print(f"characteristic times in {time.time() - st:.0f}s")
