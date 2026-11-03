from tqdm import tqdm
import itertools
import numpy as np
import cvxpy as cp
from scipy.optimize import minimize
from scipy.special import softmax, log_softmax

from utils import *
from io_utils import *


TESTSET_DIR = "testsets/"

TOL = 1e-4  # used for checking whether the optimization was good enough

SIGMA = 1.0  # unit-variance Gaussian rewards


############################## helper functions

def best_arm(mu, A):
    means = np.dot(A, mu)
    best_arm = np.argmax(means)
    return best_arm, means[best_arm]


def objective(mu, A, mu_p, A_p, N_A, N_Z):
    n, k = A.shape

    result = 0
    for i in range(n):
        result += N_A[i] * categorical_kl(A[i], A_p[i])
    for j in range(k):
        result += N_Z[j] * gaussian_kl(mu[j], mu_p[j])

    return result


def optimal_mu(mu, N_Z, d):
    # min_{lambda in [0,1]^k} sum_j N_Z[j] (mu_j - lambda_j)^2 / 2 s.t. d^T lambda >= 0, returns (cost, lambda, nu)
    # KKT: lambda(nu) = clip(mu + nu * d / N_Z, 0, 1) and d^T lambda(nu) is nondecreasing in nu.
    # nu is the multiplier of d^T lambda >= 0, so the gradient of the cost w.r.t. d is -nu * lambda.
    cost = lambda lam: np.sum(N_Z * np.square(mu - lam)) / (2 * SIGMA**2)

    lam = np.clip(mu, 0, 1)
    if d @ lam >= 0:
        return cost(lam), lam, 0.0

    c = SIGMA**2 * d / N_Z
    # the closed form of the problem without the box constraint, usually already inside the box
    nu = -(d @ mu) / (d @ c)
    lam = mu + nu * c
    if np.all(lam >= 0) and np.all(lam <= 1):
        return cost(lam), lam, nu

    # otherwise find the root of nu -> d^T lambda(nu) between two of its breakpoints
    nz = c != 0
    breakpoints = np.concatenate(((0 - mu[nz]) / c[nz], (1 - mu[nz]) / c[nz]))
    breakpoints = np.sort(breakpoints[breakpoints > 0])
    g = lambda nu: d @ np.clip(mu + nu * c, 0, 1)
    values = np.array([g(nu) for nu in breakpoints])
    idx = np.argmax(values >= 0)
    lo = breakpoints[idx - 1] if idx > 0 else 0.0
    hi = breakpoints[idx]
    g_lo, g_hi = g(lo), values[idx]
    nu = lo + (hi - lo) * (0 - g_lo) / (g_hi - g_lo) if g_hi > g_lo else hi
    lam = np.clip(mu + nu * c, 0, 1)
    return cost(lam), lam, nu


def optimal_A(mu, A, w, mu_p, s, solver=None, slack=0):
    # convex problem: best context distributions of i_star and s for fixed rewards mu_p
    n, k = A.shape
    i_star, _ = best_arm(mu, A)

    A_pi = cp.Variable(k)
    A_ps = cp.Variable(k)

    constraints = [
        (A_ps - A_pi) @ mu_p >= 0 - slack,
        A_pi >= 0,
        A_ps >= 0,
        cp.sum(A_pi) == 1,
        cp.sum(A_ps) == 1
    ]

    objective = (
        w[i_star] * cp.sum(cp.rel_entr(A[i_star], A_pi)) +
        w[s] * cp.sum(cp.rel_entr(A[s], A_ps))
    )

    problem = cp.Problem(cp.Minimize(objective), constraints)

    try:
        problem.solve(solver=solver)
        A_p = A.copy()
        A_p[i_star] = np.clip(A_pi.value, 1e-12, None)
        A_p[s] = np.clip(A_ps.value, 1e-12, None)
        if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            print(f"non optimal value at A {problem.status}")
        return A_p
    except Exception as e:
        print(f"failed optimization at A: {e}")
        return A


############################## GLR Optimization

def plain_GLR_pair(mu, A, N_A, N_Z, s, n_random=3, method="BFGS", rng=None):
    # local solver over the context distributions (B_{i_star}, B_s) = softmax(x), the rewards are solved exactly
    # by optimal_mu. Starts from B = A, from the best B for lambda = mu, and from n_random random points.
    n, k = A.shape
    i_star, _ = best_arm(mu, A)
    rng = np.random if rng is None else rng
    log_A_star, log_A_s = np.log(A[i_star]), np.log(A[s])
    # the GLR is linear in (N_A, N_Z), so we solve with weights that sum to 1 and scale back
    scale = np.sum(N_A)
    N_A, N_Z = N_A / scale, N_Z / scale

    def solved_mu_objective(x):
        log_B_star, log_B_s = log_softmax(x[:k]), log_softmax(x[k:])
        B_star, B_s = np.exp(log_B_star), np.exp(log_B_s)
        reward_cost, lam, nu = optimal_mu(mu, N_Z, B_s - B_star)
        result = N_A[i_star] * A[i_star] @ (log_A_star - log_B_star) + N_A[s] * A[s] @ (log_A_s - log_B_s) + reward_cost

        # gradient through the softmax, the reward part comes from the envelope theorem
        g = -nu * lam
        grad_star = N_A[i_star] * (B_star - A[i_star]) - B_star * (g - B_star @ g)
        grad_s = N_A[s] * (B_s - A[s]) + B_s * (g - B_s @ g)
        return result, np.concatenate((grad_star, grad_s))

    A0 = optimal_A(mu, A, N_A, np.clip(mu, 0, 1), s)
    starts = [np.concatenate((log_A_star, log_A_s)), np.log(np.concatenate((A0[i_star], A0[s])))]
    starts += [np.log(rng.dirichlet(np.ones(k), size=2)).ravel() for _ in range(n_random)]

    best = None
    for x0 in starts:
        if method == "Nelder-Mead":
            result = minimize(lambda x: solved_mu_objective(x)[0], x0=x0, method=method,
                              options={"xatol": 1e-6, "fatol": 1e-9, "maxiter": 400 * k})
        else:
            result = minimize(solved_mu_objective, x0=x0, jac=True, method=method, options={"gtol": 1e-10})
        if best is None or result.fun < best.fun:
            best = result
    if method != "Nelder-Mead":
        # the objective has kinks (e.g. where B_star = B_s), a short gradient-free polish handles them
        result = minimize(lambda x: solved_mu_objective(x)[0], x0=best.x, method="Nelder-Mead",
                          options={"xatol": 1e-8, "fatol": 1e-12, "maxiter": 100 * k})
        if result.fun < best.fun:
            best = result

    A_p = np.copy(A)
    A_p[i_star] = softmax(best.x[:k])
    A_p[s] = softmax(best.x[k:])
    obj, mu_p, _ = optimal_mu(mu, N_Z, A_p[s] - A_p[i_star])
    obj += N_A[i_star] * categorical_kl(A[i_star], A_p[i_star]) + N_A[s] * categorical_kl(A[s], A_p[s])
    return obj * scale, mu_p, A_p


def plain_GLR(mu, A, N_A, N_Z, n_random=3, verbose=False):
    n, k = A.shape
    i_star, _ = best_arm(mu, A)

    obj_star, mu_star, A_star = np.inf, None, None
    for s in range(n):
        if s == i_star:
            continue
        obj, mu_p, A_p = plain_GLR_pair(mu, A, N_A, N_Z, s, n_random=n_random)
        if obj < obj_star:
            obj_star, mu_star, A_star = obj, mu_p, A_p
    return obj_star, mu_star, A_star


def exact_GLR_k2_pair(mu, A, N_A, N_Z, s):
    # With 2 contexts (B_s - B_star)^T lambda = (b_s - b_star)(lambda_1 - lambda_2), so an alternative either flips
    # the order of the rewards (keeping A) or the order of the context probabilities (keeping lambda = clip(mu)).
    n, k = A.shape
    assert k == 2
    i_star, _ = best_arm(mu, A)
    kl = lambda p, x: p * np.log(p / x) + (1 - p) * np.log((1 - p) / (1 - x))

    # flip the rewards
    obj, mu_p, _ = optimal_mu(mu, N_Z, A[s] - A[i_star])
    A_p = A.copy()

    # flip the context probabilities: both arms meet at the weighted average of their first context probability
    lam0 = np.clip(mu, 0, 1)
    base_cost = np.sum(N_Z * np.square(mu - lam0)) / (2 * SIGMA**2)
    if lam0[0] == lam0[1]:
        obj_c, x = base_cost, None
    else:
        x = (N_A[s] * A[s, 0] + N_A[i_star] * A[i_star, 0]) / (N_A[s] + N_A[i_star])
        obj_c = N_A[s] * kl(A[s, 0], x) + N_A[i_star] * kl(A[i_star, 0], x) + base_cost
    if obj_c < obj:
        obj, mu_p = obj_c, lam0
        if x is not None:
            A_p[s] = A_p[i_star] = [x, 1 - x]
    return obj, mu_p, A_p


def exact_GLR_k2(mu, A, N_A, N_Z, verbose=False):
    n, k = A.shape
    i_star, _ = best_arm(mu, A)

    obj_star, mu_star, A_star = np.inf, None, None
    for s in range(n):
        if s == i_star:
            continue
        obj, mu_p, A_p = exact_GLR_k2_pair(mu, A, N_A, N_Z, s)
        if obj < obj_star:
            obj_star, mu_star, A_star = obj, mu_p, A_p
    return obj_star, mu_star, A_star


def reference_GLR_pair(mu, A, N_A, N_Z, s, n_random=40):
    # our best estimate of the true (non-convex) Lambda_s: exact for k = 2, many starting points otherwise
    if A.shape[1] == 2:
        return exact_GLR_k2_pair(mu, A, N_A, N_Z, s)
    return plain_GLR_pair(mu, A, N_A, N_Z, s, n_random=n_random)


def reference_GLR(mu, A, N_A, N_Z, n_random=40, verbose=False):
    # our best estimate of the true (non-convex) GLR: exact for k = 2, many starting points otherwise
    if A.shape[1] == 2:
        return exact_GLR_k2(mu, A, N_A, N_Z)
    return plain_GLR(mu, A, N_A, N_Z, n_random=n_random)


def grid_search_GLR(mu, A, N_A, N_Z, div=21, div2=21, solver=cp.CLARABEL, verbose=True):
    # slow, only used to validate the other methods
    n, k = A.shape

    i_star, _ = best_arm(mu, A)

    obj_star = np.inf
    mu_star, A_star = None, None

    grid_range = np.linspace(0, 1, div)
    grid = itertools.product(grid_range, repeat=k)
    for mu_p in grid:
        mu_p = np.array(mu_p)
        for s in range(n):
            if s == i_star:
                continue
            A_p = optimal_A(mu, A, N_A, mu_p, s)

            obj = objective(mu, A, mu_p, A_p, N_A, N_Z)
            if obj < obj_star:
                obj_star = obj
                mu_star = mu_p
                A_star = A_p

    mu_p_base = mu_star.copy()
    gap = 1/(div-1)
    grid_range = np.linspace(-gap, gap, div2)
    grid = itertools.product(grid_range, repeat=k)
    for noise in grid:
        mu_p = np.clip(mu_p_base + noise, 0, 1)
        for s in range(n):
            if s == i_star:
                continue
            A_p = optimal_A(mu, A, N_A, mu_p, s, solver=solver)

            obj = objective(mu, A, mu_p, A_p, N_A, N_Z)
            if obj < obj_star:
                obj_star = obj
                mu_star = mu_p
                A_star = A_p

    return obj_star, mu_star, A_star


########## Wrapper

ALGS_GLR = {
    "plain": plain_GLR,
    "reference": reference_GLR,
    "exact_k2": exact_GLR_k2,
    "grid": grid_search_GLR,
}

def optimize_GLR(mu, A, N_A, N_Z, alg="plain", verbose=False):
    if alg not in ALGS_GLR.keys():
        print("Invalid GLR alg name!")
        return None
    optimization_alg = ALGS_GLR[alg]

    return optimization_alg(mu, A, N_A, N_Z, verbose=verbose)


############################## w optimization

def optimize_bfgs(mu, A, method="BFGS", inner_alg="plain", w0=None, verbose=False):
    # maximizes the concave w -> F(mu, A, w, A^T w) over softmax(w), the gradient comes from Danskin's theorem
    n, k = A.shape

    if w0 is None:
        w0 = np.ones(n) / n
    # Need to convert w0 to pre-image of softmax
    EPS = 1e-10
    w0 = np.log(w0 + EPS)

    # caching the optimization to be used both by fun and jac
    last_w, last_result = None, None
    def solve(w):
        nonlocal last_w, last_result
        w_soft = softmax(w)
        if not np.array_equal(w_soft, last_w):
            last_result = optimize_GLR(mu, A, w_soft, np.dot(A.T, w_soft), alg=inner_alg)
        last_w = w_soft
        return w_soft, last_result

    def fun(w):
        return -solve(w)[1][0]

    def jac(w):
        w_soft, (obj_star, mu_star, A_star) = solve(w)
        # d/dw_i of sum_i w_i KL(A_i||B_i) + sum_j (A^T w)_j d(mu_j, lambda_j), at the optimal alternative
        grad = np.array([categorical_kl(A[i], A_star[i]) for i in range(n)])
        grad += A @ (np.square(mu - mu_star) / (2 * SIGMA**2))
        grad = np.multiply(w_soft, (grad - np.dot(w_soft, grad)))  # accounting for softmax
        return -grad

    result = minimize(
            fun=fun,
            jac=jac,
            x0=w0,
            method=method,
            options={"disp": verbose}
        )

    w_star = softmax(result.x)
    obj_star, mu_star, A_star = optimize_GLR(mu, A, w_star, np.dot(A.T, w_star), alg=inner_alg)

    return obj_star, w_star


############################## optimization functions

ALGS = {
    "bfgs": optimize_bfgs,
}

def optimize(mu, A, alg="bfgs", inner_alg="plain", w0=None, verbose=False):
    if alg not in ALGS.keys():
        print("Invalid alg name!")
        return None
    optimization_alg = ALGS[alg]

    return optimization_alg(mu, A, inner_alg=inner_alg, w0=w0, verbose=verbose)


############################## optimization testing

def test_method_fixed_w(n, k, name="plain", experiment_cnt=None):
    # compares a GLR method with the grid search ground truth stored in testsets/
    testset_path = TESTSET_DIR + f"fixed_w_n={n}_k={k}.json"
    with open(testset_path, 'r') as F:
        testset = json.load(F)
    if experiment_cnt is None:
        experiment_cnt = len(testset)

    gaps, ratios = [], []
    fail_cnt = 0
    for experiment in tqdm(testset[:experiment_cnt], "testing"):
        mu = np.array(experiment["mu"])
        A = np.array(experiment["A"])
        w = np.array(experiment["w"])
        obj_star = experiment["obj_star"]

        alg_obj, _, _ = optimize_GLR(mu, A, w, np.dot(A.T, w), alg=name)

        gaps.append(alg_obj - obj_star)
        ratios.append(alg_obj / obj_star)
        if obj_star + TOL < alg_obj:
            fail_cnt += 1
            print(f"##### failed test {experiment['index']} with {alg_obj - obj_star} gap!")

    print(f"{name} on n={n}, k={k}, {experiment_cnt} tests")
    print("success rate:", 1 - fail_cnt / experiment_cnt)
    print("worst suboptimality gap:", max(gaps), "best:", min(gaps))
    print("ratio to grid search: min", min(ratios), "median", np.median(ratios), "max", max(ratios))


def test_method(n, k, name="bfgs", experiment_cnt=None):
    # compares a w optimization method with the grid search ground truth stored in testsets/
    testset_path = TESTSET_DIR + f"opt_n={n}_k={k}.json"
    with open(testset_path, 'r') as F:
        testset = json.load(F)
    if experiment_cnt is None:
        experiment_cnt = len(testset)

    gaps = []
    fail_cnt = 0
    for experiment in tqdm(testset[:experiment_cnt], "testing"):
        mu = np.array(experiment["mu"])
        A = np.array(experiment["A"])
        obj_star = experiment["obj_star"]

        alg_obj, alg_w = optimize(mu, A, alg=name)

        gaps.append(obj_star - alg_obj)
        if obj_star - TOL > alg_obj:
            fail_cnt += 1
            print(f"##### failed test {experiment['index']} with {obj_star - alg_obj} gap!")

    print(f"{name} on n={n}, k={k}, {experiment_cnt} tests")
    print("success rate:", 1 - fail_cnt / experiment_cnt)
    print("worst suboptimality gap:", max(gaps), "best:", min(gaps))


########## main

if __name__ == "__main__":
    args = get_optimization_arguments()
    if args.fixed_w:
        test_method_fixed_w(args.n, args.k, experiment_cnt=args.experiment_cnt, name=args.name)
    else:
        test_method(args.n, args.k, experiment_cnt=args.experiment_cnt, name=args.name)
