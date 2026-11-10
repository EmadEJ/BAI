import numpy as np
import cvxpy as cp
from scipy.optimize import minimize_scalar

# Closed-form lower bounds on the information cost Lambda(w, q) (eq. (1) of the bound report).
# Convention of the repo: A is n x k (row i is the context distribution of arm i), mu has k entries,
# w has n entries (arm counts N_A or weights) and q has k entries (context counts N_Z or A^T w).

SIGMA = 1.0  # unit-variance Gaussian rewards


def variance(P, f):
    return P @ np.square(f - P @ f)


def pair_terms(mu, A, w, q, i, i_star, sigma=SIGMA):
    # gap, H_i, kappa^N_i and m = min(w_i, w_*) of the pair (i, i_star), eqs. (3)-(6) of the bound report
    D = A[i] - A[i_star]
    delta = max(-D @ mu, 0.0)
    H = variance(A[i], mu) / w[i] + variance(A[i_star], mu) / w[i_star] + sigma**2 * np.sum(D**2 / q)
    a = max((1 - np.min(A[i])) / w[i], (1 - np.min(A[i_star])) / w[i_star])
    s = np.max((A[i] / w[i] + A[i_star] / w[i_star]) / q)
    kappa = (a + np.sqrt(a**2 + 4 * sigma**2 * s)) / 2
    m = min(w[i], w[i_star])
    return delta, H, kappa, m


def root_bound(delta, H, kappa):
    # the C solving delta = sqrt(2 H C) + kappa C
    return delta**2 / (H + kappa * delta + np.sqrt(H**2 + 2 * H * kappa * delta))


def L_bound(mu, A, w, q, sigma=SIGMA):
    # L_i^new of Theorem 1 for every arm (inf at the best arm), Lambda >= min(L)
    n = A.shape[0]
    i_star = np.argmax(A @ mu)

    L = np.full(n, np.inf)
    for i in range(n):
        if i == i_star:
            continue
        delta, H, kappa, _ = pair_terms(mu, A, w, q, i, i_star, sigma)
        L[i] = root_bound(delta, H, kappa)
    return L


def L_naive(mu, A, w, q, sigma=SIGMA):
    # L_i^+ of eq. (26) (Pinsker): the variances replaced by their worst case 1/4 and no correction term
    n = A.shape[0]
    i_star = np.argmax(A @ mu)

    L = np.full(n, np.inf)
    for i in range(n):
        if i == i_star:
            continue
        D = A[i] - A[i_star]
        delta = max(-D @ mu, 0.0)
        L[i] = delta**2 / (2 * (1 / (4 * w[i]) + 1 / (4 * w[i_star]) + sigma**2 * np.sum(D**2 / q)))
    return L


############################## proven gap (Theorems 2 and 3)

def kinf(P, f, x):
    # inf {KL(P||Q) : Q^T f >= x} = max_{0 <= theta < 1/(max f - x)} sum_j P_j ln(1 - theta (f_j - x))
    if P @ f >= x:
        return 0.0
    if x >= np.max(f):
        return np.inf
    theta_max = 1 / (np.max(f) - x)
    res = minimize_scalar(
        lambda u: -P @ np.log1p(-u * theta_max * (f - x)),
        bounds=(0, 1 - 1e-12),
        method="bounded",
        options={"xatol": 1e-12}
    )
    return -res.fun


def hard_factor(gamma, b, sigma=SIGMA):
    # d_i C_i of Theorem 3
    d = (1 + b * gamma + np.sqrt(1 + 2 * b * gamma)) / 2
    return d * (1 + 3 * sigma * gamma)**2 / (1 - 2 * gamma)


def hard_regime(gamma, mu, sigma=SIGMA):
    # conditions of Theorem 3
    return (
        sigma * gamma <= 0.1 and gamma <= 0.25 and
        np.all(mu >= 1.3 * sigma**2 * gamma) and np.all(mu <= 1 - 1.3 * sigma**2 * gamma)
    )


def proven_factor(mu, A, w=None, q=None, sigma=SIGMA):
    """
    Per-arm factor f_i with L_i <= Lambda_i <= f_i * L_i.
    Without (w, q) this is the allocation-free factor min(rho_i, d_i C_i at (beta_i, gamma_bar_i)) of Theorems 2-3.
    With (w, q) Theorem 3 is used at the (b_i, gamma_i) of that allocation, which can only be smaller.
    """
    n = A.shape[0]
    i_star = np.argmax(A @ mu)
    mu_in_box = np.all(mu >= 0) and np.all(mu <= 1)  # Theorem 2 needs lambda = mu to be feasible

    factor = np.full(n, np.inf)
    for i in range(n):
        if i == i_star:
            continue
        D = A[i] - A[i_star]
        delta = -D @ mu
        if delta <= 0:
            continue
        V_i, V_star = variance(A[i], mu), variance(A[i_star], mu)
        p = np.min(np.minimum(A[i], A[i_star]))
        beta = (1 - p + np.sqrt((1 - p)**2 + 4 * sigma**2)) / 2

        if mu_in_box:
            H0 = V_i + V_star + sigma**2 * np.sum(D**2 / (A[i] + A[i_star]))
            K_i = max(
                kinf(A[i], mu, A[i_star] @ mu),  # move arm i up to the mean of i_star
                kinf(A[i_star], 1 - mu, 1 - A[i] @ mu)  # move i_star down to the mean of i
            )
            F_i = H0 + beta * delta + np.sqrt(H0**2 + 2 * H0 * beta * delta)
            factor[i] = K_i * F_i / delta**2

        if w is None:
            gamma, b = delta / min(V_i, V_star), beta
        else:
            _, H, kappa, m = pair_terms(mu, A, w, q, i, i_star, sigma)
            gamma, b = delta / (m * H), m * kappa
        if hard_regime(gamma, mu, sigma):
            factor[i] = min(factor[i], hard_factor(gamma, b, sigma))
    return factor


############################## sampling rule

def optimal_w(mu, A, bound="L", sigma=SIGMA):
    """
    Maximizer of w -> min_i L_i(w, A^T w) over the simplex (or of L_naive with bound="naive"). Returns (U^-1, w).
    L_naive: 1/L_naive_i is convex, so we minimize max_i 1/L_naive_i.
    L: L_i is concave and homogeneous of degree one in w, and L_i(w) >= c <=> sqrt(2 H_i c) + kappa_i c <= delta_i.
    So we minimize sum(w) s.t. L_i(w) >= c for every i, and U^-1 = c / sum(w). sqrt(H_i) and kappa_i are convex in w,
    they are written with the epigraph variables u >= w^-1/2, r >= q^-1/2 and g >= (w_l q_j)^-1/2.
    """
    n, k = A.shape
    i_star = np.argmax(A @ mu)

    w = cp.Variable(n, nonneg=True)
    q = A.T @ w
    deltas = np.maximum((A[i_star] - A) @ mu, 1e-9)
    scale = np.min(np.delete(deltas, i_star))**2  # keeps the problem of order 1 when two arms are almost tied

    if bound == "naive":
        inv_L = []
        for i in range(n):
            if i == i_star:
                continue
            D = A[i] - A[i_star]
            reward_term = sigma**2 * cp.sum(cp.multiply(D**2, cp.inv_pos(q)))
            denom = 2 * (cp.inv_pos(w[i]) / 4 + cp.inv_pos(w[i_star]) / 4 + reward_term)
            inv_L.append(denom * (scale / deltas[i]**2))
        problem = cp.Problem(cp.Minimize(cp.max(cp.hstack(inv_L))), [cp.sum(w) == 1])
    else:
        u, r, g = cp.Variable(n), cp.Variable(k), cp.Variable((n, k))
        a, root_s = cp.Variable(n), cp.Variable(n)  # a_i and sqrt(s_i) of eq. (5)
        R = 1 - np.min(A, axis=1)
        constraints = [u >= cp.power(w, -0.5), r >= cp.power(q, -0.5)]
        constraints += [g[l, j] >= cp.inv_pos(cp.geo_mean(cp.hstack([w[l], q[j]]))) for l in range(n) for j in range(k)]
        for i in range(n):
            if i == i_star:
                continue
            D = A[i] - A[i_star]
            root_H = cp.norm(cp.hstack([
                np.sqrt(variance(A[i], mu)) * u[i], np.sqrt(variance(A[i_star], mu)) * u[i_star], sigma * cp.multiply(np.abs(D), r)
            ]))
            constraints += [
                a[i] >= R[i] * cp.inv_pos(w[i]),
                a[i] >= R[i_star] * cp.inv_pos(w[i_star]),
                root_s[i] >= cp.max(cp.hstack([
                    cp.norm(cp.hstack([np.sqrt(A[i, j]) * g[i, j], np.sqrt(A[i_star, j]) * g[i_star, j]])) for j in range(k)
                ])),
            ]
            kappa = (a[i] + cp.norm(cp.hstack([a[i], 2 * sigma * root_s[i]]))) / 2
            constraints.append(np.sqrt(2 * scale) * root_H + scale * kappa <= deltas[i])
        problem = cp.Problem(cp.Minimize(cp.sum(w)), constraints)

    try:
        problem.solve()
    except cp.error.SolverError:
        problem.solve(solver=cp.SCS)

    w_star = np.clip(w.value, 0, None)
    if bound == "naive":
        return scale / problem.value, w_star / np.sum(w_star)
    return scale / np.sum(w_star), w_star / np.sum(w_star)
