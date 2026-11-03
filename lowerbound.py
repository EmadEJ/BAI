import numpy as np
import cvxpy as cp
from scipy.optimize import minimize_scalar

# Closed-form lower bounds on the information cost Lambda(w, q) (eq. (1) of the bound report).
# Convention of the repo: A is n x k (row i is the context distribution of arm i), mu has k entries,
# w has n entries (arm counts N_A or weights) and q has k entries (context counts N_Z or A^T w).

SIGMA = 1.0  # unit-variance Gaussian rewards


def c_sigma(sigma=SIGMA):
    return 1 + np.sqrt(1 + sigma**2)


def variance(P, f):
    return P @ np.square(f - P @ f)


def pair_terms(mu, A, w, q, i, i_star, sigma=SIGMA):
    # gap, H_i and m = min(w_i, w_*) of the pair (i, i_star)
    D = A[i] - A[i_star]
    delta = max(-D @ mu, 0.0)
    H = variance(A[i], mu) / w[i] + variance(A[i_star], mu) / w[i_star] + sigma**2 * np.sum(D**2 / q)
    m = min(w[i], w[i_star])
    return delta, H, m


def L_bound(mu, A, w, q, sigma=SIGMA):
    # L_i of Theorem 1 for every arm (inf at the best arm), Lambda >= min(L)
    n = A.shape[0]
    i_star = np.argmax(A @ mu)

    L = np.full(n, np.inf)
    for i in range(n):
        if i == i_star:
            continue
        delta, H, m = pair_terms(mu, A, w, q, i, i_star, sigma)
        L[i] = delta**2 / (2 * H + 2 * c_sigma(sigma) / m * delta)
    return L


def L_naive(mu, A, w, q, sigma=SIGMA):
    # L_i^+ of Proposition 6: the variances replaced by their worst case 1/4 and no correction term
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


def R_factor(gamma, sigma=SIGMA):
    return (1 + c_sigma(sigma) * gamma) * (1 + 3 * sigma * gamma)**2 / (1 - 2 * gamma)


def hard_regime(gamma, mu, sigma=SIGMA):
    # conditions of Theorem 3
    return (
        sigma * gamma <= 0.1 and gamma <= 0.25 and
        np.all(mu >= 1.3 * sigma**2 * gamma) and np.all(mu <= 1 - 1.3 * sigma**2 * gamma)
    )


def proven_factor(mu, A, w=None, q=None, sigma=SIGMA):
    """
    Per-arm factor f_i with L_i <= Lambda_i <= f_i * L_i.
    Without (w, q) this is the allocation-free factor min(rho_i, R(gamma_bar_i)) of Theorems 2-3.
    With (w, q) Theorem 3 is used at the gamma_i of that allocation, which can only be smaller.
    """
    n = A.shape[0]
    i_star = np.argmax(A @ mu)
    cs = c_sigma(sigma)
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

        if mu_in_box:
            T_i = np.sum(D**2 / (A[i] + A[i_star]))
            K_i = max(
                kinf(A[i], mu, A[i_star] @ mu),  # move arm i up to the mean of i_star
                kinf(A[i_star], 1 - mu, 1 - A[i] @ mu)  # move i_star down to the mean of i
            )
            factor[i] = K_i * (2 * (V_i + V_star + sigma**2 * T_i) / delta**2 + 2 * cs / delta)

        if w is None:
            gamma = delta / min(V_i, V_star)
        else:
            _, H, m = pair_terms(mu, A, w, q, i, i_star, sigma)
            gamma = delta / (m * H)
        if hard_regime(gamma, mu, sigma):
            factor[i] = min(factor[i], R_factor(gamma, sigma))
    return factor


############################## sampling rule

def optimal_w(mu, A, bound="L", sigma=SIGMA):
    """
    Maximizer of w -> min_i L_i(w, A^T w) over the simplex (or of L_naive with bound="naive").
    Both are concave in w, so we minimize the convex max_i 1/L_i. Returns (U^-1, w).
    """
    n, k = A.shape
    i_star = np.argmax(A @ mu)

    w = cp.Variable(n, nonneg=True)
    q = A.T @ w
    inv_L = []
    for i in range(n):
        if i == i_star:
            continue
        D = A[i] - A[i_star]
        delta = max(-D @ mu, 1e-9)
        reward_term = sigma**2 * cp.sum(cp.multiply(D**2, cp.inv_pos(q)))
        if bound == "L":
            H = variance(A[i], mu) * cp.inv_pos(w[i]) + variance(A[i_star], mu) * cp.inv_pos(w[i_star]) + reward_term
            denom = 2 * H + 2 * c_sigma(sigma) * delta * cp.inv_pos(cp.minimum(w[i], w[i_star]))
        else:
            denom = 2 * (cp.inv_pos(w[i]) / 4 + cp.inv_pos(w[i_star]) / 4 + reward_term)
        inv_L.append(denom / delta**2)

    problem = cp.Problem(cp.Minimize(cp.max(cp.hstack(inv_L))), [cp.sum(w) == 1])
    problem.solve()

    w_star = np.clip(w.value, 0, None)
    return 1 / problem.value, w_star / np.sum(w_star)
