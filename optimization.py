import numpy as np
import cvxpy as cp
import spsa
from tqdm import tqdm
import itertools
from utils import *
from io_utils import *
from scipy.optimize import minimize, Bounds, LinearConstraint
from scipy.special import softmax
import plotly.express as px
from pathlib import Path


TESTSET_DIR = "testsets/"

TOL = 1e-4  # used for checking whether the optimization was good enough


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


def fixed_w_grid_search(mu, A, w, div=21, div2=11, solver=cp.CLARABEL, verbose=True):
    n, k = A.shape
    
    N_A = w
    N_Z = np.dot(A.T, w)
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
            A_p = optimal_A(mu, A, w, mu_p, s)

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
            A_p = optimal_A(mu, A, w, mu_p, s, solver=solver)

            obj = objective(mu, A, mu_p, A_p, N_A, N_Z)
            if obj < obj_star:
                obj_star = obj
                mu_star = mu_p
                A_star = A_p
    
    return obj_star, mu_star, A_star


def grid_search(mu, A, div=101, solver=cp.CLARABEL, verbose=True):
    n, k = A.shape
    # Used coordinate ternary search to get to the result
    obj_star = -np.inf  
    w_star = None
    
    objs = []
    grid_range = np.linspace(0, 1, div)
    grid = itertools.product(grid_range, repeat=n-1)
    # Note that w[0] is always 1 - sum of the rest.
    for w_p in grid:
        w0 = 1 - sum(w_p)
        if w0 < 0:
            continue
        
        w_p = np.array(([w0] + list(w_p)))
        
        obj_p, _, _ = fixed_w_grid_search(mu, A, w_p, solver=solver, div=11, div2=11, verbose=False)
        # obj_p, _, _ = optimize_solved_mu(mu, A, w_p, np.dot(A.T, w_p), method="SLSQP")
        objs.append(obj_p)
        
        if obj_p > obj_star:
            w_star = w_p
            obj_star = obj_p
    
    if verbose and n == 2:
        fig = px.line(x=np.linspace(0, 1, div), y=objs)
        fig.show()
    
    obj_star, _, _ = optimize_solved_mu(mu, A, w_star, np.dot(A.T, w_star), method="SLSQP") 
    return obj_star, w_star


def ternary_search(mu, A, EPS=1e-3, max_iter=20, solver=cp.CLARABEL, verbose=True):
    n, k = A.shape
    # Used coordinate ternary search to get to the result
    w_star = np.random.rand(n)
    w_star = w_star / np.sum(w_star)
    
    # Note that w[0] is always 1 - sum of the rest.
    for _ in range(max_iter):
        old_w_star = w_star.copy()
        
        for i in tqdm(range(1, n), desc="optimizing over coordinates", disable=not verbose, leave=False):
            
            l = 0.0
            r = w_star[i] + w_star[0]
            
            while l + EPS < r:
                wi1 = (5*l + 4*r) / 9
                w_p1 = np.copy(w_star)
                w_p1[i] = wi1
                w_p1[0] = w_star[i] + w_star[0] - wi1
                w_p1 = w_p1 / sum(w_p1)
                wi2 = (4*l + 5*r) / 9
                w_p2 = np.copy(w_star)
                w_p2[i] = wi2
                w_p2[0] = w_star[i] + w_star[0] - wi2
                w_p2 = w_p2 / sum(w_p2)
                obj1, _, _ = fixed_w_grid_search(mu, A, w_p1, div=11, div2=11, verbose=False)
                obj2, _, _ = fixed_w_grid_search(mu, A, w_p2, div=11, div2=11, verbose=False)
                if obj1 > obj2:
                    r = wi2
                else:
                    l = wi1
            
            wi = (l + r) / 2
            w0 = w_star[i] + w_star[0] - wi
            w_star[i] = wi
            w_star[0] = w0
            w_star = w_star / np.sum(w_star)  # make sure sum to 1
        
        if np.allclose(old_w_star, w_star, atol=EPS, rtol=0):
            break

    obj_star, _, _ = fixed_w_grid_search(mu, A, w_star, solver=solver, verbose=False)    
    return obj_star, w_star


def optimal_mu(mu, A, w, A_p, s, slack=0):
    n, k = A.shape
    i_star, _ = best_arm(mu, A)
    
    mu_p = cp.Variable(k)
    
    constraint = [(A_p[s] - A_p[i_star]) @ mu_p >= 0 - slack]
    
    objective = 0.5 * cp.sum(cp.multiply(np.dot(w, A), cp.square(mu - mu_p)))
    
    problem = cp.Problem(cp.Minimize(objective), constraint)
    
    try:
        problem.solve()
        if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            return mu_p.value
        else:
            print("non optimal value at mu")
            return None
    except:
        print("failed optimization at mu")
        return None
    

def optimal_A(mu, A, w, mu_p, s, solver=None, slack=0):
    n, k = A.shape
    i_star, _ = best_arm(mu, A)
    
    A_p = cp.Variable((n, k))
    
    constraints = [(A_p[s] - A_p[i_star]) @ mu_p >= 0 - slack]
    for i in range(n):
        constraints += [cp.sum(A_p[i]) == 1]
    
    objective = cp.sum([cp.multiply(w[i], cp.sum(cp.rel_entr(A[i], A_p[i]))) for i in range(n)])
    
    problem = cp.Problem(cp.Minimize(objective), constraints)
    
    try:
        problem.solve(solver=solver)
        if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            return A_p.value
        else:
            print("non optimal value at A")
            return None
    except:
        print("failed optimization at A")
        return None


def optimal_w(mu, A, mu_p, A_p, solver=None):
    n, k = A.shape
    
    w = cp.Variable(n, nonneg=True)
    coef = np.array([categorical_kl(A[i], A_p[i]) + 0.5*np.dot(A[i], np.square(mu - mu_p)) for i in range(n)])
    constraints = [cp.sum(w) == 1]
    
    objective = cp.sum(cp.multiply(w, coef))
    
    problem = cp.Problem(cp.Maximize(objective), constraints)
    
    try:
        problem.solve(solver=solver)
        if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            return w.value
        else:
            print("non optimal value at w")
            return None
    except:
        print("failed optimization at w")
        return None


def contribution_optimization(mu, A, w, verbose=True):
    n, k = A.shape
    
    means = np.dot(A, mu)
    i_star = np.argmax(means)
    
    obj_star = np.inf
    mu_star, A_star = None, None
    for s in range(n):
        if s == i_star:
            continue
        
        delta_s = means[i_star] - means[s]
        
        objs_A = []
        objs_mu = []
        grid = np.linspace(0, 1, 101)
        for A_contribution in grid:
            mu_contribution = 1 - A_contribution
            
            # A first
            A_p = optimal_A(mu, A, w, mu, s, slack=delta_s*mu_contribution)
            mu_p = optimal_mu(mu, A, w, A_p, s)
            obj = objective(mu, A, mu_p, A_p, w, np.dot(A.T, w))
            objs_A.append(obj)
            if obj < obj_star:
                obj_star = obj
                mu_star = mu_p
                A_star = A_p
                
            # mu first
            mu_p = optimal_mu(mu, A, w, A, s, slack=delta_s*A_contribution)
            A_p = optimal_A(mu, A, w, mu_p, s)
            obj = objective(mu, A, mu_p, A_p, w, np.dot(A.T, w))
            objs_mu.append(obj)
            if obj < obj_star:
                obj_star = obj
                mu_star = mu_p
                A_star = A_p
        
        if True:
            plt.plot(grid, objs_A, label="A first")
            plt.plot(grid, objs_mu, label="mu first")
            plt.legend()
            # plt.show()

    return obj_star, mu_star, A_star


def coordinate_descent(mu, A, w, iters=5, verbose=True):
    n, k = A.shape
    
    i_star, _ = best_arm(mu, A)
    
    obj_star = np.inf
    mu_star, A_star = None, None
    for s in tqdm(range(n), desc="coordinate_descent", disable=~verbose):
        if s == i_star:
            continue
        
        # mu first
        A_p = A
        mu_p = mu
        for _ in range(iters):
            mu_p = optimal_mu(mu, A, w, A_p, s)
            A_p = optimal_A(mu, A, w, mu_p, s)
        
        obj = objective(mu, A, mu_p, A_p, w, np.dot(A.T, w))
        if obj < obj_star:
            obj_star = obj
            mu_star = mu_p
            A_star = A_p
        
        # A first
        A_p = A
        mu_p = mu
        for _ in range(iters):
            A_p = optimal_A(mu, A, w, mu_p, s)
            mu_p = optimal_mu(mu, A, w, A_p, s)
        
        obj = objective(mu, A, mu_p, A_p, w, np.dot(A.T, w))
        if obj < obj_star:
            obj_star = obj
            mu_star = mu_p
            A_star = A_p
    
    return obj_star, mu_star, A_star


def lowerbound_fixed_w(mu, A, w, verbose=False):
    return lowerbound_GLR(mu, A, w, np.dot(A.T, w)), None, None


def my_lowerbound_fixed_w(mu, A, w, verbose=False):
    return my_lowerbound_GLR(mu, A, w, np.dot(A.T, w)), None, None


def optimize_solved_mu(mu, A, N_A, N_Z, method=None, verbose=False):
    n, k = A.shape
    i_star, _ = best_arm(mu, A)

    def solved_mu_objective(A_p_list, s):  # assumes gaussian
        A_p = np.array(A_p_list).reshape((n, k))
        
        result = 0
        for i in range(n):
            result += N_A[i] * categorical_kl(A[i], A_p[i])
        
        denom = 0
        for j in range(k):
            denom += (A_p[i_star][j] - A_p[s][j])**2 / N_Z[j]
        
        delta = max(np.dot(A_p[i_star], mu) - np.dot(A_p[s], mu), 0.0)
        if delta == 0:
            added_val = 0
        elif denom == 0:
            added_val = np.inf
        else:
            added_val = delta**2 / (2*denom)
        result += added_val

        return result
    
    # distribution constraints
    mat = np.zeros((n, n * k))
    for i in range(n):
        for j in range(k):
            mat[i][i*k + j] = 1
    
    constraints = LinearConstraint(mat, [1.0 for _ in range(n)], [1.0 for _ in range(n)])
    bounds = Bounds([1e-6 for _ in range(n * k)], [1.0 for _ in range(n * k)])
    
    obj_star = np.inf
    mu_star, A_star = None, None
    for s in range(n):
        if s == i_star:
            continue

        # TODO: have only 2 rows of the matrix as variables
        result = minimize(
            solved_mu_objective, 
            x0=np.reshape(A, (n*k)).tolist(), 
            args=(s), 
            bounds=bounds, 
            constraints=constraints,
            method=method
        )
        # doing the optimization with 2 starting points
        A0 = optimal_A(mu, A, N_A, mu, s)
        result2 = minimize(
            solved_mu_objective, 
            x0=np.reshape(A0, (n*k)).tolist(), 
            args=(s), 
            bounds=bounds, 
            constraints=constraints,
            method=method
        )
        if result2.fun < result.fun:
            result = result2
        
        A_p = np.array(result.x).reshape((n, k))
        mu_p = optimal_mu(mu, A, N_A, A_p, s)
        obj = objective(mu, A, mu_p, A_p, N_A, N_Z)
        
        if np.abs(result.fun - obj) > 1e-6:
            print("something fishy going on")
            print(obj, result.fun)
            print("A:", A)
            print("mu:", mu)
            print("N_A:", N_A)
            print("N_Z:", N_Z)
            print("A_p:", A_p)
            print("mu_p:", mu_p)
        
        if obj < obj_star:
            obj_star = obj
            mu_star = mu_p
            A_star = A_p

    if verbose and n == 2 and k == 2:
        grid_range = np.linspace(0, 1, 101)
        grid = itertools.product(enumerate(grid_range), repeat=2)
        fun_map = np.zeros((101, 101))
        for x, y in grid:
            i, a00 = x
            j, a10 = y
            
            A_p_list = [a00, 1-a00, a10, 1-a10]
            fun_map[i][j] = solved_mu_objective(A_p_list, 1-i_star)

        fig = px.imshow(np.log(fun_map))
        fig.show()

    return obj_star, mu_star, A_star

def COBYQA_solved_mu(mu, A, w, verbose=True):
    return optimize_solved_mu(mu, A, w, np.dot(A.T, w), "COBYQA", verbose)

def SLSQP_solved_mu(mu, A, w, verbose=True):
    return optimize_solved_mu(mu, A, w, np.dot(A.T, w), "SLSQP", verbose)


# SLSQP doesn't work well in the second layer.
def optimize_scipy(mu, A, method="COBYQA", inner_method="SLSQP", verbose=True):
    n, k = A.shape
    
    def neg_optimize_fixed_w(w, mu, A, method):
        return -optimize_solved_mu(mu, A, w, np.dot(A.T, w), method)[0]

    bounds = Bounds([1e-6 for _ in range(n)], [1.0 for _ in range(n)])
    constraints = LinearConstraint([[1.0 for _ in range(n)]], [1.0], [1.0])
    
    w0 = np.random.rand(n)
    w0 = w0 / np.sum(w0)
    result = minimize(
            neg_optimize_fixed_w, 
            x0=w0, 
            args=(mu, A, inner_method), 
            bounds=bounds, 
            constraints=constraints,
            method=method,
            options={'disp': verbose}
        )
    
    w_star = result.x
    obj_star, mu_star, A_star = optimize_solved_mu(mu, A, w_star, np.dot(A.T, w_star), inner_method)
    
    return obj_star, w_star


def optimize_scipy_softmax(mu, A, method="Nelder-Mead", inner_method="SLSQP", verbose=True):
    n, k = A.shape
    
    w0 = np.random.rand(n)
    w0 = w0 / np.sum(w0)
    
    def fun(w):
        w_normalized = softmax(w)
        return -optimize_solved_mu(mu, A, w_normalized, np.dot(A.T, w_normalized), inner_method)[0]
    
    w0 = np.random.rand(n)
    w0 = w0 / np.sum(w0)
    result = minimize(
            fun, 
            x0=w0, 
            method=method,
            options={'disp': verbose, 'fatol': TOL}
        )
    
    w_star = softmax(result.x)
    obj_star, mu_star, A_star = optimize_solved_mu(mu, A, w_star, np.dot(A.T, w_star), inner_method)

    return obj_star, w_star


def optimize_scipy_grad(mu, A, method="BFGS", inner_method="SLSQP", verbose=True):
    n, k = A.shape
    
    w0 = np.random.rand(n)
    w0 = w0 / np.sum(w0)
    
    def fun(w):
        w_soft = softmax(w)
        return -optimize_solved_mu(mu, A, w_soft, np.dot(A.T, w_soft), inner_method)[0]
    
    def jac(w):
        w_soft = softmax(w)
                
        obj_star, mu_star, A_star = optimize_solved_mu(mu, A, w_soft, np.dot(A.T, w_soft), inner_method)
        i_star, _ = best_arm(mu, A)
        s, _ = best_arm(mu_star, A_star)
        if i_star == s:
            # This happens when mean of s and i_star equal exactly
            # Replace s with the second biggest value
            sorted_arms = np.argsort(np.dot(A_star, mu_star))
            if sorted_arms[-1] == i_star:
                s = sorted_arms[-2]
            else:
                s = sorted_arms[-1]
        
        d_mu = np.square(mu - mu_star) / 2
        grad = A @ d_mu
        grad[i_star] += categorical_kl(A[i_star], A_star[i_star])
        grad[s] += categorical_kl(A[s], A_star[s])
        
        grad = np.multiply(w_soft, (grad - np.dot(w_soft, grad)))  # accounting for softmax
                
        return -grad
    
    w0 = np.random.rand(n)
    w0 = w0 / np.sum(w0)
    result = minimize(
            fun=fun, 
            jac=jac,
            x0=w0, 
            method=method,
            options={"disp": verbose, "xrtol": TOL}
        )
    
    if verbose and n == 2:
        grid = np.linspace(0, 1, 101)
        funs = []
        jacs = []
        for w_0 in grid:
            w_p = np.array([w_0, 1 - w_0])
            funs.append(fun(w_p))
            jacs.append(jac(w_p)[0])
        
        plt.plot(grid, funs, label="function")
        plt.plot(grid, jacs, label="gradient")
        plt.legend()
        plt.show()
    
    w_star = softmax(result.x)
    obj_star, mu_star, A_star = optimize_solved_mu(mu, A, w_star, np.dot(A.T, w_star), inner_method)

    return obj_star, w_star


def adverserial_descent(mu, A, iters=10, method="SLSQP", verbose=True):
    # This method doesn't work
    n, k = A.shape
    
    mu_star, A_star = mu, A
    w_star = np.random.rand(n)

    objs = []
    for _ in tqdm(range(iters), disable=not verbose):
        obj_star, mu_star, A_star = optimize_solved_mu(mu, A, w_star, np.dot(A.T, w_star), method=method)
        objs.append(obj_star)

        w_star = optimal_w(mu, A, mu_star, A_star)
        
    obj_star, _, _ = optimize_solved_mu(mu, A, w_star, np.dot(A.T, w_star), method=method)
    
    return obj_star, w_star


###############################################################


def my_lowerbound_GLR(mu, A, N_A, N_Z, EPS=1e-6):
    n, k = A.shape
    means = np.dot(A, mu)
    i_star = np.argmax(means)    
    
    for s in range(n):
        if s == i_star:
            continue
        delta_s = means[i_star] - means[s]
        Ai_p = cp.Variable(k)
        As_p = cp.Variable(k)
        mu_p = cp.Variable(k)
        alpha = cp.Variable(k)
        
        d_Ai = N_A[i_star] * cp.sum(cp.rel_entr(A[i_star], Ai_p))
        d_As = N_A[s] * cp.sum(cp.rel_entr(A[s], As_p))
        d_mu = cp.sum(cp.multiply(N_Z, cp.square(mu - mu_p))) / 2  # assume gaussian
        
        constraints = [
            cp.sum(alpha) == delta_s,
            Ai_p >= EPS,
            cp.sum(Ai_p) == 1,
            As_p >= EPS,
            cp.sum(As_p) == 1,
            mu_p >= 0,
            mu_p <= 1,
        ]
        for j in range(k):
            # McCormick Relaxations
            constraints.append(alpha[j] >= -mu_p[j])
            constraints.append(alpha[j] <= mu_p[j])
            constraints.append(alpha[j] >= As_p[j] - Ai_p[j] + mu_p[j] - 1)
            constraints.append(alpha[j] <= As_p[j] - Ai_p[j] - mu_p[j] + 1)
        
        problem = cp.Problem(cp.Minimize(d_Ai + d_As + d_mu), constraints)
        
        try:
            problem.solve()
            if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                return problem.value
            else:
                print("non optimal value at glr")
                return None
        except:
            print("failed optimization at glr")
            return None
        

def lowerbound_GLR(mu, A, N_A, N_Z):
    c_m = 0.5  # assumed gaussian distribution
    n, k = A.shape
    
    means = np.dot(A, mu)
    sorted_arms = np.argsort(means)
    i_star = sorted_arms[-1]   
    s = sorted_arms[-2]
    
    delta_s = means[i_star] - means[s]
    denom = 1/(2 * N_A[s]) + 1/(2 * N_A[i_star]) + sum([1/(c_m * N_Z[i]) for i in range(k)])

    return delta_s**2 / denom


def lowerbound_optimize(mu, A):
    c_m = 0.5  # assumed gaussian distribution
    n, k = A.shape
    
    means = np.dot(A, mu)
    i_star = np.argmax(means)    
    s = np.argsort(means)[-2]
    
    w = cp.Variable(n)

    objective = cp.Minimize(
        cp.inv_pos(w[i_star]) / 2 + 
        cp.inv_pos(w[s]) / 2 + 
        cp.sum(cp.inv_pos(A.T @ w)) / c_m
    )
    constraints = [
        cp.sum(w) == 1,
        w >= 0
    ]
    problem = cp.Problem(objective, constraints)
    
    try:
        problem.solve()
        w_star = w.value
        if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            return lowerbound_GLR(mu, A, w_star, np.dot(A.T, w_star)), w_star
        else:
            print("non optimal value at glr")
            return None, None
    except:
        print("failed optimization at glr")
        return None, None


def optimize_GLR(mu, A, N_A, N_Z, alg="solved_mu"):
    ALGS = {
        "solved_mu": optimize_solved_mu,
    }
    optimization_alg: function = ALGS[alg]
    
    return optimization_alg(mu, A, N_A, N_Z)


def optimize(mu, A, alg="scipy", verbose=False):
    ALGS = {
        "adverserial": adverserial_descent,
        "scipy": optimize_scipy,
        "scipy_softmax": optimize_scipy_softmax,
        "scipy_grad": optimize_scipy_grad,
        "ternary": ternary_search,
        "grid": grid_search
    }
    optimization_alg: function = ALGS[alg]
    
    return optimization_alg(mu, A, verbose=verbose)


###############################################################


def create_testset_fixed_w(n, k, cnt):
    output_path = TESTSET_DIR + f"fixed_w_n={n}_k={k}.json"
    
    with open(output_path, "r") as F:
        testset = json.load(F)
    
    for _ in tqdm(range(cnt), desc="expanding dataset"):
        mu = np.random.rand(k)
        A = np.random.rand(n, k)
        A = (A.T / np.sum(A, axis=1)).T
        w = np.random.rand(n)
        w = w / np.sum(w)
        
        obj_star, mu_star, A_star = fixed_w_grid_search(mu, A, w, verbose=False)
        
        idx = len(testset) + 1
        testset.append({
            "index": idx,
            "mu": mu.tolist(),
            "A": A.tolist(),
            "w": w.tolist(),
            "obj_star": obj_star,
            "mu_star": mu_star.tolist(),
            "A_star": A_star.tolist()
        })
    
    with open(output_path, "w") as F:
        json.dump(testset, F, indent=4)
        
    return testset


def create_testset(n, k, cnt):
    output_path = TESTSET_DIR + f"opt_n={n}_k={k}.json"
    
    with open(output_path, "r") as F:
        testset = json.load(F)
    
    for _ in tqdm(range(cnt), desc="expanding dataset"):
        mu = np.random.rand(k)
        A = np.random.rand(n, k)
        A = (A.T / np.sum(A, axis=1)).T
        
        obj_star, w_star = ternary_search(mu, A, verbose=False)
        
        idx = len(testset) + 1
        testset.append({
            "index": idx,
            "mu": mu.tolist(),
            "A": A.tolist(),
            "w_star": w_star.tolist(),
            "obj_star": obj_star,
        })
    
    with open(output_path, "w") as F:
        json.dump(testset, F, indent=4)
        
    return testset


def prep_testest(testset_path):
    file_path = Path(testset_path)
    if not file_path.exists():
        with file_path.open('w') as F:
            json.dump([], F)
    
    with open(testset_path, 'r') as F:
        testset = json.load(F)
    
    return testset


def display_results_fixed_w(obj_star, mu_star, A_star, file=None):
    print("minimum achieved:", obj_star, file=file)
    print("optimal mu:", file=file)
    print(mu_star, file=file)
    print("optimal A:", file=file)
    print(A_star, file=file)

def display_results(obj_star, w_star, file=None):
    print("maximum achieved:", obj_star, file=file)
    print("optimal w:", file=file)
    print(w_star, file=file)


def test_method_fixed_w(n, k, name="grid", experiment_cnt=10, rep=1):
    ALGS = {
        "coordinate": coordinate_descent,
        "COBYQA": COBYQA_solved_mu,
        "SLSQP": SLSQP_solved_mu,
        "grid": fixed_w_grid_search,
        "contribution": contribution_optimization,
        "lowerbound": lowerbound_fixed_w,
        "my_lowerbound": my_lowerbound_fixed_w
    }
    testset_path = TESTSET_DIR + f"fixed_w_n={n}_k={k}.json"
    output_path=f"results/fixed_w_{name}.txt"
    if name not in ALGS.keys():
        print("Invalid alg name!")
        return
    optimization_alg: function = ALGS[name]
    
    testset = prep_testest(testset_path)
    if experiment_cnt > len(testset):
        testset = create_testset_fixed_w(n, k, experiment_cnt-len(testset))
    
    with open(output_path, 'w') as F:
        print(f"~~~~~~~~~~ testing on n={n}, k={k}, {experiment_cnt} times", file=F)
        suboptimal_gaps = {}
        suboptimal_ratios = {}
        fail_cnt = 0
        for iter in tqdm(range(experiment_cnt), "testing"):
            experiment = testset[iter]
            mu = np.array(experiment["mu"])
            A = np.array(experiment["A"])
            w = np.array(experiment["w"])
            obj_star = experiment["obj_star"]
            mu_star = np.array(experiment["mu_star"])
            A_star = np.array(experiment["A_star"])

            alg_obj, alg_mu, alg_A = np.inf, None, None
            for _ in range(rep):
                result = optimization_alg(mu, A, w, verbose=False)
                plt.axhline(y=obj_star, color="black", linestyle='--')
                plt.show()
                if result[0] < alg_obj:
                    alg_obj, alg_mu, alg_A = result

            suboptimal_gaps[iter] = alg_obj - obj_star
            suboptimal_ratios[iter] = alg_obj / obj_star
            if obj_star + TOL < alg_obj:
                fail_cnt += 1
                print(f"##### failed test {iter} with {alg_obj - obj_star} gap!", file=F)
            else:
                print(f"##### succeeded test {iter} with {alg_obj - obj_star} gap!", file=F)
            
            print(f"mu:\n{mu}\nA:\n{A}\nw:\n{w}", file=F)
            print("-" * 20, "ground truth", "-" * 20, file=F)
            display_results_fixed_w(obj_star, mu_star, A_star, file=F)
            print("-" * 20, name, "-" * 20, file=F)
            display_results_fixed_w(alg_obj, alg_mu, alg_A, file=F)
            print("#" * 60, file=F)
        
        print("#" * 20, "final success rate:", 1 - fail_cnt / experiment_cnt, file=F)
        print("#" * 20, "average suboptimality gap:", sum(suboptimal_gaps.values()) / experiment_cnt, file=F)
        print("#" * 20, "average suboptimality ratio:", sum(suboptimal_ratios.values()) / experiment_cnt, file=F)
        print("### suboptimality gaps:", file=F)
        print(suboptimal_gaps, file=F)
        print("### suboptimality ratios:", file=F)
        print(suboptimal_ratios, file=F)
        
        print("final success rate:", 1 - fail_cnt / experiment_cnt)
        print("average suboptimality gap:", sum(suboptimal_gaps.values()) / experiment_cnt)
        print("average suboptimality ratio:", sum(suboptimal_ratios.values()) / experiment_cnt)

    
    fig = px.box(x=suboptimal_gaps.values(), title=f"{name} suboptimality gaps")
    fig.show()


def test_method(n, k, name="", experiment_cnt=10, rep=1):
    testset_path = TESTSET_DIR + f"opt_n={n}_k={k}.json"
    output_path=f"results/opt_{name}.txt"
    
    ALGS = {
        "adverserial": adverserial_descent,
        "scipy": optimize_scipy,
        "scipy_softmax": optimize_scipy_softmax,
        "scipy_grad": optimize_scipy_grad,
        "ternary": ternary_search,
        "grid": grid_search
    }
    if name not in ALGS.keys():
        print("Invalid alg name!")
        return
    
    testset = prep_testest(testset_path)
    if experiment_cnt > len(testset):
        testset = create_testset(n, k, experiment_cnt-len(testset))
    
    with open(output_path, 'w') as F:
        print(f"~~~~~~~~~~ testing on n={n}, k={k}, {experiment_cnt} times", file=F)
        suboptimal_gaps = {}
        fail_cnt = 0
        for iter in tqdm(range(experiment_cnt), "testing"):
            experiment = testset[iter]
            mu = np.array(experiment["mu"])
            A = np.array(experiment["A"])
            w_star = np.array(experiment["w_star"])
            obj_star = experiment["obj_star"]
            
            alg_obj, alg_w = -np.inf, None
            for _ in range(rep):
                result = optimize(mu, A, alg=name, verbose=False)
                if result[0] > alg_obj:
                    alg_obj, alg_w = result

            suboptimal_gaps[iter] = obj_star - alg_obj
            if obj_star - TOL > alg_obj:
                fail_cnt += 1
                print(f"##### failed test {iter} with {obj_star - alg_obj} gap!", file=F)
            else:
                print(f"##### succeeded test {iter} with {obj_star - alg_obj} gap!", file=F)
            
            print(f"mu:\n{mu}\nA:\n{A}", file=F)
            print("-" * 20, "ground truth", "-" * 20, file=F)
            display_results(obj_star, w_star, file=F)
            print("-" * 20, name, "-" * 20, file=F)
            display_results(alg_obj, alg_w, file=F)
            print("#" * 60, file=F)
        
        print("#" * 20, "final success rate:", 1 - fail_cnt / experiment_cnt, file=F)
        print("#" * 20, "average suboptimality gap:", sum(suboptimal_gaps.values()) / experiment_cnt, file=F)
        print("### suboptimality gaps:", file=F)
        print(suboptimal_gaps, file=F)
        
        print("final success rate:", 1 - fail_cnt / experiment_cnt)
        print("average suboptimality gap:", sum(suboptimal_gaps.values()) / experiment_cnt)
    
    fig = px.box(x=suboptimal_gaps.values(), title=f"{name} suboptimality gaps")
    fig.show()


def optimize_instance(index, alg):
    instance_path = f"instances/instance{index}.json"
    _, _, _, mu, A, _, _ = read_instance_from_json(instance_path)
    
    ALGS = {
        "adverserial": adverserial_descent,
        "scipy": optimize_scipy,
        "ternary": ternary_search,
        "grid": grid_search
    }
    if alg not in ALGS.keys():
        print("Invalid alg name!")
        return
    
    obj_star, w_star = optimize(mu, A, alg, True)
    
    print(obj_star, w_star)
    

if __name__ == "__main__":
    args = get_optimization_arguments()
    if args.instance_index is None:
        n = int(input("set arm count:"))
        k = int(input("set context count:"))
        if args.fixed_w:
            test_method_fixed_w(n, k, experiment_cnt=args.experiment_cnt, name=args.name)
        else:
            test_method(n, k, experiment_cnt=args.experiment_cnt, name=args.name)
    else:
        optimize_instance(index=args.instance_index, name=args.name)
