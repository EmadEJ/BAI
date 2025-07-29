import numpy as np
import cvxpy as cp
from tqdm import tqdm
from tqdm.contrib import itertools
from itertools import product
from utils import *
from io_utils import *


def best_arm(mu, A):
    means = np.dot(A, mu)
    best_arm = np.argmax(means)  
    return best_arm, means[best_arm]


def deviation(mu, A, mu_p, A_p, N_A, N_Z):
    n, k = A.shape
    
    result = 0
    for i in range(n):
        result += N_A[i] * categorical_kl(A[i], A_p[i])
    for j in range(k):
        result += N_Z[j] * gaussian_kl(mu[j], mu_p[j])
    
    return result


def grid_search(mu, A, w, div=11, EPS=1e-6, verbose=True):
    n, k = A.shape
    if n != 2 or k != 3:
        raise NotImplementedError
    
    N_A = w
    N_Z = np.dot(A.T, w)
    i_star, _ = best_arm(mu, A)
    
    dev_star = np.inf
    mu_star, A_star = None, None
    
    grid = np.linspace(EPS, 1, div)
    for mu1, mu2, mu3 in itertools.product(grid, grid, grid, disable=~verbose):
        for a11, a12 in product(grid, grid):
            if a11 + a12 > 1:
                continue
            for a21, a22 in product(grid, grid):
                if a21 + a22 > 1:
                    continue
                
                mu_p = np.array([mu1, mu2, mu3])
                A_p = np.array([
                    [a11, a12, 1-a11-a12],
                    [a21, a22, 1-a21-a22],
                ])

                i_star_p, _ = best_arm(mu_p, A_p)
                if i_star == i_star_p:
                    continue

                dev = deviation(mu, A, mu_p, A_p, N_A, N_Z)
                if dev < dev_star:
                    dev_star = dev
                    mu_star = mu_p
                    A_star = A_p
    
    return dev_star, mu_star, A_star


def optimal_mu(mu, A, w, A_p, s):
    n, k = A.shape
    i_star, _ = best_arm(mu, A)
    
    mu_p = cp.Variable(k)
    
    constraint = [(A_p[s] - A_p[i_star]) @ mu_p >= 0]
    
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
    

def optimal_A(mu, A, w, mu_p, s):
    n, k = A.shape
    i_star, _ = best_arm(mu, A)
    
    A_p = cp.Variable((n, k))
    
    constraints = [(A_p[s] - A_p[i_star]) @ mu_p >= 0]
    for i in range(n):
        constraints += [cp.sum(A_p[i]) == 1]
    
    objective = cp.sum([cp.multiply(w[i], cp.sum(cp.kl_div(A[i], A_p[i]))) for i in range(n)])
    
    problem = cp.Problem(cp.Minimize(objective), constraints)
    
    try:
        problem.solve()
        if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            return A_p.value
        else:
            print("non optimal value at A")
            return None
    except:
        print("failed optimization at A")
        return None


def coordinate_descent(mu, A, w, iters=20, verbose=True):
    n, k = A.shape
    A_p = np.random.rand(n, k)
    mu_p = np.random.rand(k)
    
    i_star, _ = best_arm(mu, A)
    
    dev_star = np.inf
    mu_star, A_star = None, None
    for s in tqdm(range(n), desc="coordinate_descent", disable=~verbose):
        if s == i_star:
            continue
        
        for _ in range(iters):
            mu_p = optimal_mu(mu, A, w, A_p, s)
            A_p = optimal_A(mu, A, w, mu_p, s)
        
        dev = deviation(mu, A, mu_p, A_p, w, np.dot(A.T, w))
        if dev < dev_star:
            dev_star = dev
            mu_star = mu_p
            A_star = A_p
    
    return dev_star, mu_star, A_star
    

def optimal_w(mu, A, method="grid_seach"):
    pass


def display_results(dev_star, mu_star, A_star):
    print("minimum achieved:")
    print(dev_star)
    print("optimal mu:")
    print(mu_star)
    print("optimal A:")
    print(A_star)


def compare_methods(n, k, num_iters=100):
    fail_cnt = 0
    for iter in tqdm(range(num_iters)):
        mu = np.random.rand(k)
        A = np.random.rand(n, k)
        A = (A.T / np.sum(A, axis=1)).T
        w = np.random.rand(n)

        grid_dev, grid_mu, grid_A = grid_search(mu, A, w, verbose=False)

        descent_dev, descent_mu, descent_A = coordinate_descent(mu, A, w, verbose=False)

        if grid_dev < descent_dev:
            fail_cnt += 1
            print(f"##### failed on:\nmu:\n{mu}\nA:\n{A}\nw:\n{w}")
            print("### grid search got:")
            display_results(grid_dev, grid_mu, grid_A)
            print("### coordinate descent got:")
            display_results(descent_dev, descent_mu, descent_A)
        else:
            print("success!")

    print("#" * 20, "final success rate:", 1 - fail_cnt / num_iters)


def optimize():
    index = get_optimization_arguments().instance_index
    
    instance_path = f"instances/instance{index}.json"
    n, k, _, mu, A, _, _ = read_instance_from_json(instance_path)
    
    w = np.random.rand(n)
    print("w:", w)
    
    print("#" * 30, "grid search results:")
    display_results(*grid_search(mu, A, w))
    
    print("#" * 30, "coordinate_descent results:")
    display_results(*coordinate_descent(mu, A, w))


if __name__ == "__main__":
    # optimize()
    compare_methods(2, 3, 200)
