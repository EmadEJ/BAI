import numpy as np
import cvxpy as cp
from tqdm import tqdm
from tqdm.contrib import itertools
from itertools import product
from utils import *
from io_utils import *
import matplotlib.pyplot as plt

EPSILON = 1e-4  # used for checking whether the optimization was good enough

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


def grid_search(mu, A, w, div=11, EPS=1e-6, verbose=True):
    n, k = A.shape
    if n != 2 or k != 3:
        raise NotImplementedError
    
    N_A = w
    N_Z = np.dot(A.T, w)
    i_star, _ = best_arm(mu, A)
    
    obj_star = np.inf
    mu_star, A_star = None, None
    
    grid = np.linspace(EPS, 1, div)
    for mu1, mu2, mu3 in itertools.product(grid, grid, grid, disable=not verbose):
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

                obj = objective(mu, A, mu_p, A_p, N_A, N_Z)
                if obj < obj_star:
                    obj_star = obj
                    mu_star = mu_p
                    A_star = A_p
    
    return obj_star, mu_star, A_star


def fast_grid_search(mu, A, w, div=21, EPS=1e-6, verbose=True):
    n, k = A.shape
    if n != 2 or k != 3:
        raise NotImplementedError
    
    N_A = w
    N_Z = np.dot(A.T, w)
    i_star, _ = best_arm(mu, A)
    
    obj_star = np.inf
    mu_star, A_star = None, None
    
    grid = np.linspace(EPS, 1, div)
    for mu1, mu2, mu3 in itertools.product(grid, grid, grid, disable=not verbose):
        mu_p = np.array([mu1, mu2, mu3])
        for s in range(n):
            if s == i_star:
                continue
            A_p = optimal_A(mu, A, w, mu_p, s)

            obj = objective(mu, A, mu_p, A_p, N_A, N_Z)
            if obj < obj_star:
                obj_star = obj
                mu_star = mu_p
                A_star = A_p
    
    gap = 1/(div-1)
    for mu1, mu2, mu3 in itertools.product(
        np.linspace(mu_star[0] - gap, mu_star[0] + gap, 11), 
        np.linspace(mu_star[1] - gap, mu_star[1] + gap, 11), 
        np.linspace(mu_star[2] - gap, mu_star[2] + gap, 11), disable=not verbose):
        mu_p = np.array([mu1, mu2, mu3])
        for s in range(n):
            if s == i_star:
                continue
            A_p = optimal_A(mu, A, w, mu_p, s)

            obj = objective(mu, A, mu_p, A_p, N_A, N_Z)
            if obj < obj_star:
                obj_star = obj
                mu_star = mu_p
                A_star = A_p
    
    return obj_star, mu_star, A_star


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
    
    objective = cp.sum([cp.multiply(w[i], cp.sum(cp.rel_entr(A[i], A_p[i]))) for i in range(n)])
    
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


def coordinate_descent(mu, A, w, iters=10, verbose=True, lr=0.01):
    n, k = A.shape
    # A_p = np.random.rand(n, k)
    # mu_p = np.random.rand(k)
    A_p = A
    mu_p = mu
    
    i_star, _ = best_arm(mu, A)

    objectives = []
    
    obj_star = np.inf
    mu_star, A_star = None, None
    for s in tqdm(range(n), desc="coordinate_descent", disable=~verbose):
        if s == i_star:
            continue
        
        for _ in range(iters):
            mu_p = optimal_mu(mu, A, w, A_p, s)
            A_p = optimal_A(mu, A, w, mu_p, s)
            # print("objective:")
            # print(objective(mu, A, mu_p, A_p, w, np.dot(A.T, w)))
            # print("mu:")
            # print(mu_p)
            # print("A:")
            # print(A_p)
            objectives.append(objective(mu, A, mu_p, A_p, w, np.dot(A.T, w)))
        
        obj = objective(mu, A, mu_p, A_p, w, np.dot(A.T, w))
        if obj < obj_star:
            obj_star = obj
            mu_star = mu_p
            A_star = A_p
    
    # print(objectives)
    # plt.plot(range(iters), objectives)
    # plt.show()
    
    return obj_star, mu_star, A_star
    

def optimal_w(mu, A, method="grid_seach"):
    pass


def display_results(obj_star, mu_star, A_star, file=None):
    print("minimum achieved:", obj_star, file=file)
    print("optimal mu:", file=file)
    print(mu_star, file=file)
    print("optimal A:", file=file)
    print(A_star, file=file)


def create_testset(n, k, cnt, output_path="instances/opt_testset.json"):
    with open(output_path, "r") as F:
        testset = json.load(F)
    
    for _ in tqdm(range(cnt), desc="expanding dataset"):
        mu = np.random.rand(k)
        A = np.random.rand(n, k)
        A = (A.T / np.sum(A, axis=1)).T
        w = np.random.rand(n)
        w = w / np.sum(w)
        
        obj_star, mu_star, A_star = fast_grid_search(mu, A, w, verbose=False)
        
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


def test_method(n, k, experiment_cnt=10, testset_path="instances/opt_testset.json", name=""):
    output_path=f"results/opt_{name}.txt"
    
    with open(testset_path, 'r') as F:
        testset = json.load(F)
    
    if experiment_cnt > len(testset):
        testset = create_testset(n, k, experiment_cnt-len(testset))
    
    with open(output_path, 'w') as F:
        suboptimal_gaps = {}
        fail_cnt = 0
        for iter in tqdm(range(experiment_cnt)):
            experiment = testset[iter]
            mu = np.array(experiment["mu"])
            A = np.array(experiment["A"])
            w = np.array(experiment["w"])
            obj_star = experiment["obj_star"]
            mu_star = np.array(experiment["mu_star"])
            A_star = np.array(experiment["A_star"])

            
            descent_obj, descent_mu, descent_A = np.inf, None, None
            for _ in range(1):
                result = coordinate_descent(mu, A, w, verbose=False)
                if result[0] < descent_obj:
                    descent_obj, descent_mu, descent_A = result

            suboptimal_gaps[iter] = descent_obj - obj_star
            if obj_star + EPSILON < descent_obj:
                fail_cnt += 1
                print(f"##### failed test {iter} with {descent_obj - obj_star} gap!", file=F)
            else:
                print(f"##### succeeded test {iter} with {descent_obj - obj_star} gap!", file=F)
            
            print(f"mu:\n{mu}\nA:\n{A}\nw:\n{w}", file=F)
            print("### ground truth got:", file=F)
            display_results(obj_star, mu_star, A_star, file=F)
            print("### coordinate descent got:", file=F)
            display_results(descent_obj, descent_mu, descent_A, file=F)
        
        plt.hist(x=suboptimal_gaps.values(), bins=100)
        print("#" * 20, "final success rate:", 1 - fail_cnt / experiment_cnt, file=F)
        print("#" * 20, "average suboptimality gap:", sum(suboptimal_gaps.values()) / experiment_cnt, file=F)
        print("### suboptimality gaps:", file=F)
        print(suboptimal_gaps, file=F)
        
        print("final success rate:", 1 - fail_cnt / experiment_cnt)
        print("average suboptimality gap:", sum(suboptimal_gaps.values()) / experiment_cnt)
    
    plt.show()


def optimize(index):
    instance_path = f"instances/instance{index}.json"
    n, k, _, mu, A, _, _ = read_instance_from_json(instance_path)
    
    w = np.random.rand(n)
    w = w / np.sum(w)
    print("w:", w)
    
    print("#" * 30, "fast grid search results:")
    display_results(*fast_grid_search(mu, A, w))
    
    print("#" * 30, "coordinate_descent results:")
    display_results(*coordinate_descent(mu, A, w))


if __name__ == "__main__":
    args = get_optimization_arguments()
    if args.instance_index is None:
        test_method(2, 3, experiment_cnt=args.experiment_cnt, name=args.name)
    else:
        optimize(index=args.instance_index)
