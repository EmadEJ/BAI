import numpy as np
import cvxpy as cp
from tqdm import tqdm
from tqdm.contrib import itertools
from itertools import product
from utils import *
from io_utils import *
from scipy.optimize import minimize, Bounds, LinearConstraint
import plotly.express as px

TOL = 1e-4  # used for checking whether the optimization was good enough
INF = 1e-9

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


def fixed_w_grid_search(mu, A, w, div=21, EPS=1e-6, solver=cp.CLARABEL, verbose=True):
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
            A_p = optimal_A(mu, A, w, mu_p, s, solver=solver)

            obj = objective(mu, A, mu_p, A_p, N_A, N_Z)
            if obj < obj_star:
                obj_star = obj
                mu_star = mu_p
                A_star = A_p
    
    return obj_star, mu_star, A_star


def grid_search(mu, A, EPS=1e-2, solver=cp.CLARABEL, verbose=True):
    # Used ternary search for now to fine the optimal value
    n, k = A.shape
    if n != 2 or k != 3:
        raise NotImplementedError
        
    l, r = 0.0, 1.0
    
    with tqdm(total=1, disable=not verbose) as pbar:
        while l + EPS < r:
            w1 = (5*l + 4*r) / 9
            w_p1 = np.array([w1, 1-w1])
            w2 = (4*l + 5*r) / 9
            w_p2 = np.array([w2, 1-w2])
            obj1, _, _ = fixed_w_grid_search(mu, A, w_p1, div=11, verbose=False)
            obj2, _, _ = fixed_w_grid_search(mu, A, w_p2, div=11, verbose=False)
            if obj1 > obj2:
                pbar.update(r - w2)
                r = w2
            else:
                pbar.update(w1 - l)
                l = w1
            
    w = (l + r) / 2
    w_star = np.array([w, 1-w])
    obj_star, _, _ = fixed_w_grid_search(mu, A, w_star, solver=solver, verbose=False)
    
    return obj_star, w_star


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
    

def optimal_A(mu, A, w, mu_p, s, solver=None):
    n, k = A.shape
    i_star, _ = best_arm(mu, A)
    
    A_p = cp.Variable((n, k))
    
    constraints = [(A_p[s] - A_p[i_star]) @ mu_p >= 0]
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


def coordinate_descent(mu, A, w, iters=10, verbose=True):
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


def optimize_solved_mu(mu, A, N_A, N_Z, method=None, verbose=True):
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
            
        delta = np.dot(A_p[i_star], mu) - np.dot(A_p[s], mu) 
        result += (delta**2 / (2*denom) if denom != 0 else np.inf)

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

        result = minimize(
            solved_mu_objective, 
            x0=np.reshape(A, (n*k)).tolist(), 
            args=(s), 
            bounds=bounds, 
            constraints=constraints,
            method=method
        )
        A_p = np.array(result.x).reshape((n, k))
        mu_p = optimal_mu(mu, A, N_A, A_p, s)
        obj = objective(mu, A, mu_p, A_p, N_A, N_Z)
        
        if np.abs(result.fun - obj) > 1e-6:
            print("something fishy going on")
            print(obj, result.fun)
        
        if obj < obj_star:
            obj_star = obj
            mu_star = mu_p
            A_star = A_p

    return obj_star, mu_star, A_star

def COBYQA_solved_mu(mu, A, w, verbose=True):
    return optimize_solved_mu(mu, A, w, np.dot(A.T, w), "COBYQA", verbose)

def SLSQP_solved_mu(mu, A, w, verbose=True):
    return optimize_solved_mu(mu, A, w, np.dot(A.T, w), "SLSQP", verbose)


def optimize_scipy(mu, A, method="SLSQP", inner_method="SLSQP", verbose=True):
    n, k = A.shape
    
    def neg_optimize_fixed_w(w, mu, A, method):
        return -optimize_solved_mu(mu, A, w, np.dot(A.T, w), method)[0]

    bounds = Bounds([0 for _ in range(n)], [1.0 for _ in range(n)])
    constraints = LinearConstraint([[1.0 for _ in range(n)]], [1.0], [1.0])
    
    result = minimize(
            neg_optimize_fixed_w, 
            x0=np.random.rand(n), 
            args=(mu, A, inner_method), 
            bounds=bounds, 
            constraints=constraints,
            method=method,
            options={'disp': verbose}
        )
    
    return -result.fun, result.x


def adverserial_descent(mu, A, iters=10, method="SLSQP", verbose=True):
    # This method doesn't work
    n, k = A.shape
    
    mu_star, A_star = mu, A
    w_star = np.random.rand(n)

    objs = []
    for _ in tqdm(range(iters), disable=not verbose):
        obj_star, mu_star, A_star = optimize_solved_mu(mu, A, w_star, method=method)
        objs.append(obj_star)

        w_star = optimal_w(mu, A, mu_star, A_star)
        
    obj_star, _, _ = optimize_solved_mu(mu, A, w_star, method=method)
    
    return obj_star, w_star


def optimize_GLR(mu, A, N_A, N_Z, alg="solved_mu"):
    ALGS = {
        "solved_mu": optimize_solved_mu,
    }
    optimization_alg: function = ALGS[alg]
    
    return optimization_alg(mu, A, N_A, N_Z)


def optimize(mu, A, alg="scipy"):
    ALGS = {
        "adverserial": adverserial_descent,
        "scipy": optimize_scipy,
        "grid": grid_search
    }
    optimization_alg: function = ALGS[alg]
    
    return optimization_alg(mu, A, verbose=False)


def create_testset_fixed_w(n, k, cnt, output_path="instances/fixed_w_testset.json"):
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


def create_testset(n, k, cnt, output_path="instances/opt_testset.json"):
    with open(output_path, "r") as F:
        testset = json.load(F)
    
    for _ in tqdm(range(cnt), desc="expanding dataset"):
        mu = np.random.rand(k)
        A = np.random.rand(n, k)
        A = (A.T / np.sum(A, axis=1)).T
        
        obj_star, w_star = grid_search(mu, A, verbose=False)
        
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


def test_method_fixed_w(n, k, name="grid", experiment_cnt=10, rep=1, testset_path="instances/fixed_w_testset.json"):
    ALGS = {
        "coordinate": coordinate_descent,
        "solved_mu_COBYQA": COBYQA_solved_mu,
        "solved_mu_SLSQP": SLSQP_solved_mu,
        "grid": fixed_w_grid_search
    }
    output_path=f"results/fixed_w_{name}.txt"
    optimization_alg: function = ALGS[name]
    
    with open(testset_path, 'r') as F:
        testset = json.load(F)
    
    if experiment_cnt > len(testset):
        testset = create_testset_fixed_w(n, k, experiment_cnt-len(testset))
    
    with open(output_path, 'w') as F:
        suboptimal_gaps = {}
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
                if result[0] < alg_obj:
                    alg_obj, alg_mu, alg_A = result

            suboptimal_gaps[iter] = alg_obj - obj_star
            if obj_star + TOL < alg_obj:
                fail_cnt += 1
                print(f"##### failed test {iter} with {alg_obj - obj_star} gap!", file=F)
            else:
                print(f"##### succeeded test {iter} with {alg_obj - obj_star} gap!", file=F)
            
            print(f"mu:\n{mu}\nA:\n{A}\nw:\n{w}", file=F)
            print("#" * 20, "ground truth", "#" * 20, file=F)
            display_results_fixed_w(obj_star, mu_star, A_star, file=F)
            print("#" * 20, name, "#" * 20, file=F)
            display_results_fixed_w(alg_obj, alg_mu, alg_A, file=F)
            print("-" * 60, file=F)
        
        print("#" * 20, "final success rate:", 1 - fail_cnt / experiment_cnt, file=F)
        print("#" * 20, "average suboptimality gap:", sum(suboptimal_gaps.values()) / experiment_cnt, file=F)
        print("### suboptimality gaps:", file=F)
        print(suboptimal_gaps, file=F)
        
        print("final success rate:", 1 - fail_cnt / experiment_cnt)
        print("average suboptimality gap:", sum(suboptimal_gaps.values()) / experiment_cnt)
    
    fig = px.box(x=suboptimal_gaps.values(), title=f"{name} suboptimality gaps")
    fig.show()


def test_method(n, k, name="", experiment_cnt=10, rep=1, testset_path="instances/opt_testset.json"):
    output_path=f"results/opt_{name}.txt"
    
    with open(testset_path, 'r') as F:
        testset = json.load(F)
    
    if experiment_cnt > len(testset):
        testset = create_testset(n, k, experiment_cnt-len(testset))
    
    with open(output_path, 'w') as F:
        suboptimal_gaps = {}
        fail_cnt = 0
        for iter in tqdm(range(experiment_cnt), "testing"):
            experiment = testset[iter]
            mu = np.array(experiment["mu"])
            A = np.array(experiment["A"])
            w_star = np.array(experiment["w_star"])
            obj_star = experiment["obj_star"]
            
            alg_obj, alg_w = np.inf, None
            for _ in range(rep):
                result = optimize(mu, A, alg=name, verbose=False)
                if result[0] < alg_obj:
                    alg_obj, alg_w = result

            suboptimal_gaps[iter] = obj_star - alg_obj
            if obj_star - TOL > alg_obj:
                fail_cnt += 1
                print(f"##### failed test {iter} with {obj_star - alg_obj} gap!", file=F)
            else:
                print(f"##### succeeded test {iter} with {obj_star - alg_obj} gap!", file=F)
            
            print(f"mu:\n{mu}\nA:\n{A}", file=F)
            print("#" * 20, "ground truth", "#" * 20, file=F)
            display_results(obj_star, w_star, file=F)
            print("#" * 20, name, "#" * 20, file=F)
            display_results(alg_obj, alg_w, file=F)
            print("-" * 60, file=F)
        
        print("#" * 20, "final success rate:", 1 - fail_cnt / experiment_cnt, file=F)
        print("#" * 20, "average suboptimality gap:", sum(suboptimal_gaps.values()) / experiment_cnt, file=F)
        print("### suboptimality gaps:", file=F)
        print(suboptimal_gaps, file=F)
        
        print("final success rate:", 1 - fail_cnt / experiment_cnt)
        print("average suboptimality gap:", sum(suboptimal_gaps.values()) / experiment_cnt)
    
    fig = px.box(x=suboptimal_gaps.values(), title=f"{name} suboptimality gaps")
    fig.show()


def optimize_instance(index):
    instance_path = f"instances/instance{index}.json"
    _, _, _, mu, A, _, _ = read_instance_from_json(instance_path)
    
    obj_star, w_star = grid_search(mu, A)
    # obj_star, w_star = adverserial_descent(mu, A)
    # obj_star, w_star = optimize_scipy(mu, A)
    
    print(obj_star, w_star)
    

if __name__ == "__main__":
    args = get_optimization_arguments()
    if args.instance_index is None:
        if args.fixed_w:
            test_method_fixed_w(2, 3, experiment_cnt=args.experiment_cnt, name=args.name)
        else:
            test_method(2, 3, experiment_cnt=args.experiment_cnt, name=args.name)
    else:
        optimize_instance(index=args.instance_index)
