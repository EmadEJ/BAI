from utils import *
import cvxpy as cp
from scipy.optimize import minimize_scalar
from algorithms.TS import TS
import itertools
from tqdm import tqdm

# known Mu Separator Track and Stop
class MuSTS(TS):
    def __init__(self, n, k, mu, confidence, tracking, mode = {'average_w': False, 'fast': False}):
        super().__init__(n, k, confidence, tracking, mode)
        self.mu = mu
        
        self.N_A = np.zeros(n)
        self.N_Z = np.zeros(k)
        self.cnt_post_actions = np.zeros((n, k))
        
        self.last_w = None
    
    def get_A_hat(self):
        return self.cnt_post_actions / self.N_A.reshape((self.n, 1))
    
    def get_means_hat(self):
        return self.get_A_hat() @ self.mu

    def best_empirical_arm(self):
        A_hat = self.get_A_hat()

        means_hat = np.dot(A_hat, self.mu)
        delta_hat = np.max(means_hat) - means_hat

        best_arm = np.argmax(means_hat)
        return best_arm, means_hat, delta_hat

    def get_T_star(self, mu, A):
        # TODO: implement
        T_star_inv, w_star = None, None
        return 1/T_star_inv, w_star

    def lambda_hat(self, w = None, A = None):
        N_A = self.N_A if w is None else w
        A_hat = self.get_A_hat() if A is None else A

        i_star = np.argmax(np.dot(A_hat, self.mu))
        
        obj_star = np.inf
        for s in range(self.n):
            if s == i_star:
                continue
            # Compute the weight for each arm
            Ai = cp.Variable(self.k)
            As = cp.Variable(self.k)

            objective = cp.Minimize(
                N_A[i_star] * cp.sum(cp.rel_entr(A_hat[i_star], Ai)) + 
                N_A[s] * cp.sum(cp.rel_entr(A_hat[s], As))
            )
            constraints = [
                Ai >= 0,
                cp.sum(Ai) == 1,
                As >= 0,
                cp.sum(As) == 1,
                (As - Ai) @ self.mu >= 0
            ]
            problem = cp.Problem(objective, constraints)
            problem.solve()
            
            if problem.status == "optimal":
                if problem.value <= obj_star:
                    obj_star = problem.value
                    A_p = A_hat.copy()
                    A_p[i_star] = Ai.value
                    A_p[s] = As.value
            else:
                self.optimization_failed_flag = True
                self.optimization_failed_number_of_rounds += 1

        if w is None:
            return obj_star
        
        return obj_star, A_p

    def beta_t_A(self, delta):
        return (
            np.log(1/delta) +
            (self.k-1) * sum([np.log(np.e * (1 + self.N_A[i] / (self.k-1))) for i in range(self.n)])
        )

    def stopping_rule(self):
        # returns True if need to stop and are confident enough
        lambda_hat_t = self.lambda_hat() 
        beta_t = self.beta_t_A(self.confidence)
        return lambda_hat_t > beta_t, lambda_hat_t, beta_t

    def optimal_w(self):
        if self.mode['fast'] and self.last_w is not None:
            if self.T > 100 and self.T % 10 != 0:
                return self.last_w
            if self.T > 1000 and self.T % 100 != 0:
                return self.last_w
            if self.T > 10000 and self.T % 1000 != 0:
                return self.last_w
            if self.T > 100000 and self.T % 10000 != 0:
                return self.last_w
            if self.T > 1000000 and self.T % 100000 != 0:
                return self.last_w
        
        A_hat = self.get_A_hat()
        i_star, _, _ = self.best_empirical_arm()
        
        # Our Method
        EPS = 0.01  # Tolerence in w error
        def inner_func(w, s):
            Ai = cp.Variable(self.k)
            As = cp.Variable(self.k)

            objective = cp.Minimize(
                w[0] * cp.sum(cp.rel_entr(A_hat[i_star], Ai)) + 
                w[1] * cp.sum(cp.rel_entr(A_hat[s], As))
            )
            constraints = [
                Ai >= 0,
                cp.sum(Ai) == 1,
                As >= 0,
                cp.sum(As) == 1,
                (As - Ai) @ self.mu >= 0
            ]
            problem = cp.Problem(objective, constraints)
            problem.solve()
            
            return problem.value
        
        def fixed_w_istar(w_istar, give_w=False):
            w = np.zeros(self.n)
            w[i_star] = w_istar
            
            funcs = [inner_func([w_istar, w[i]], i) for i in range(self.n)]
                
            while sum(w) < 1:
                s_star = None
                for s in range(self.n):
                    if s == i_star:
                        continue
                    if s_star is None or funcs[s] < funcs[s_star]:
                        s_star = s
            
                w[s_star] += EPS
            
                funcs[s_star] = inner_func([w_istar, w[s_star]], s_star)
           
            if give_w:
                return w
            
            return -np.sort(funcs)[1]
        
        res = minimize_scalar(
            fixed_w_istar,
            bounds=(0, 1),
            method="bounded",
            options={"xatol": EPS}
        )
        
        w_star = fixed_w_istar(res.x, give_w=True)
        w_star = w_star / np.sum(w_star)
        self.last_w = w_star
        
        return w_star
    
    def plot_w(self, mu, A, div=11, ax=None):
        if self.n != 3:
            print("plotting only available for n=3")
            return
        grid_range = np.linspace(0, 1, div)
        grid = itertools.product(grid_range, repeat=self.n-1)
        Ts = {}
        for w in grid:
            w0 = 1 - sum(w)
            if w0 < 0:
                continue
            w = np.array(([w0] + list(w)))
            
            obj = self.lambda_hat(w, A)[0]
            
            Ts[tuple(w)] = obj
            
        return draw_simplex_heatmap(Ts, ax)
    
    def get_action(self):
        # Initialization phase
        if np.any(self.cnt_post_actions == 0):  # explore all of A
            unexplored_a = np.where(self.cnt_post_actions == 0)
            return unexplored_a[0][0], True
        
        if self.tracking == 'C':
            return self.C_tracking()
        elif self.tracking == 'G': 
            return self.G_tracking()
        elif self.tracking == 'D':
            return self.D_tracking()
        elif self.tracking == 'E':
            return self.E_tracking()
        else:
            print("INVALID TRACKING")
            return -1, False

    def update(self, main_action, post_action, reward):
        self.T += 1
        self.cnt_post_actions[main_action][post_action] += 1
        self.N_A[main_action] += 1
        self.N_Z[post_action] += 1
        
        self.optimization_failed_flag = False
