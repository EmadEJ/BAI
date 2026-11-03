from utils import *
import cvxpy as cp
from algorithms.TS import TS
import itertools

# Sub-Gaussian Seperator Track and Stop
class SGTS(TS):
    def __init__(self, n, k, confidence, tracking, mode = {'average_w': False, 'fast': True}):
        # Note that k is not used throughout the algorithm because we are blind to post-actions
        super().__init__(n, k, confidence, tracking, mode)
        
        self.N_A = np.zeros(n)
        self.sum_of_rewards = np.zeros(n)
        
        self.var = 5/4  # sub-Gaussian parameter

    def get_means_hat(self):
        return self.sum_of_rewards / self.N_A

    def best_empirical_arm(self):
        means_hat = self.get_means_hat()
        delta_hat = np.max(means_hat) - means_hat

        best_arm = np.argmax(means_hat)  
        return best_arm, means_hat, delta_hat

    def get_T_star(self, mu, A):
        # TODO: implement
        T_star_inv, w_star = None, None
        return 1/T_star_inv, w_star

    def lambda_hat(self, w=None, means=None):
        if w is None:
            i_star, _, delta_hat = self.best_empirical_arm()
            N_A = self.N_A
        else:
            i_star = np.argmax(means)
            delta_hat = means - means[i_star]
            N_A = w
        
        lambda_star = np.inf
        for s in range(self.n):
            if s == i_star:
                continue
            glr = delta_hat[s]**2 / (2 * self.var * (1/N_A[i_star] + 1/N_A[s]))
            lambda_star = min(lambda_star, glr)
        
        return lambda_star

    def beta_t(self, delta):
        # Lemma 3.2 of [TD25a], eq. (41) of the paper
        return (self.n / 2) * W_bar(
            2 * np.log(np.e * np.pi**2 / 6) +
            (2/self.n) * sum([2 * np.log(1 + np.log(self.N_A[i])) for i in range(self.n)]) +
            (2/self.n) * np.log(1/delta)
        )

    def stopping_rule(self):
        # returns True if need to stop and are confident enough
        lambda_hat_t = self.lambda_hat() 
        beta_t = self.beta_t(self.confidence)
        return lambda_hat_t > beta_t, lambda_hat_t, beta_t

    def optimal_w(self):
        if self.skip_optimization():
            return self.last_w
        self.n_optimizations += 1
        
        i_star, _, delta_hat = self.best_empirical_arm()

        w = cp.Variable(self.n)
        t = cp.Variable()
        
        objective = cp.Maximize(t)
        constraints = [
            w >= 0,
            cp.sum(w) == 1
        ]
        for s in range(self.n):
            if s == i_star:
                continue
            constraints.append(
                t <= (delta_hat[s]**2) * (0.5 * cp.harmonic_mean(cp.hstack([w[i_star], w[s]]))) / (2 * self.var)
            )
        
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve()
            if problem.status != cp.OPTIMAL:
                self.optimization_failed_flag = True
                return np.ones(self.n) / self.n
            self.last_w = w.value
            return w.value
        except Exception as e:
            self.optimization_failed_flag = True
            print("Optimization failed:", e)

        return np.ones(self.n) / self.n

    def F_on_simplex(self, mu, A, div=101):
        # F(w) of the sub-Gaussian bandit that ignores the contexts, on a grid of the simplex (n = 3)
        grid_range = np.linspace(0, 1, div)
        grid = itertools.product(grid_range, repeat=self.n-1)
        Ts = {}
        for w in grid:
            w0 = 1 - sum(w)
            if w0 < -1e-9:
                continue
            w = np.array(([max(w0, 0)] + list(w)))
            
            with np.errstate(divide='ignore'):
                obj = self.lambda_hat(w, means=np.dot(A, mu))
            
            Ts[tuple(w)] = obj
        
        return Ts

    def get_action(self):
        # Initialization phase
        if np.any(self.N_A == 0):  # explore all contexts at least once
            return np.random.randint(0, self.n), True

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
        self.N_A[main_action] += 1
        self.sum_of_rewards[main_action] += reward
        
        self.optimization_failed_flag = False
