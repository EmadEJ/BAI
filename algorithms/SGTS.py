from utils import *
import cvxpy as cp
from algorithms.TS import TS

# Sub-Gaussian Seperator Track and Stop
class SGTS(TS):
    def __init__(self, n, k, confidence, tracking, mode = {'average_w': False}):
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

    def lambda_hat(self):
        i_star, _, delta_hat = self.best_empirical_arm()
        
        lambda_star = np.inf
        for s in range(self.n):
            if s == i_star:
                continue
            glr = delta_hat[s]**2 / (2 * self.var * (1/self.N_A[i_star] + 1/self.N_A[s]))
            lambda_star = min(lambda_star, glr)
        
        return lambda_star

    def beta_t(self, delta):
        return 0.5 * self.n * Cg(
            (2/self.n) * np.log(1/delta) +
            4 * np.log(np.log(np.e * self.T / self.n)) +
            2 * np.log(np.e * np.pi**2 / 6)
        )

    def stopping_rule(self):
        # returns True if need to stop and are confident enough
        lambda_hat_t = self.lambda_hat() 
        beta_t = self.beta_t(self.confidence)
        return lambda_hat_t > beta_t, lambda_hat_t, beta_t

    def optimal_w(self):
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
            return w.value
        except Exception as e:
            self.optimization_failed_flag = True
            print("Optimization failed:", e)

        return np.ones(self.n) / self.n

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
