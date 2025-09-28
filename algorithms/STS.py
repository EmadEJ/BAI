from utils import *
from optimization import optimize, optimize_GLR, lowerbound_GLR
from algorithms.TS import TS
import itertools

# general Separator Track and Stop
class STS(TS):
    def __init__(self, n, k, confidence, tracking, mode = {'average_w': False}):
        super().__init__(n, k, confidence, tracking, mode)
        
        self.N_A = np.zeros(n)
        self.N_Z = np.zeros(k)
        self.cnt_post_actions = np.zeros((n, k))
        self.sum_of_rewards = np.zeros(k)
        
        # This is used for faster convergence
        self.last_w = None

    def get_mu_hat(self):
        return self.sum_of_rewards / self.N_Z
    
    def get_A_hat(self):
        return self.cnt_post_actions / self.N_A.reshape((self.n, 1))

    def get_means_hat(self):
        return self.get_A_hat() @ self.get_mu_hat()

    def best_empirical_arm(self):
        mu_hat = self.get_mu_hat()
        A_hat = self.get_A_hat()

        means_hat = np.dot(A_hat, mu_hat)
        delta_hat = np.max(means_hat) - means_hat

        best_arm = np.argmax(means_hat)  
        return best_arm, means_hat, delta_hat

    def get_T_star(self, mu, A):
        T_star_inv, w_star = optimize(mu, A)
        return 1/T_star_inv, w_star

    def plot_w(self, mu, A, div=21):
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
            
            obj = optimize_GLR(mu, A, w, np.dot(A.T, w), verbose=False)[0]
            
            Ts[tuple(w)] = obj
            
        draw_simplex_heatmap(Ts)

    def lambda_true(self):
        mu_hat = self.get_mu_hat()
        A_hat = self.get_A_hat()
        obj_star = optimize_GLR(mu_hat, A_hat, self.N_A, self.N_Z, alg="grid")[0]
        return obj_star
    
    def lambda_hat(self):
        mu_hat = self.get_mu_hat()
        A_hat = self.get_A_hat()
        obj_star = optimize_GLR(mu_hat, A_hat, self.N_A, self.N_Z)[0]
        return obj_star
    
    def lambda_lb(self):
        mu_hat = self.get_mu_hat()
        A_hat = self.get_A_hat()
        obj_star = lowerbound_GLR(mu_hat, A_hat, self.N_A, self.N_Z)[0]
        return obj_star

    def beta_t_mu(self, delta):
        return (
            self.k * Cg(np.log(1/delta) / self.k) +
            3 * sum([np.log(1 + np.log(self.N_Z[j])) for j in range(self.k)])
        )
    
    def beta_t_A(self, delta):
        return (
            np.log(1/delta) +
            (self.k-1) * sum([np.log(np.e * (1 + self.N_A[i] / (self.k-1))) for i in range(self.n)])
        )

    def beta_t(self, delta):
        # This threshold is derived from the combined martingale of categorical and exponential
        return (
            self.k * Cg(np.log(1/delta) / self.k) +
            3 * sum([np.log(1 + np.log(self.N_Z[j])) for j in range(self.k)]) +
            (self.k-1) * sum([np.log(np.e * (1 + self.N_A[i] / (self.k-1))) for i in range(self.n)])
        )

    def stopping_rule(self):
        # returns True if need to stop and are confident enough
        lambda_hat_t = self.lambda_hat() 
        beta_t = self.beta_t(self.confidence)
        return lambda_hat_t > beta_t, lambda_hat_t, beta_t

    def stopping_rule2(self):
        # returns True if need to stop and are confident enough
        lambda_hat_t = self.lambda_hat() 
        beta_t = self.beta_t_A(self.confidence / 2) + self.beta_t_mu(self.confidence / 2)
        return lambda_hat_t > beta_t, lambda_hat_t, beta_t

    def stopping_rule_lb(self):
        lambda_lb = self.lambda_lb()
        beta_t = self.beta_t(self.confidence)
        return lambda_lb > beta_t, lambda_lb, beta_t

    def optimal_w(self):
        mu_hat = self.get_mu_hat()
        A_hat = self.get_A_hat()
        
        T_star, w_star = optimize(mu_hat, A_hat, w0=self.last_w)
        self.last_w = w_star
        
        return w_star

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
        self.sum_of_rewards[post_action] += reward
        self.N_Z[post_action] += 1
        
        self.optimization_failed_flag = False
