from utils import *
import cvxpy as cp
from algorithms.TS import TS

class ASTS(TS):
    def __init__(self, n, k, A, confidence, tracking, mode = {'average_w': False}):
        super().__init__(n, k, A, confidence, tracking, mode)
        self.A = A
        
        self.N_A = np.zeros(n)
        self.N_Z = np.zeros(k)
        self.sum_of_rewards = np.zeros(k)

    def get_mu_hat(self):
        return self.sum_of_rewards / self.N_Z
    
    def get_means_hat(self):
        return self.A @ self.get_mu_hat()

    def best_empirical_arm(self):
        mu_hat = self.get_mu_hat()

        means_hat = np.dot(self.A, mu_hat)
        delta_hat = np.max(means_hat) - means_hat

        best_arm = np.argmax(means_hat)  
        return best_arm, means_hat, delta_hat

    def lambda_hat(self):
        best_arm, _, delta_hat = self.best_empirical_arm()

        denom = np.sum((self.A - self.A[best_arm])**2 / self.N_Z, axis=1) 
        result = (delta_hat**2) / (2 * denom)

        #since we have NaN in the i*-th element and that should not be computed in min
        return np.nanmin(result)

    def beta_t_mu(self, delta):
        return (
            self.k * Cg(np.log(1/delta) / self.k) +
            3 * sum([np.log(1 + np.log(self.N_Z[j])) for j in range(self.k)])
        )

    def stopping_rule(self):
        # returns True if need to stop and are confident enough
        lambda_hat_t = self.lambda_hat() 
        beta_t = self.beta_t_mu(self.confidence)
        return lambda_hat_t > beta_t, lambda_hat_t, beta_t

    def optimal_w(self):
        A = self.A
        i_star, _, delta_hat = self.best_empirical_arm()

        T_star, w_star = np.inf, None
        for s in range(self.n):
            if s == i_star:
                continue
            
            w = cp.Variable(self.n)
            
            coef = (A[i_star] - A[s])**2
            objective = cp.Minimize(cp.sum(cp.multiply(coef, cp.inv_pos(w @ A))))
            
            constraints = [w >= 0, cp.sum(w) == 1]
            
            problem = cp.Problem(objective, constraints)
            problem.solve()
            
            T_s = (delta_hat[s]**2) / (2 * problem.value)
            if T_s < T_star:
                T_star = T_s
                w_star = w.value

        return w_star
    
    def get_action(self):
        # Initialization phase
        if np.any(self.N_Z == 0):  # explore all contexts at least once
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
        self.sum_of_rewards[post_action] += reward
        self.N_Z[post_action] += 1
        
        self.optimization_failed_flag = False
