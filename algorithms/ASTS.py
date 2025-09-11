from utils import *
import cvxpy as cp
from algorithms.TS import TS

# known A Seperator Track and Stop
class ASTS(TS):
    def __init__(self, n, k, A, confidence, tracking, mode = {'average_w': False}):
        super().__init__(n, k, confidence, tracking, mode)
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
        # Adapted directly from the old implementation of ASTS
        i_star, _, delta = self.best_empirical_arm()

        w = cp.Variable(self.n)  # Weights for convex combination
        w_p = self.A.T @ w  # w_p is a convex combination of rows of A
        t = cp.Variable()

        # Constraints
        constraints = [cp.sum(w) == 1, w >= 0]
        for i in range(self.n):
            if i == i_star:
                continue
            denom = cp.sum([(self.A[i, j] - self.A[i_star, j])**2 * cp.inv_pos(w_p[j]) for j in range(self.k)])
            constraints.append(
                t >= denom / (delta[i] ** 2)
            )

        objective = cp.Minimize(t)
        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve()
            if problem.status not in ["optimal", "optimal_inaccurate"]:
                self.optimization_failed_flag = True
            else:
                return w.value
        except Exception as e:
            self.optimization_failed_flag = True
            print("Optimization failed:", e)
            return None

        return np.ones(self.n) / self.n

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
