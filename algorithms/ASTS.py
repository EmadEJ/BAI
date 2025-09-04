from utils import *
import cvxpy as cp

class ASTS:
    def __init__(self, n, k, A, confidence, tracking, mode = {'average_w': False}):
        self.n = n
        self.k = k
        self.A = A
        
        self.T = 0
        self.N_A = np.zeros(n)
        self.N_Z = np.zeros(k)
        self.sum_of_rewards = np.zeros(k)
        self.confidence = confidence
        
        self.optimization_failed_flag = False
        self.optimization_failed_number_of_rounds = 0
        
        self.tracking = tracking
        self.mode = mode
        
        self.sum_ws = np.zeros(n)

    def get_mu_hat(self):
        return self.sum_of_rewards / self.N_Z
    
    def get_means_hat(self):
        return self.A @ self.get_mu_hat()

    def best_empirical_arm_calculator(self):
        mu_hat = self.get_mu_hat()

        means_hat = np.dot(self.A, mu_hat)
        delta_hat = np.max(means_hat) - means_hat

        best_arm = np.argmax(means_hat)  
        return best_arm, means_hat, delta_hat

    def lambda_hat(self):
        best_arm, _, delta_hat = self.best_empirical_arm_calculator()

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
        # returns True if need to stop and are enough confident
        lambda_hat_t = self.lambda_hat() 
        beta_t = self.beta_t_mu(self.confidence)
        return lambda_hat_t > beta_t, lambda_hat_t, beta_t

    def optimal_w(self):
        mu_hat = self.get_mu_hat()
        A = self.A
        i_star, means_hat, delta_hat = self.best_empirical_arm_calculator()

        T_star, w_star = np.inf, None
        for s in range(self.n):
            if s == i_star:
                continue
            
            w = cp.Variable(self.n)
            
            coef = (A[:, i_star] - A[:, s])**2
            objective = cp.Minimize(cp.sum(cp.multiply(coef, cp.inv_pos(A @ w))))
            
            constraints = [w >= 0, cp.sum(w) == 1]
            
            problem = cp.Problem(objective, constraints)
            problem.solve()
            
            T_s = (delta_hat[s]**2) / (2 * problem.value)
            if T_s < T_star:
                T_star = T_s
                w_star = w.value

        return w_star

    def C_projection(self, w):
        eps = 0.5/np.sqrt(self.n**2 + self.T)

        v = cp.Variable(self.n)
        t = cp.Variable()

        objective = cp.Minimize(t)

        constraints = [
            v >= eps,   
            v <= 1,     
            cp.sum(v) == 1,    
            v - w <= t,     
            v - w >= -t 
        ]

        problem = cp.Problem(objective, constraints)

        try:
            problem.solve()
            if problem.status != cp.OPTIMAL:
                self.optimization_failed_flag = True
                return np.array([])
        except Exception as e:
            self.optimization_failed_flag = True
            return np.array([])

        return v.value
    
    def C_tracking(self):
        # C-tracking
        w = self.optimal_w()
        w_projected = self.C_projection(w)
            
        self.sum_ws += w_projected/np.sum(w_projected)  # make sure w sums up to 1
        
        if self.mode["average_w"]:
            target = (self.sum_ws)/ np.sum(self.sum_ws) * np.sum(self.N_A)  # some optimization rounds may have failed so scale up to match it
        else:
            target = w_projected * np.sum(self.N_A)
        
        result = target - self.N_A
        return np.argmax(result), False

    def D_tracking(self):
        # Forced exploration
        if np.any(self.N_A < np.sqrt(self.T) - self.n/2):
            unexplored_a = np.where(self.N_A < np.sqrt(self.T) - self.n/2)
            return unexplored_a[0][0], False
        
        # Direct tracking
        w = self.optimal_w()
            
        self.sum_ws += w/np.sum(w)  # make sure w sums up to 1
        
        if self.mode["average_w"]:
            target = (self.sum_ws)/ np.sum(self.sum_ws) * np.sum(self.N_A)
        else:
            target = w * np.sum(self.N_A)
        
        result = target - self.N_A
        return np.argmax(result), False

    def G_projection(self, w, v):
        if np.array_equal(w, v):  # already too close to w
            return w
        dir = w - v
        t = np.divide(dir, w, out=np.zeros_like(w), where=dir!=0)
        proj = w - dir / np.min(t)
        return proj

    def G_tracking(self):
        # Forced exploration
        if np.sqrt(self.T) % 1 == 0:
            # Selects an arm at random
            arm = np.random.randint(0, self.n)
            return arm, False
        
        # Direct tracking
        w = self.optimal_w()     
        w_projected = self.G_projection(w, self.N_A / sum(self.N_A))
                    
        self.sum_ws += w_projected/np.sum(w_projected)  # make sure w sums up to 1
        
        if self.mode["average_w"]:
            target = (self.sum_ws)/ np.sum(self.sum_ws) * np.sum(self.N_A)  # some optimization rounds may have failed so scale up to match it
        else:
            target = w_projected * np.sum(self.N_A)
        
        result = target - self.N_A
        return np.argmax(result), False
    
    def E_projection(self, w, v):
        if np.array_equal(w, v):  # already too close to w
            return w
        dir = w - v
        target = w + dir * self.T
        if np.any(target < 0):
            return self.G_projection(w, v)
        return target
    
    def E_tracking(self):
        # Forced exploration
        if np.sqrt(self.T) % 1 == 0:
            # Selects an arm at random
            arm = np.random.randint(0, self.n)
            return arm, False
        
        # Direct tracking
        w = self.optimal_w()
        w_projected = self.E_projection(w, self.N_A / sum(self.N_A))

        self.sum_ws += w_projected/np.sum(w_projected)  # make sure w sums up to 1
        
        if self.mode["average_w"]:
            target = (self.sum_ws)/ np.sum(self.sum_ws) * np.sum(self.N_A)  # some optimization rounds may have failed so scale up to match it
        else:
            target = w_projected * np.sum(self.N_A)
        
        result = target - self.N_A
        return np.argmax(result), False
    
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
