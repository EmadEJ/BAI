from utils import *
from optimization import optimize, optimize_GLR, lowerbound_GLR

class STS:
    def __init__(self, n, k, confidence, tracking, mode = {'average_w': False}):
        self.n = n 
        self.k = k 
        self.T = 0 
        self.N_A = np.zeros(n)
        self.N_Z = np.zeros(k)
        self.cnt_post_actions = np.zeros((n, k))
        self.sum_of_rewards = np.zeros(k)
        self.confidence = confidence
        
        self.optimization_failed_flag = False
        self.optimization_failed_number_of_rounds = 0
        
        self.tracking = tracking
        self.mode = mode
        
        self.sum_ws = np.zeros(n)

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

    def stopping_rule(self):
        # returns True if need to stop and are confident enough
        lambda_hat_t = self.lambda_hat() 
        beta_t = self.beta_t_A(self.confidence / 2) + self.beta_t_mu(self.confidence / 2)
        return lambda_hat_t > beta_t, lambda_hat_t, beta_t

    def stopping_rule_lb(self):
        lambda_lb = self.lambda_lb()
        beta_t = self.beta_t_A(self.confidence / 2) + self.beta_t_mu(self.confidence / 2)
        return lambda_lb > beta_t, lambda_lb, beta_t

    def optimal_w(self):
        mu_hat = self.get_mu_hat()
        A_hat = self.get_A_hat()
        
        T_star, w_star = optimize(mu_hat, A_hat)
        
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
