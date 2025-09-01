from utils import *
from scipy.optimize import linprog
from cvxpy.error import SolverError
from optimization import optimize, optimize_GLR, lowerbound_GLR

class STS:
    def __init__(self, n, k, confidence, mode = {'use_optimized_p': False, 'average_w': False, 'average_points_played': False}):
        self.n = n 
        self.k = k 
        self.T = 0 
        self.N_A = np.zeros(n)
        self.cnt_post_actions = np.zeros((n, k))
        self.N_Z = np.zeros(k)
        self.sum_of_rewards = np.zeros(k)
        self.confidence = confidence
        # self.exploration_vector = 1/n * np.ones(n)  # uniform random arm pull for exploration
        
        self.optimization_failed_flag = False
        self.optimization_failed_number_of_rounds = 0
        
        self.mode = mode
        
        # self.sum_points_played = np.zeros(k)
        
        self.sum_ws = np.zeros(n)


    def get_mu_hat(self):
        return self.sum_of_rewards / self.N_Z

    
    def get_A_hat(self):
        return self.cnt_post_actions / self.N_A.reshape((self.n, 1))
    

    def best_empirical_arm_calculator(self):
        mu_hat = self.get_mu_hat()
        A_hat =self.get_A_hat()

        actions_mu_hat = np.dot(A_hat, mu_hat)
        delta_hat = np.max(actions_mu_hat) - actions_mu_hat

        best_arm = np.argmax(actions_mu_hat)  
        return best_arm, actions_mu_hat, delta_hat


    def lambda_hat(self):
        mu_hat = self.get_mu_hat()
        A_hat = self.get_A_hat()
        obj_star, mu_star, A_star = optimize_GLR(mu_hat, A_hat, self.N_A, self.N_Z)
        return obj_star
    
    
    def lambda_lb(self):
        mu_hat = self.get_mu_hat()
        A_hat = self.get_A_hat()
        obj_star = lowerbound_GLR(mu_hat, A_hat, self.N_A, self.N_Z)
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
        # returns True if need to stop and are enough confident
        lambda_hat_t = self.lambda_hat() 
        beta_t = self.beta_t_mu(self.confidence / 2) + self.beta_t_mu(self.confidence / 2)
        return lambda_hat_t > beta_t, lambda_hat_t, beta_t


    def stopping_rule_lb(self):
        lambda_lb = self.lambda_lb()
        beta_t = self.beta_t_mu(self.confidence / 2) + self.beta_t_mu(self.confidence / 2)
        return lambda_lb > beta_t, lambda_lb, beta_t

    # def optimization_line_coefficient(self, w_t, v_k): 
    #     w_t = np.asarray(w_t)
    #     v_k = np.asarray(v_k)
    #     original_array = np.asarray(self.A)

    #     def target_vector(alpha):
    #         return (1 + alpha) * w_t - alpha * v_k

    #     c = np.zeros(self.n + 1)  # n lambdas + 1 alpha
    #     c[-1] = -1  # Coefficient for -alpha
    # def optimization_for_p(self, w):
    #     N_t = self.N_Z

    #     lambda_ = cp.Variable(self.n)
    #     p = self.A.T @ lambda_
        
    #     # ||N_t + p - w||^2 = ||p - (w - N_t)||^2
    #     y = w - N_t  
    #     objective = cp.Minimize(cp.norm((p + N_t)/(np.sum(N_t)+1) - w/np.sum(w), 2)**2)

    #     constraints = [
    #         lambda_ >= 0, 
    #         cp.sum(lambda_) == 1
    #     ]

    #     problem = cp.Problem(objective, constraints)
    #     problem.solve()

    #     if problem.status not in ["optimal", "optimal_inaccurate"]:
    #         raise ValueError(f"Solver failed: {problem.status}")

    #     lambda_opt = lambda_.value  
    #     p_opt = self.A.T @ lambda_opt 
        
    #     p_opt /= np.sum(p_opt)

    #     return p_opt, lambda_opt

    #     A_eq = np.zeros((self.k + 1, self.n + 1))
    #     A_eq[:self.k, :self.n] = original_array.T
    #     A_eq[:self.k, -1] = -(w_t - v_k) 
    #     A_eq[self.k, :self.n] = 1  
    #     A_eq[self.k, -1] = 0  # Alpha does not contribute to sum of lambdas

    #     b_eq = np.zeros(self.k + 1)
    #     b_eq[:self.k] = w_t
    #     b_eq[self.k] = 1 

    #     A_ub = np.zeros((self.n, self.n + 1))
    #     A_ub[:, :self.n] = -np.eye(self.n)  # -lambda_i <= 0
    #     b_ub = np.zeros(self.n)

    #     bounds = [(0, None)] * self.n + [(None, None)]
    #     result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    #     if result.success:
    #         return result.x[-1]  # alpha
    #     else:
    #         self.optimization_failed_flag = True
    #         # raise ValueError("Optimization failed: " + result.message)
    #         return None


    # def optimization_for_Z_optimal_vector(self):
    #     i_star, _, delta = self.best_empirical_arm_calculator()

    #     lambda_var = cp.Variable(self.n, nonneg=True)  # Weights for convex combination
    #     w = self.A.T @ lambda_var  # w is a convex combination of rows of A
    #     t = cp.Variable()

    #     constraints = [cp.sum(lambda_var) == 1]  # Convex combination constraint
        
    #     constraints += [w[j] >= 0 for j in range(self.k)]
        
    #     for i in range(self.n):
    #         if i == i_star:
    #             continue

    #         middle = cp.sum([(self.A[i, j] - self.A[i_star, j])**2 * cp.inv_pos(w[j]) for j in range(self.k)])
    #         constraints.append(
    #             t >= middle / (delta[i] ** 2)
    #         )

    #     objective = cp.Minimize(t)
    #     problem = cp.Problem(objective, constraints)
        
    #     try:
    #         problem.solve()
    #     except SolverError as e:
    #         self.optimization_failed_flag = True

    #     if problem.status not in ["optimal", "optimal_inaccurate"]:
    #         self.optimization_failed_flag = True
    #         #raise ValueError("Optimization problem did not converge.")
    #         return 0, 0

    #     # Return the optimal w and t
    #     w_opt = w.value
    #     w_opt /= np.sum(w_opt)
    #     t_opt = t.value
        
    #     return w_opt, t_opt


    # def optimization_for_p(self, w):
    #     N_t = self.N_Z

    #     lambda_ = cp.Variable(self.n)
    #     p = self.A.T @ lambda_
        
    #     # ||N_t + p - w||^2 = ||p - (w - N_t)||^2
    #     y = w - N_t  
    #     objective = cp.Minimize(cp.norm((p + N_t)/(np.sum(N_t)+1) - w/np.sum(w), 2)**2)

    #     constraints = [
    #         lambda_ >= 0, 
    #         cp.sum(lambda_) == 1
    #     ]

    #     problem = cp.Problem(objective, constraints)
    #     problem.solve()

    #     if problem.status not in ["optimal", "optimal_inaccurate"]:
    #         raise ValueError(f"Solver failed: {problem.status}")

    #     lambda_opt = lambda_.value  
    #     p_opt = self.A.T @ lambda_opt 
        
    #     p_opt /= np.sum(p_opt)

    #     return p_opt, lambda_opt


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
    

    def C_Tracking(self):
        # Initialization phase
        if np.any(self.cnt_post_actions == 0):  # explore all of A
            unexplored_pa = np.where(self.cnt_post_actions == 0)
            return unexplored_pa[0][0], True

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


    def D_Tracking(self):
        # Initialization phase
        if np.any(self.cnt_post_actions == 0):  # explore all of A
            unexplored_a = np.where(self.cnt_post_actions == 0)
            return unexplored_a[0][0], True
        
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


    def G_Tracking(self):
        # Initialization phase
        if np.any(self.cnt_post_actions == 0):  # explore all of A
            unexplored_a = np.where(self.cnt_post_actions == 0)
            return unexplored_a[0][0], True
        
        # Forced exploration
        if np.sqrt(self.T) % 1 == 0:
            # Selects an arm at random
            arm = np.random.randint(0, self.n)
            return arm, False
        
        # Direct tracking
        w = self.optimal_w()     
        w_projected = self.G_projection(w, self.N_A)
            
        self.sum_ws += w_projected/np.sum(w_projected)  # make sure w sums up to 1
        
        if self.mode["average_w"]:
            target = (self.sum_ws)/ np.sum(self.sum_ws) * np.sum(self.N_A)  # some optimization rounds may have failed so scale up to match it
        else:
            target = w_projected * np.sum(self.N_A)
        
        result = target - self.N_A
        return np.argmax(result), False
        

    def update(self, main_action, post_action, reward):
        self.T += 1
        self.cnt_post_actions[main_action][post_action] += 1
        self.N_A[main_action] += 1
        self.sum_of_rewards[post_action] += reward
        self.N_Z[post_action] += 1
        
        self.optimization_failed_flag = False
