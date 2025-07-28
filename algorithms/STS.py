from utils import *
from scipy.optimize import linprog
from scipy.optimize import bisect
from scipy.optimize import root_scalar
from cvxpy.error import SolverError


class STS:
    def __init__(self, n, k, delta, contexts=None, mode = {'use_optimized_p': False, 'average_w': False, 'average_points_played': False}):
        self.contexts = contexts
        self.n = n 
        self.k = k 
        self.T = 0 
        self.N_times_seen = np.zeros(n)
        self.K_times_seen = np.zeros(k)
        self.sum_of_rewards = np.zeros(k)
        self.delta = delta
        self.exploration_vector = find_projected_on_simplex_equivalent_in_X_space(contexts, 1/k * np.ones(k))
        
        self.optimization_failed_flag = False
        self.optimization_failed_number_of_rounds = 0
        
        self.mode = mode
        
        self.exploration_vector_Z = (contexts.T @ self.exploration_vector) / np.sum(contexts.T @ self.exploration_vector)
        
        
        self.sum_points_played = np.zeros(k)
        self.sum_ws = np.zeros(k)#only used for G-tracking
        
        self.sum_ws_c = np.zeros(n) #only used for c-tracking

    def best_empirical_arm_calculator(self):
        mean_hat = self.sum_of_rewards / self.K_times_seen
        
        actions_mu_hat = np.dot(self.contexts, mean_hat)
        delta_hat = np.max(actions_mu_hat) - actions_mu_hat
        
        best_arm = np.argmax(actions_mu_hat)  
        return best_arm, actions_mu_hat, delta_hat

    def lambda_hat(self):
        #compute delta hat from estimated means
        best_arm, _, delta_hat = self.best_empirical_arm_calculator()

        result = 1 / (np.sum((self.contexts - self.contexts[best_arm])**2 / self.K_times_seen, axis=1)) 
        result = result * (delta_hat**2) / 2    

        #since we have 0 in the i*-th element and that should not be computed in arg min
        return SecondMin(result)

    def Stopping_Rule(self):
        lambda_hat_t = self.lambda_hat() 

        c_t = c_hat_sep(self.k, self.K_times_seen, self.delta)

        return c_t >= lambda_hat_t

    def Alternate_Stopping_Rule(self):
        lambda_hat_t = self.lambda_hat() 
        
        c_t = c_hat_t(self.n, 1 , self.T, self.delta)

        return c_t >= lambda_hat_t

    def C_Stopping_Rule(self):  
        best_arm, _, delta_hat = self.best_empirical_arm_calculator()
        
        V = - self.contexts + self.contexts[best_arm]
        
        expected_K = self.contexts.T @ self.N_times_seen
        
        confidence = np.sqrt(
            2*(
                self.k * Cg(np.log((self.n - 1) / self.delta)/self.k) + 
                np.sum (2 * np.log(4 + np.log(self.K_times_seen)))
            ) * np.sum(V ** 2 / expected_K, axis = 1))
        
        return SecondMin(delta_hat - confidence) < 0

    def optimization_line_coefficient(self, w_t, v_k): 

        w_t = np.asarray(w_t)
        v_k = np.asarray(v_k)
        original_array = np.asarray(self.contexts)


        def target_vector(alpha):
            return (1 + alpha) * w_t - alpha * v_k


        c = np.zeros(self.n + 1)  # n lambdas + 1 alpha
        c[-1] = -1  # Coefficient for -alpha

        A_eq = np.zeros((self.k + 1, self.n + 1))
        A_eq[:self.k, :self.n] = original_array.T
        A_eq[:self.k, -1] = -(w_t - v_k) 
        A_eq[self.k, :self.n] = 1  
        A_eq[self.k, -1] = 0  # Alpha does not contribute to sum of lambdas

        b_eq = np.zeros(self.k + 1)
        b_eq[:self.k] = w_t
        b_eq[self.k] = 1 

        A_ub = np.zeros((self.n, self.n + 1))
        A_ub[:, :self.n] = -np.eye(self.n)  # -lambda_i <= 0
        b_ub = np.zeros(self.n)

        bounds = [(0, None)] * self.n + [(None, None)]
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

        if result.success:
            return result.x[-1]  #alpha
        else:
            self.optimization_failed_flag = True
            #raise ValueError("Optimization failed: " + result.message)

            
    
    def optimization_for_Z_optimal_vector(self):
        
        i_star, _, delta = self.best_empirical_arm_calculator()

        lambda_var = cp.Variable(self.n, nonneg=True)  # Weights for convex combination
        w = self.contexts.T @ lambda_var  # w is a convex combination of rows of A
        t = cp.Variable()

        constraints = [cp.sum(lambda_var) == 1]  # Convex combination constraint
        
        constraints += [w[j] >= 0 for j in range(self.k)]
        
        for i in range(self.n):
            if i == i_star:
                continue
                

            middle = cp.sum([(self.contexts[i, j] - self.contexts[i_star, j])**2 * cp.inv_pos(w[j]) for j in range(self.k)])
            constraints.append(
                t >= middle / (delta[i] ** 2)
            )

        objective = cp.Minimize(t)
        problem = cp.Problem(objective, constraints)
        
        
        
        try:
            problem.solve()
        except SolverError as e:
            self.optimization_failed_flag = True


        if problem.status not in ["optimal", "optimal_inaccurate"]:
            self.optimization_failed_flag = True
            return 0,0
            #raise ValueError("Optimization problem did not converge.")

        # Return the optimal w and t
        w_opt = w.value
        w_opt /= np.sum(w_opt)
        t_opt = t.value
        
        return w_opt, t_opt
    
   




    def optimization_for_p(self, w):
        
        N_t = self.K_times_seen

        lambda_ = cp.Variable(self.n)
        p = self.contexts.T @ lambda_
        

        # ||N_t + p - w||^2 = ||p - (w - N_t)||^2
        y = w - N_t  
        objective = cp.Minimize(cp.norm((p + N_t)/(np.sum(N_t)+1) - w/np.sum(w), 2)**2)

        constraints = [
            lambda_ >= 0, 
            cp.sum(lambda_) == 1
        ]

        problem = cp.Problem(objective, constraints)
        problem.solve()

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            raise ValueError(f"Solver failed: {problem.status}")

            
        lambda_opt = lambda_.value  
        p_opt = self.contexts.T @ lambda_opt 
        
        p_opt /= np.sum(p_opt)

        return p_opt, lambda_opt
        



    def Optimal_W(self):
        
        if not self.mode['use_optimized_p']:
            
            if self.mode['average_points_played']:
                #handling the first round
                v_k = self.sum_points_played / np.sum(self.sum_points_played) if np.sum(self.sum_points_played)!=0 else self.sum_points_played
                
            else:
                v_k = self.K_times_seen / np.sum(self.K_times_seen)

            w_t, _ = self.optimization_for_Z_optimal_vector()
            


            if self.optimization_failed_flag == True:
                self.optimization_failed_number_of_rounds += 1
                print("failed1")
                self.sum_points_played += self.exploration_vector_Z
                return self.exploration_vector, np.array([])

            
            if self.mode['average_w']:
                w = self.sum_ws + w_t
                w /= np.sum(w)
            else:
                w = w_t

                
            alpha = self.optimization_line_coefficient(w, v_k)

            if self.optimization_failed_flag == True:
                self.optimization_failed_number_of_rounds += 1
                print("failed2")
                self.sum_points_played += self.exploration_vector_Z
                return self.exploration_vector, np.array([])

            OPT_Z = (1 + alpha) * w_t - alpha * v_k

            OPT_X = convert_back_to_X_space(self.contexts, OPT_Z)


            if not isinstance(OPT_X, np.ndarray):
                self.optimization_failed_number_of_rounds += 1
                print("failed3")
                self.sum_points_played += self.exploration_vector_Z
                return self.exploration_vector ,  np.array([])

            self.sum_ws += w_t
            self.sum_points_played += OPT_Z
         
        
        
            return OPT_X, w_t
        
        else:
            
            w_t, _ = self.optimization_for_Z_optimal_vector()
            
            if self.optimization_failed_flag == True:
                self.optimization_failed_number_of_rounds += 1
                print("failed1")
                self.sum_points_played += self.exploration_vector_Z
                return self.exploration_vector,  np.array([])
            
            w = self.sum_ws + w_t
            
            OPT_Z, _ = self.optimization_for_p(w) 
            
            
            
            
            OPT_X = convert_back_to_X_space(self.contexts, OPT_Z)


            if not isinstance(OPT_X, np.ndarray):
                self.optimization_failed_number_of_rounds += 1
                print("failed3")
                self.sum_points_played += self.exploration_vector_Z
                return self.exploration_vector, np.array([])

            
            self.sum_points_played += OPT_Z
            self.sum_ws += w_t
            
            return OPT_X, w_t
            

    
    def C_optimal_W(self):
        
        
        i_star, _, delta = self.best_empirical_arm_calculator()

        lambda_var = cp.Variable(self.n, nonneg=True)  # Weights for convex combination
        w = self.contexts.T @ lambda_var  # w is a convex combination of rows of A
        t = cp.Variable() 


        # Constraints
        constraints = [cp.sum(lambda_var) == 1]  # Convex combination constraint
        
        constraints += [w[j] >= 0 for j in range(self.k)]
        
        
        for i in range(self.n):
            if i == i_star:
                continue
                
            middle = cp.sum([(self.contexts[i, j] - self.contexts[i_star, j])**2 * cp.inv_pos(w[j]) for j in range(self.k)])
            constraints.append(
                t >= middle / (delta[i] ** 2)
            )

        objective = cp.Minimize(t)

        problem = cp.Problem(objective, constraints)
        
        
        
        try:
            problem.solve()
        except SolverError as e:
            self.optimization_failed_flag = True


        if problem.status not in ["optimal", "optimal_inaccurate"]:
            self.optimization_failed_flag = True
            return 0
            #raise ValueError("Optimization problem did not converge.")

        # Return the optimal lambda_var
        lambda_var_opt = lambda_var.value / np.sum(lambda_var.value)
        
        return lambda_var_opt
    
    
    
    def projected_c_optimal_w(self, w):
        
        eps = 0.5/np.sqrt(self.n **2 + self.T)

        v = cp.Variable(self.n)
        t = cp.Variable()

        objective = cp.Minimize(t)

        # Constraints
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
                #print("1")
                return np.array([])
        except Exception as e:
            self.optimization_failed_flag = True
            #print("2")
            #print(f"An error occurred during optimization: {str(e)}")
            return np.array([])

        # Return the projected vector
        return v.value
        
            
    
    
    def C_Tracking(self):
        w = self.C_optimal_W()
        if self.optimization_failed_flag:
            print("optimization failed 1")
            self.optimization_failed_number_of_rounds += 1
            return 0
        
        w_eps = self.projected_c_optimal_w(w)
        if self.optimization_failed_flag:
            self.optimization_failed_number_of_rounds += 1
            print("optimization failed 2")
            return 0
            
        self.sum_ws_c += w/np.sum(w) #make sure w sums up to 1
        
        sum_w_scaled = (self.sum_ws_c)/ np.sum(self.sum_ws_c) * np.sum(self.N_times_seen) #some optimization rounds may have failed so scale up to match it
        result = sum_w_scaled - self.N_times_seen
        return np.argmax(result)
        
        



    def G_Tracking(self):
        #no need for exploration!
#         if int(self.T**0.5) ** 2 == self.T:
#             return hidden_action_sampler(self.exploration_vector)
        
        w, _ = self.Optimal_W()
        result = self.T * w - self.N_times_seen
        return np.argmax(result)
    


    
    def Initialization_Phase(self, means, samples):
        while np.any(self.K_times_seen == 0):
            result = np.where(self.K_times_seen == 0)
            j_prime = result[0][0]
            i = np.argmax(self.contexts[:, j_prime])
            j = hidden_action_sampler(self.contexts[i])
            self.N_times_seen[i] += 1
            self.K_times_seen[j] += 1
            self.sum_of_rewards[j] += samples[self.T] + means[j] #sample according to x ~ N(u, 1) === x = y + u , u ~ N(0, 1)
            self.T += 1
#             self.sum_ws += self.contexts[i]
            self.sum_points_played += self.contexts[i]

#         if self.context_estimate == True:
#             self.contexts = self.N_times_seen / self.N_times_seen.sum(axis=1, keepdims=True)

        return self.T
        

    def update(self, main_action, post_action, reward):
        self.T += 1
        self.N_times_seen[main_action] += 1
        self.sum_of_rewards[post_action] += reward
        self.K_times_seen[post_action] += 1
        
        self.optimization_failed_flag = False
