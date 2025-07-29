import numpy as np
import cvxpy as cp
from algorithms.STS import *
from utils import *

class Environment:
    
    def __init__(self, mus, A, algorithm, n, k, confidence, mode = {'use_optimized_p': False, 'average_w': False, 'average_points_played': False}, stopping_rule = 'd_stopping_rule'):

        self.algorithm = algorithm
        self.mus = mus  # mean reward of each context
        self.A = A  # context given arm probabilty matrix
        self.n = n
        self.k = k
        self.T = 0
        self.confidence = confidence
        self.samples = np.random.normal(loc=0, scale=1, size=1000000)  # sample noise at each arm pull
        self.mode = mode
        
        self.means = np.dot(self.A, self.mus)
        self.best_arm = np.argmax(self.means)
        self.delta = self.means[self.best_arm] - self.means
        
        self.optimal_W, self.T_star = self.optimal_weight()
        
        self.T_star = 0.5 / self.T_star  # NOTE: why 0.5??
        
        self.mu_hats = []
        self.A_hat = []
        self.w_s = []
        self.N_times_seens = []
        
        self.log_period = 20  # every 20 iteration save the data of the arms played up to now and optimal w
        
        self.stopping_rule = stopping_rule  # for c-tracking 
       
    def optimal_weight(self):
        # TODO: this should change for unkonwn A case. and should be moved to algorithms
        
        delta = self.delta
        i_star = self.best_arm

        lambda_var = cp.Variable(self.n, nonneg=True)  # Weights for convex combination
        w = self.A.T @ lambda_var  # w is a convex combination of rows of A
        t = cp.Variable()

        constraints = [cp.sum(lambda_var) == 1]  
        
        constraints += [w[j] >= 0 for j in range(self.k)]
        
        
        for i in range(self.n):
            if i == i_star:  # constraints are for non-optimal arms
                continue

            middle = cp.sum([(self.A[i, j] - self.A[i_star, j])**2 * cp.inv_pos(w[j]) for j in range(self.k)])
            constraints.append(t >= middle / (delta[i] ** 2))

        objective = cp.Minimize(t)
        
        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve()
        except cp.SolverError as e:
            self.optimization_failed_flag = True


        if problem.status not in ["optimal", "optimal_inaccurate"]:
            self.optimization_failed_flag = True
            return 0,0
            # raise ValueError("Optimization problem did not converge.")

        return w.value, t.value 

    def loop(self):
        if self.algorithm == 'STS': #Seperator Track and Stop            
            alg = STS(self.n, self.k, self.confidence, self.A, self.mode)  # shouldn't get A
            self.T = alg.Initialization_Phase(self.mus, self.samples)
            
            print(f"initialization finished with {self.T} rounds")
            
            while alg.Stopping_Rule():
                # print(self.T)
                # Select an action using the algorithm
                action = alg.G_Tracking()
                post_action = hidden_action_sampler(self.A[action])
                reward = self.samples[self.T] + self.mus[post_action]
                alg.update(action, post_action, reward)
    
                self.T += 1
                
                w = alg.Optimal_W()[1]
                w = w.tolist()
                
                if w and self.T % self.log_period == 0:
                    self.w_s.append(w)
                    _, actions_mu_hat, _ = alg.best_empirical_arm_calculator()

                    self.mu_hats.append(actions_mu_hat.tolist())
                    self.N_times_seens.append(alg.N_times_seen.tolist())
                
            best_arm, _, _ = alg.best_empirical_arm_calculator()
            print(f"number of failed optimization rounds is {alg.optimization_failed_number_of_rounds}")
        
        return best_arm, self.mu_hats, self.N_times_seens, self.w_s, self.T