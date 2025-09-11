import numpy as np

from algorithms.STS import *
from algorithms.ASTS import *
from algorithms.MuSTS import *
from algorithms.SGTS import *
from utils import *

class Environment:
    
    def __init__(self, mus, A, n, k):

        self.mus = mus  # mean reward of each context
        self.A = A  # context given arm probabilty matrix
        self.n = n
        self.k = k
        self.T = 0
        self.samples = np.random.normal(loc=0, scale=1, size=1000000)  # sample noise at each arm pull
        
        self.means = np.dot(self.A, self.mus)
        self.best_arm = np.argmax(self.means)
        self.delta = self.means[self.best_arm] - self.means
                
        self.log_period = 10  # every <log_period> iterations save the data of the arms played up to now and optimal w


    def take_action(self, action):
        post_action = hidden_action_sampler(self.A[action])
        reward = self.samples[self.T] + self.mus[post_action]    
        self.T += 1
        return post_action, reward

    def run_STS(self, alg: STS, verbose=False):
        mu_hats = []
        A_hats = []
        w_s = []
        N_As = []
        N_Zs = []
        lambda_lbs = []
        lambdas = []
        betas = []

        in_init = True
        while in_init or not alg.stopping_rule()[0]:
            # Select an action using the algorithm
            action, init = alg.get_action()
            
            if in_init and not init:
                in_init = False
                if verbose:
                    print(f"----- initialization finished with {self.T} rounds -----")

            post_action, reward = self.take_action(action)
            alg.update(action, post_action, reward)
            
            if not in_init and self.T % self.log_period == 0:
                w = alg.optimal_w().tolist()
                w_s.append(w)
                mu_hats.append(alg.get_mu_hat().tolist())
                A_hats.append(alg.get_A_hat().tolist())
                N_As.append(alg.N_A.tolist())
                N_Zs.append(alg.N_Z.tolist())

                _, lambda_lb_t, _ = alg.stopping_rule_lb()
                _, lambda_t, beta_t = alg.stopping_rule()
                lambda_lbs.append(lambda_lb_t)
                lambdas.append(lambda_t)
                betas.append(beta_t)

                if verbose:
                    print(f"Round {self.T}, action {action}, post_action {post_action}, reward {reward}")
                    print(f"lambda_lb_t: {lambda_lb_t}, lambda_hat_t: {lambda_t}, beta_t: {beta_t}, confidence: {alg.confidence}")
                    print(f"w: {w}")
                    print(f"means: {alg.get_means_hat()}")
                    print("#" * 50)
            
        if verbose:
            print("number of failed optimization rounds is ", alg.optimization_failed_number_of_rounds)
        
        _, lambda_lb_t, _ = alg.stopping_rule_lb()
        _, lambda_t, beta_t = alg.stopping_rule()
        lambda_lbs.append(lambda_lb_t)
        lambdas.append(lambda_t)
        betas.append(beta_t)
        best_arm = int(alg.best_empirical_arm()[0])

        result = {
            'T': self.T,
            'best_arm': best_arm,
            'mu_hats': mu_hats,
            'A_hats': A_hats,
            'N_As': N_As,
            'N_Zs': N_Zs,
            'w_s': w_s,
            'lambda_lbs': lambda_lbs,
            'lambdas': lambdas,
            'betas': betas
        }
        return result
    
    def run_MuSTS(self, alg: MuSTS, verbose=False):
        A_hats = []
        w_s = []
        N_As = []
        N_Zs = []
        lambdas = []
        betas = []

        in_init = True
        while in_init or not alg.stopping_rule()[0]:
            # Select an action using the algorithm
            action, init = alg.get_action()
            
            if in_init and not init:
                in_init = False
                if verbose:
                    print(f"----- initialization finished with {self.T} rounds -----")                  
            
            post_action, reward = self.take_action(action)
            alg.update(action, post_action, reward)
            
            
            if not in_init and self.T % self.log_period == 0:
                w = alg.optimal_w().tolist()
                w_s.append(w)
                A_hats.append(alg.get_A_hat().tolist())
                N_As.append(alg.N_A.tolist())
                N_Zs.append(alg.N_Z.tolist())

                _, lambda_t, beta_t = alg.stopping_rule()
                lambdas.append(lambda_t)
                betas.append(beta_t)

                if verbose:
                    print(f"Round {self.T}, action {action}, post_action {post_action}, reward {reward}")
                    print(f"lambda_hat_t: {lambda_t}, beta_t: {beta_t}, confidence: {alg.confidence}")
                    print(f"w: {w}")
                    print(f"means: {alg.get_means_hat()}")
                    print("#" * 50)

        if verbose:
            print("number of failed optimization rounds is ", alg.optimization_failed_number_of_rounds)

        _, lambda_t, beta_t = alg.stopping_rule()
        lambdas.append(lambda_t)
        betas.append(beta_t)
        best_arm = int(alg.best_empirical_arm()[0])

        result = {
            'T': self.T,
            'best_arm': best_arm,
            'A_hats': A_hats,
            'N_As': N_As,
            'N_Zs': N_Zs,
            'w_s': w_s,
            'lambdas': lambdas,
            'betas': betas
        }
        return result

    def run_ASTS(self, alg: ASTS, verbose=False):
        mu_hats = []
        w_s = []
        N_As = []
        N_Zs = []
        lambdas = []
        betas = []

        in_init = True
        while in_init or not alg.stopping_rule()[0]:
            # Select an action using the algorithm
            action, init = alg.get_action()
            
            if in_init and not init:
                in_init = False
                if verbose:
                    print(f"----- initialization finished with {self.T} rounds -----")

            post_action, reward = self.take_action(action)
            alg.update(action, post_action, reward)
            
            if not in_init and self.T % self.log_period == 0:
                w = alg.optimal_w().tolist()
                w_s.append(w)
                mu_hats.append(alg.get_mu_hat().tolist())
                N_As.append(alg.N_A.tolist())
                N_Zs.append(alg.N_Z.tolist())

                _, lambda_t, beta_t = alg.stopping_rule()
                lambdas.append(lambda_t)
                betas.append(beta_t)
                
                if verbose:
                    print(f"Round {self.T}, action {action}, post_action {post_action}, reward {reward}")
                    print(f"lambda_hat_t: {lambda_t}, beta_t: {beta_t}, confidence: {alg.confidence}")
                    print(f"w: {w}")
                    print(f"means: {alg.get_means_hat()}")
                    print("#" * 50)

        if verbose:
            print("number of failed optimization rounds is ", alg.optimization_failed_number_of_rounds)

        _, lambda_t, beta_t = alg.stopping_rule()
        lambdas.append(lambda_t)
        betas.append(beta_t)
        best_arm = int(alg.best_empirical_arm()[0])

        result = {
            'T': self.T,
            'best_arm': best_arm,
            'mu_hats': mu_hats,
            'N_As': N_As,
            'N_Zs': N_Zs,
            'w_s': w_s,
            'lambdas': lambdas,
            'betas': betas
        }
        return result

    def run_SGTS(self, alg: SGTS, verbose=False):
        w_s = []
        N_As = []
        lambdas = []
        betas = []

        in_init = True
        while in_init or not alg.stopping_rule()[0]:
            # Select an action using the algorithm
            action, init = alg.get_action()
            
            if in_init and not init:
                in_init = False
                if verbose:
                    print(f"----- initialization finished with {self.T} rounds -----")

            post_action, reward = self.take_action(action)
            alg.update(action, post_action, reward)
            
            if not in_init and self.T % self.log_period == 0:
                w = alg.optimal_w().tolist()
                w_s.append(w)
                N_As.append(alg.N_A.tolist())

                _, lambda_t, beta_t = alg.stopping_rule()
                lambdas.append(lambda_t)
                betas.append(beta_t)

                if verbose:
                    print(f"Round {self.T}, action {action}, post_action {post_action}, reward {reward}")
                    print(f"lambda_hat_t: {lambda_t}, beta_t: {beta_t}, confidence: {alg.confidence}")
                    print(f"w: {w}")
                    print(f"means: {alg.get_means_hat()}")
                    print("#" * 50)
            
        if verbose:
            print("number of failed optimization rounds is ", alg.optimization_failed_number_of_rounds)
        
        _, lambda_t, beta_t = alg.stopping_rule()
        lambdas.append(lambda_t)
        betas.append(beta_t)
        best_arm = int(alg.best_empirical_arm()[0])

        result = {
            'T': self.T,
            'best_arm': best_arm,
            'N_As': N_As,
            'w_s': w_s,
            'lambdas': lambdas,
            'betas': betas
        }
        return result

    def run(self, confidence, algorithm, tracking, mode = {'average_w': False}, verbose=False):
        if algorithm == "STS":
            alg = STS(self.n, self.k, confidence, tracking, mode)
            return self.run_STS(alg, verbose=verbose)
        if algorithm == "ASTS":
            alg = ASTS(self.n, self.k, self.A, confidence, tracking, mode)
            return self.run_ASTS(alg, verbose=verbose)
        if algorithm == "MuSTS":
            alg = MuSTS(self.n, self.k, self.mus, confidence, tracking, mode)
            return self.run_MuSTS(alg, verbose=verbose)
        if algorithm == "SGTS":
            alg = SGTS(self.n, self.k, confidence, tracking, mode)
            return self.run_SGTS(alg, verbose=verbose)
        print("Invalid Algorithm!")
        return None
        