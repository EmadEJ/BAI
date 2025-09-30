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
        w_s = []
        lambda_lbs = []
        lambdas = []
        # true_lambdas = []
        betas = []
        beta2s = []

        if verbose:
            alg.plot_w(self.mus, self.A)

        T_star, w_star = alg.get_T_star(self.mus, self.A)
        if verbose:
            print(f"Asymptotically expecting {dB(alg.confidence) * T_star} arm pulls!")
            print(T_star, w_star)

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

                lambda_lb_t = alg.lambda_lb()
                # lambda_true = alg.lambda_true()
                _, _, beta_t = alg.stopping_rule()
                _, lambda_t, beta_t2 = alg.stopping_rule2()
                lambda_lbs.append(lambda_lb_t)
                # true_lambdas.append(lambda_true)
                lambdas.append(lambda_t)
                betas.append(beta_t)
                beta2s.append(beta_t2)

                if verbose:
                    print(f"Round {self.T}, action {action}, post_action {post_action}, reward {reward}")
                    print(f"lambda_lb_t: {lambda_lb_t}, lambda_hat_t: {lambda_t},\nbeta_t: {beta_t}, beta_t2: {beta_t2}, confidence: {alg.confidence}")
                    print(f"w: {w}")
                    print(f"A_hat: {alg.get_A_hat()}")
                    print(f"mu_hat: {alg.get_mu_hat()}")
                    print(f"means: {alg.get_means_hat()}")
                    print("#" * 50)
            
        if verbose:
            print("number of failed optimization rounds is ", alg.optimization_failed_number_of_rounds)
        
        lambda_lb_t = alg.lambda_lb()
        # lambda_true = alg.lambda_true()
        _, _, beta_t = alg.stopping_rule()
        _, lambda_t, beta_t2 = alg.stopping_rule2()
        lambda_lbs.append(lambda_lb_t)
        # true_lambdas.append(lambda_true)
        lambdas.append(lambda_t)
        betas.append(beta_t)
        beta2s.append(beta_t2)
        
        best_arm = int(alg.best_empirical_arm()[0])

        result = {
            'T': self.T,
            'best_arm': best_arm,
            'w_s': w_s,
            'lambda_lbs': lambda_lbs,
            # 'true_lambdas': true_lambdas, 
            'lambdas': lambdas,
            'betas': betas,
            'beta2s': beta2s
        }
        return result
    
    def run_MuSTS(self, alg: MuSTS, verbose=False):
        w_s = []
        lambdas = []
        betas = []

        if verbose:
            alg.plot_w(self.mus, self.A)

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
            'w_s': w_s,
            'lambdas': lambdas,
            'betas': betas
        }
        return result

    def run_ASTS(self, alg: ASTS, verbose=False):
        w_s = []
        lambdas = []
        betas = []

        if verbose:
            alg.plot_w(self.mus, self.A)

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
            'w_s': w_s,
            'lambdas': lambdas,
            'betas': betas
        }
        return result

    def run_SGTS(self, alg: SGTS, verbose=False):
        w_s = []
        lambdas = []
        betas = []

        if verbose:
            alg.plot_w(self.mus, self.A)

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
        