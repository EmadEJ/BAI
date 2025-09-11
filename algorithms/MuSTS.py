from utils import *
import cvxpy as cp
from scipy.optimize import minimize
from scipy.special import softmax
from algorithms.TS import TS

# known Mu Seperator Track and Stop
class MuSTS(TS):
    def __init__(self, n, k, mu, confidence, tracking, mode = {'average_w': False}):
        super().__init__(n, k, confidence, tracking, mode)
        self.mu = mu
        
        self.N_A = np.zeros(n)
        self.N_Z = np.zeros(k)
        self.cnt_post_actions = np.zeros((n, k))
    
    def get_A_hat(self):
        return self.cnt_post_actions / self.N_A.reshape((self.n, 1))
    
    def get_means_hat(self):
        return self.get_A_hat() @ self.mu

    def best_empirical_arm(self):
        A_hat = self.get_A_hat()

        means_hat = np.dot(A_hat, self.mu)
        delta_hat = np.max(means_hat) - means_hat

        best_arm = np.argmax(means_hat)  
        return best_arm, means_hat, delta_hat

    def lambda_hat(self, w = None):
        if w is None:
            N_A = self.N_A
        else:
            N_A = w
        A_hat = self.get_A_hat()
        i_star, _, _ = self.best_empirical_arm()
        
        obj_star = np.inf
        for s in range(self.n):
            if s == i_star:
                continue
            # Compute the weight for each arm
            Ai = cp.Variable(self.k)
            As = cp.Variable(self.k)

            objective = cp.Minimize(
                N_A[i_star] * cp.sum(cp.rel_entr(A_hat[i_star], Ai)) + 
                N_A[s] * cp.sum(cp.rel_entr(A_hat[s], As))
            )
            constraints = [
                Ai >= 0,
                cp.sum(Ai) == 1,
                As >= 0,
                cp.sum(As) == 1,
                (As - Ai) @ self.mu >= 0
            ]
            problem = cp.Problem(objective, constraints)
            problem.solve()
            
            if problem.status == "optimal":
                obj_star = min(obj_star, problem.value)
            else:
                self.optimization_failed_flag = True
                self.optimization_failed_number_of_rounds += 1

        if w is None:
            return obj_star
        
        # in case we are doing coordinate descent
        A_p = A_hat.copy()
        A_p[i_star] = Ai.value
        A_p[s] = As.value
        return obj_star, A_p

    def beta_t_A(self, delta):
        return (
            np.log(1/delta) +
            (self.k-1) * sum([np.log(np.e * (1 + self.N_A[i] / (self.k-1))) for i in range(self.n)])
        )

    def stopping_rule(self):
        # returns True if need to stop and are confident enough
        lambda_hat_t = self.lambda_hat() 
        beta_t = self.beta_t_A(self.confidence)
        return lambda_hat_t > beta_t, lambda_hat_t, beta_t

    def optimal_w(self):
        # Uses coordinate descent currently
        A_hat = self.get_A_hat()
        i_star, _, _ = self.best_empirical_arm()
        
        # Emad's Method
        opts = []
        for s in range(self.n):
            if s == i_star:
                opts.append(0)
                continue
            
            w = np.array([1, 1])

            Ai = cp.Variable(self.k)
            As = cp.Variable(self.k)

            objective = cp.Minimize(
                w[0] * cp.sum(cp.rel_entr(A_hat[i_star], Ai)) + 
                w[1] * cp.sum(cp.rel_entr(A_hat[s], As))
            )
            constraints = [
                Ai >= 0,
                cp.sum(Ai) == 1,
                As >= 0,
                cp.sum(As) == 1,
                (As - Ai) @ self.mu >= 0
            ]
            problem = cp.Problem(objective, constraints)
            problem.solve()
            opts.append(problem.value)

            # grid = np.linspace(0, 1, 21)
            # opts = []
            # for w0 in grid:
            #     w = np.array([w0, 1-w0])
            #     Ai = cp.Variable(self.k)
            #     As = cp.Variable(self.k)

            #     objective = cp.Minimize(
            #         w[0] * cp.sum(cp.rel_entr(A_hat[i_star], Ai)) + 
            #         w[1] * cp.sum(cp.rel_entr(A_hat[s], As))
            #     )
            #     constraints = [
            #         Ai >= 0,
            #         cp.sum(Ai) == 1,
            #         As >= 0,
            #         cp.sum(As) == 1,
            #         (As - Ai) @ self.mu >= 0
            #     ]
            #     problem = cp.Problem(objective, constraints)
            #     problem.solve()
            #     opts.append(problem.value)
            # plt.plot(grid, opts)
            # plt.show()
            # w = np.zeros(self.n)
            # w[i_star] = 1
            # w[s] = 1
            
        # Second smallest
        target_obj = np.sort(opts)[1]
        w_star = np.zeros(self.n)
        for s in range(self.n):
            if s == i_star:
                w_star[i_star] = 1
                continue
            
            l, r = 0, 1
            while r - l > 1e-4:
                mid = (l + r) / 2
                w = np.array([1, mid])

                Ai = cp.Variable(self.k)
                As = cp.Variable(self.k)

                objective = cp.Minimize(
                    w[0] * cp.sum(cp.rel_entr(A_hat[i_star], Ai)) + 
                    w[1] * cp.sum(cp.rel_entr(A_hat[s], As))
                )
                constraints = [
                    Ai >= 0,
                    cp.sum(Ai) == 1,
                    As >= 0,
                    cp.sum(As) == 1,
                    (As - Ai) @ self.mu >= 0
                ]
                problem = cp.Problem(objective, constraints)
                problem.solve()
                
                if problem.value < target_obj:
                    l = mid
                else:
                    r = mid

            w_star[s] = (l + r) / 2

        w_star = w_star / np.sum(w_star)
        
        # # Iterative differentation
        # def fun(w):
        #     w_soft = softmax(w)
        #     return -self.lambda_hat(w_soft)[0]

        # def jac(w):
        #     w_soft = softmax(w)

        #     _, A_star = self.lambda_hat(w_soft)
        #     s = np.argmax(np.dot(A_star, self.mu))
        #     if i_star == s:
        #         # This happens when mean of s and i_star equal exactly
        #         # Replace s with the second biggest value
        #         sorted_arms = np.argsort(np.dot(A_star, self.mu))
        #         if sorted_arms[-1] == i_star:
        #             s = sorted_arms[-2]
        #         else:
        #             s = sorted_arms[-1]

        #     grad = np.zeros(self.n)
        #     for i in range(self.n):
        #         grad[i] = categorical_kl(A_hat[i], A_star[i])
            
        #     grad = np.multiply(w_soft, (grad - np.dot(w_soft, grad)))  # accounting for softmax
                    
        #     return -grad
        
        # w0 = np.random.rand(self.n)
        # w0 = w0 / np.sum(w0)
        # result = minimize(
        #         fun=fun, 
        #         jac=jac,
        #         x0=w0, 
        #         method="BFGS",
        #         options={"xrtol": 1e-4}
        #     )
        # scipy_w_star = softmax(result.x)
        
        # if np.allclose(w_star, scipy_w_star):
        #     print("Good News!")
        # else:
        #     obj_star = self.lambda_hat(w_star)[0]
        #     scipy_obj_star = self.lambda_hat(scipy_w_star)[0]
        #     if obj_star > scipy_obj_star:
        #         print(f"I'm Better! {obj_star - scipy_obj_star}")
        #     else:
        #         print(f"Scipy Better! {obj_star - scipy_obj_star}")
        
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
        self.N_Z[post_action] += 1
        
        self.optimization_failed_flag = False
