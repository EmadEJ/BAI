from utils import *
import cvxpy as cp
from scipy.optimize import minimize_scalar
from algorithms.TS import TS
import itertools

# known Mu Separator Track and Stop
class MuSTS(TS):
    def __init__(self, n, k, mu, confidence, tracking, mode = {'average_w': False, 'fast': True}):
        super().__init__(n, k, confidence, tracking, mode)
        self.mu = mu
        
        self.N_A = np.zeros(n)
        self.N_Z = np.zeros(k)
        self.cnt_post_actions = np.zeros((n, k))
        
        self.projection = None

    def kl_projection(self, A_star, A_s, w_star, w_s):
        # min w_star KL(A_star||B1) + w_s KL(A_s||B2) s.t. (B2 - B1)^T mu >= 0
        # KL(P||B) = sum P log P - P^T log B, so the problem is compiled once with c = w P as parameters
        if self.projection is None:
            self.c1 = cp.Parameter(self.k, nonneg=True)
            self.c2 = cp.Parameter(self.k, nonneg=True)
            self.B1 = cp.Variable(self.k)
            self.B2 = cp.Variable(self.k)
            objective = cp.Minimize(-self.c1 @ cp.log(self.B1) - self.c2 @ cp.log(self.B2))
            constraints = [
                cp.sum(self.B1) == 1,
                cp.sum(self.B2) == 1,
                (self.B2 - self.B1) @ self.mu >= 0
            ]
            self.projection = cp.Problem(objective, constraints)
        
        self.c1.value = w_star * A_star
        self.c2.value = w_s * A_s
        self.projection.solve()
        
        xlogx = lambda P: np.sum(P[P > 0] * np.log(P[P > 0]))
        value = self.projection.value + w_star * xlogx(A_star) + w_s * xlogx(A_s)
        return max(value, 0), self.projection.status, self.B1.value, self.B2.value
    
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

    def lambda_hat(self, w = None, A = None):
        N_A = self.N_A if w is None else w
        A_hat = self.get_A_hat() if A is None else A

        i_star = np.argmax(np.dot(A_hat, self.mu))
        
        obj_star = np.inf
        for s in range(self.n):
            if s == i_star:
                continue
            value, status, B1, B2 = self.kl_projection(A_hat[i_star], A_hat[s], N_A[i_star], N_A[s])
            
            if status in ["optimal", "optimal_inaccurate"]:
                if value <= obj_star:
                    obj_star = value
                    A_p = A_hat.copy()
                    A_p[i_star] = B1
                    A_p[s] = B2
            else:
                # a failed solve must not make the algorithm stop
                obj_star, A_p = 0, A_hat.copy()
                self.optimization_failed_flag = True
                self.optimization_failed_number_of_rounds += 1

        if w is None:
            return obj_star
        
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
        if self.skip_optimization():
            return self.last_w
        self.n_optimizations += 1
        
        A_hat = self.get_A_hat()
        i_star, _, _ = self.best_empirical_arm()
        
        # Our Method
        EPS = 0.01  # Tolerence in w error
        def inner_func(w, s):
            return self.kl_projection(A_hat[i_star], A_hat[s], w[0], w[1])[0]
        
        def fixed_w_istar(w_istar, give_w=False):
            w = np.zeros(self.n)
            w[i_star] = w_istar
            
            funcs = [inner_func([w_istar, w[i]], i) for i in range(self.n)]
                
            while sum(w) < 1:
                s_star = None
                for s in range(self.n):
                    if s == i_star:
                        continue
                    if s_star is None or funcs[s] < funcs[s_star]:
                        s_star = s
            
                w[s_star] += EPS
            
                funcs[s_star] = inner_func([w_istar, w[s_star]], s_star)
           
            if give_w:
                return w
            
            return -np.sort(funcs)[1]
        
        res = minimize_scalar(
            fixed_w_istar,
            bounds=(0, 1),
            method="bounded",
            options={"xatol": EPS}
        )
        
        w_star = fixed_w_istar(res.x, give_w=True)
        w_star = w_star / np.sum(w_star)
        self.last_w = w_star
        
        return w_star
    
    def F_on_simplex(self, mu, A, div=11):
        # F_mu(w) of eq. (3) on a grid of the simplex (n = 3)
        grid_range = np.linspace(0, 1, div)
        grid = itertools.product(grid_range, repeat=self.n-1)
        Ts = {}
        for w in grid:
            w0 = 1 - sum(w)
            if w0 < -1e-9:
                continue
            w = np.array(([max(w0, 0)] + list(w)))
            
            obj = self.lambda_hat(w, A)[0]
            
            Ts[tuple(w)] = obj
            
        return Ts
    
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
