from utils import *
import cvxpy as cp
from abc import ABC, abstractmethod

# Base Track and Stop
class TS(ABC):
    def __init__(self, n, k, confidence, tracking, mode = {'average_w': False}):
        self.n = n
        self.k = k
        self.confidence = confidence
        
        self.T = 0
        
        self.tracking = tracking
        self.mode = mode
        self.sum_ws = np.zeros(n)
        
        self.optimization_failed_flag = False
        self.optimization_failed_number_of_rounds = 0

    @abstractmethod
    def stopping_rule(self):
        pass

    @abstractmethod
    def optimal_w(self):
        pass

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
        if np.min(t) == 0:
            return w
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
    
    @abstractmethod
    def get_action(self):
        pass

    @abstractmethod
    def update(self, main_action, post_action, reward):
        pass
