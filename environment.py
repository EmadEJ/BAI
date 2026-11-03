import time
import numpy as np

from algorithms.STS import *
from algorithms.ASTS import *
from algorithms.MuSTS import *
from algorithms.SGTS import *
from utils import *

# names used on the command line and in the result files, with the label used in the paper
ALGORITHMS = {
    "ASTS": r"GTS-$\mathcal{A}$",
    "MuSTS": r"GTS-$\mu$",
    "STS-LB": "GTS-Unknown (LB)",
    "STS-Plain": "GTS-Unknown (solver)",
    "SGTS": "GTS-SG",
}


class Environment:

    def __init__(self, mus, A, n, k, seed=None):

        self.mus = mus  # mean reward of each context
        self.A = A  # context given arm probabilty matrix
        self.n = n
        self.k = k
        self.T = 0

        self.seed = seed
        self.rng = np.random.default_rng(seed)
        np.random.seed(seed)  # the algorithms use np.random for the forced exploration
        self.cum_A = np.cumsum(A, axis=1)

        self.means = np.dot(self.A, self.mus)
        self.best_arm = np.argmax(self.means)
        self.delta = self.means[self.best_arm] - self.means

    def take_action(self, action):
        post_action = min(np.searchsorted(self.cum_A[action], self.rng.random(), side='right'), self.k - 1)
        reward = self.rng.normal() + self.mus[post_action]
        self.T += 1
        return post_action, reward

    def make_algorithm(self, algorithm, confidence, tracking, mode):
        if algorithm == "STS-LB":
            return STS(self.n, self.k, confidence, tracking, mode, glr="lb")
        if algorithm == "STS-Plain":
            return STS(self.n, self.k, confidence, tracking, mode, glr="plain")
        if algorithm == "ASTS":
            return ASTS(self.n, self.k, self.A, confidence, tracking, mode)
        if algorithm == "MuSTS":
            return MuSTS(self.n, self.k, self.mus, confidence, tracking, mode)
        if algorithm == "SGTS":
            return SGTS(self.n, self.k, confidence, tracking, mode)
        raise ValueError(f"Invalid Algorithm {algorithm}!")

    def run(self, confidence, algorithm, tracking="G", mode={'fast': True}, verbose=False, log_period=None, max_T=10**8):
        """
        Runs one algorithm until it stops. With log_period, the sufficient statistics of the algorithm
        are stored every log_period rounds (only for STS, used for figure 2).
        """
        alg = self.make_algorithm(algorithm, confidence, tracking, mode)
        states = []
        time_sampling, time_stopping = 0.0, 0.0

        start = time.time()
        while self.T < max_T:
            # Select an action using the algorithm
            st = time.time()
            action, init = alg.get_action()
            time_sampling += time.time() - st

            post_action, reward = self.take_action(action)
            alg.update(action, post_action, reward)
            if init:
                continue

            st = time.time()
            stop, lambda_t, beta_t = alg.stopping_rule()
            time_stopping += time.time() - st

            if log_period is not None and (self.T % log_period == 0 or stop):
                states.append(alg.state())
            if verbose and self.T % 1000 == 0:
                print(f"Round {self.T}, lambda_hat_t: {lambda_t:.3f}, beta_t: {beta_t:.3f}, w: {alg.last_w}")
            if stop:
                break

        best_arm = int(alg.best_empirical_arm()[0])
        result = {
            'seed': self.seed,
            'T': self.T,
            'best_arm': best_arm,
            'correct': best_arm == int(self.best_arm),
            'stopped': bool(self.T < max_T),
            'time': time.time() - start,
            'time_sampling': time_sampling,
            'time_stopping': time_stopping,
            'n_optimizations': alg.n_optimizations,
            'n_failed_optimizations': alg.optimization_failed_number_of_rounds,
        }
        if log_period is not None:
            result['states'] = states
        return result
