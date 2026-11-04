import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import itertools
import time
import warnings
import numpy as np
from multiprocessing import Pool

from io_utils import *
from optimization import optimize_GLR
from algorithms.ASTS import ASTS
from algorithms.MuSTS import MuSTS
from algorithms.SGTS import SGTS

# Figure 4: F(w) over the simplex for the four levels of prior knowledge (n = 3)
#   python heatmaps.py --instance_index 6 --div 61 --jobs 8
# Writes results/heatmaps/instance6.json

OUT_DIR = "results/heatmaps/"


def simplex_grid(div):
    grid = []
    for w in itertools.product(np.linspace(0, 1, div), repeat=2):
        w0 = 1 - sum(w)
        if w0 >= -1e-9:
            grid.append(np.array([max(w0, 0)] + list(w)))
    return grid


def unknown_F(job):
    # F(w) of the unknown setting with the reference (true) GLR solver
    warnings.filterwarnings("ignore")
    np.random.seed(0)
    mu, A, w = job
    w = np.maximum(w, 1e-9)
    return optimize_GLR(mu, A, w, A.T @ w, alg="reference")[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance_index", type=int, default=0)
    parser.add_argument("--div", type=int, default=61)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()

    n, k, confidence, mu, A = read_instance(args.instance_index)
    assert n == 3, "the heatmaps need 3 arms"
    grid = simplex_grid(args.div)

    result = {'instance': args.instance_index, 'div': args.div, 'w': [w.tolist() for w in grid], 'F': {}}
    for name, alg in [("ASTS", ASTS(n, k, A, confidence, "G")), ("MuSTS", MuSTS(n, k, mu, confidence, "G")),
                      ("SGTS", SGTS(n, k, confidence, "G"))]:
        st = time.time()
        Ts = alg.F_on_simplex(mu, A, div=args.div)
        result['F'][name] = [Ts[tuple(w)] for w in grid]
        print(f"{name}: {time.time() - st:.0f}s")

    st = time.time()
    with Pool(args.jobs) as pool:
        result['F']['STS'] = pool.map(unknown_F, [(mu, A, w) for w in grid], chunksize=8)
    print(f"STS: {time.time() - st:.0f}s")

    save_json(OUT_DIR + f"instance{args.instance_index}.json", result)
