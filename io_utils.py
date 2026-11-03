import json
import os
from pathlib import Path
import argparse
import numpy as np

INSTANCE_DIR = "instances/"


def get_arguments():
    parser = argparse.ArgumentParser(description="Runs an algorithm several times on an instance and stores the results")

    parser.add_argument("--algorithm", type=str, nargs="+", required=True, help="some of ASTS, MuSTS, STS-LB, STS-Plain, SGTS")
    parser.add_argument("--instance_index", type=int, nargs="+", required=True)
    parser.add_argument("--runs", type=int, default=1, help="total number of runs wanted in the result file")
    parser.add_argument("--jobs", type=int, default=1, help="number of runs in parallel")
    parser.add_argument("--delta", type=float, default=None, help="overrides the confidence of the instance")
    parser.add_argument("--out_dir", type=str, default="results/simulation/")
    parser.add_argument("--tracking", type=str, default='G')
    parser.add_argument("--no_fast", action="store_true", help="recompute the optimal w at every round")
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


def get_optimization_arguments():
    parser = argparse.ArgumentParser(description="Parse variables for optimization")

    parser.add_argument("-n", type=int, default=2)
    parser.add_argument("-k", type=int, default=2)
    parser.add_argument("--experiment_cnt", type=int, default=None)
    parser.add_argument("--fixed_w", action="store_true")
    parser.add_argument("--name", type=str, default="plain")

    return parser.parse_args()


def add_instance_to_json(n, k, confidence, A, mus, file_path = None):
    if file_path is None:
        file_path = f"{INSTANCE_DIR}instance{len(list(Path(INSTANCE_DIR).glob('*.json')))}.json"
    else:
        file_path = INSTANCE_DIR + file_path

    instance = {
        "n": n,
        "k": k,
        "confidence": confidence,
        "mus": mus.tolist(),
        "A": A.tolist()
    }

    with open(file_path, 'w') as file:
        json.dump(instance, file, indent=4)


def read_instance_from_json(file_path):
    with open(file_path, 'r') as file:
        instance = json.load(file)
        return (
            instance["n"],
            instance["k"],
            instance["confidence"],
            np.array(instance["mus"]),
            np.array(instance["A"])
        )


def read_instance(index):
    return read_instance_from_json(f"{INSTANCE_DIR}instance{index}.json")


def load_json(path, default=None):
    try:
        with open(path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return default


def save_json(path, data):
    # write to a temporary file first so an interrupted run never corrupts the results
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp_path = str(path) + ".tmp"
    with open(tmp_path, 'w') as file:
        json.dump(data, file, indent=1)
    os.replace(tmp_path, path)


def simulation_path(instance_index, algorithm, out_dir="results/simulation/"):
    return os.path.join(out_dir, f"instance{instance_index}_{algorithm}.json")
