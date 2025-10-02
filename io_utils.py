import json
from pathlib import Path
import argparse
import numpy as np

def get_arguments():
    parser = argparse.ArgumentParser(description="Parse variables simulate")
    
    parser.add_argument("--algorithm", type=str, required=False, default='STS')
    parser.add_argument("--instance_index", type=int, required=False, default=0)
    parser.add_argument("--store", type=bool, required=False, default=False)
    parser.add_argument("--tracking", type=str, required=False, default='G')
    parser.add_argument("--average_w", type=bool, required=False, default=False)
    parser.add_argument("--fast", type=bool, required=False, default=False)
    parser.add_argument("--detailed", type=bool, required=False, default=False)
    parser.add_argument("--cnt", type=int, required=False, default=1)
    
    return parser.parse_args()


def get_optimization_arguments():
    parser = argparse.ArgumentParser(description="Parse variables for optimization")
    
    parser.add_argument("-n", type=int, default=2)
    parser.add_argument("-k", type=int, default=2)
    parser.add_argument("--instance_index", type=int, default=None)
    parser.add_argument("--experiment_cnt", type=int, default=None)
    parser.add_argument("--fixed_w", type=bool, default=False)
    parser.add_argument("--name", type=str, default="grid")
    
    return parser.parse_args()


def add_instance_to_json(n, k, confidence, A, mus, file_path = None):
    DIR_PATH = "instances/"
    if file_path is None:
        file_path = f"{DIR_PATH}instance{len(Path(DIR_PATH).glob('*.json'))}.json"
    else:
        file_path = DIR_PATH + file_path

    A_list = A.tolist()
    mus_list = mus.tolist()

    instance = {
        "n": n,
        "k": k,
        "confidence": confidence,
        "mus": mus_list,
        "A": A_list
    }

    with open(file_path, 'w') as file:
        json.dump(instance, file, indent=4)


def read_instance_from_json(file_path):
    try:
        with open(file_path, 'r') as file:
            instance = json.load(file)
            return (
                instance["n"],
                instance["k"],
                instance["confidence"],
                np.array(instance["mus"]),
                np.array(instance["A"])
            )
            
    except FileNotFoundError as err:
        print(err)
        return {}


def read_all_instances_from_json(dir_path = 'instances/'):
    try:
        # Prepare lists for each parameter
        n_list = []
        k_list = []
        delta_list = []
        means_list = []
        contexts_list = []
        
        for json_file in Path(dir_path).glob('*.json'):
            with open(json_file, 'r', encoding='utf-8') as f:
                instance = json.load(f)
                n_list.append(instance["n"])
                k_list.append(instance["k"])
                delta_list.append(instance["delta"])
                means_list.append(np.array(instance["means"]))
                contexts_list.append(np.array(instance["contexts"]))
        
        return n_list, k_list, delta_list, means_list, contexts_list
            
    except FileNotFoundError as err:
        print(err)
        return [], [], [], [], [], [], []


def add_output_to_json(instance_number, args, result):
    path = f'results/simulation/instance_{instance_number}_'

    path += args.algorithm + "_" + args.tracking
    path += '.json'
    
    try:
        with open(path, 'r') as file:
            data = json.load(file)
    except FileNotFoundError as err:
        print(err)
        data = []
    
    data.append(result)
    
    with open(path, 'w') as file:
        json.dump(data, file, indent=4)


def read_outputs_from_json(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            
            # Prepare lists for each parameter
            w_s_list = []
            T_list = []
            best_arm_list = []
           
            # Populate the lists with data from the JSON file
            for instance in data:
                T_list.append(instance["T"])
                best_arm_list.append(instance["best_arm"])
                w_s_list.append(instance["w_s"])

            return w_s_list, T_list, best_arm_list

    except FileNotFoundError as err:
        print(err)
        return [], [], [], [], [], [], []
