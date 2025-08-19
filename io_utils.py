import json
from pathlib import Path
import argparse
import numpy as np

def get_arguments():
    parser = argparse.ArgumentParser(description="Parse variables simulate")
    
    parser.add_argument("--Algorithm", type=str, required=False, default='STS', help="Description for var1 (e.g., a string variable)")
    parser.add_argument("--instance_index", type=int, required=False, default=0)
    parser.add_argument("--store", type=bool, required=False, default=True)
    parser.add_argument("--use_optimized_p", type=bool, required=False, default=False) 
    parser.add_argument("--average_points_played", type=bool, required=False, default=False) 
    parser.add_argument("--average_w", type=bool, required=False, default=False)
    parser.add_argument("--stopping_rule", type=str, required=False, default='d')
    
    return parser.parse_args()


def get_optimization_arguments():
    parser = argparse.ArgumentParser(description="Parse variables for optimization", add_help=True)
    
    parser.add_argument("--instance_index", type=int, default=None)
    parser.add_argument("--experiment_cnt", type=int, default=None)
    parser.add_argument("--fixed_w", type=bool, default=False)
    parser.add_argument("--name", type=str, default="grid")
    
    return parser.parse_args()


def add_instance_to_json(n, k, confidence, A, mus, w_star, T_star, file_path = None):
    DIR_PATH = "instances/"
    if file_path is None:
        file_path = f"{DIR_PATH}instance{len(Path(DIR_PATH).glob('*.json'))}.json"
    else:
        file_path = DIR_PATH + file_path

    A_list = A.tolist()
    mus_list = mus.tolist()
    w_star_list = w_star.tolist()

    instance = {
        "n": n,
        "k": k,
        "confidence": confidence,
        "mus": mus_list,
        "A": A_list,
        "w_star": w_star_list,
        "T_star": T_star
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
                np.array(instance["A"]),
                np.array(instance["w_star"]),
                instance["T_star"],
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
        w_star_list = []
        T_star_list = []
        
        for json_file in Path(dir_path).glob('*.json'):
            with open(json_file, 'r', encoding='utf-8') as f:
                instance = json.load(f)
                n_list.append(instance["n"])
                k_list.append(instance["k"])
                delta_list.append(instance["delta"])
                means_list.append(np.array(instance["means"]))
                contexts_list.append(np.array(instance["contexts"]))
                w_star_list.append(np.array(instance["w_star"]))
                T_star_list.append(instance["T_star"])
        
        return n_list, k_list, delta_list, means_list, contexts_list, w_star_list, T_star_list
            
    except FileNotFoundError as err:
        print(err)
        return [], [], [], [], [], [], []


def add_output_to_json(instance_number, args, mu_hats, N_times_seens, w_s, T, best_arm):
    path = f'results/instance_{instance_number}_'
    
    path += args.Algorithm
    
    if args.Algorithm == 'STS':
        if args.use_optimized_p:
            path += '_optimizedPTrue'
        else:
            if args.average_w:
                path += '_averagedWTrue'
            if args.average_points_played:
                path += '_averagePointsPlayedTrue'
    
    if args.Algorithm == 'STS_C_Tracking':
        if args.stopping_rule == 'c_stopping_rule':
            path += '_CStoppingRule'

    instance = {
        "mu_hats": mu_hats,
        "N_times_seens": N_times_seens,
        "w_s": w_s,
        "T": T,
        "best_arm": int(best_arm)
    }
    
    path += '.json'
    
    try:
        with open(path, 'r') as file:
            data = json.load(file)
    except FileNotFoundError as err:
        print(err)
        data = []
    
    data.append(instance)
    
    with open(path, 'w') as file:
        json.dump(data, file, indent=4)


def read_outputs_from_json(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            
            # Prepare lists for each parameter
            mu_hat_list = []
            N_times_seens_list = []
            T_list = []
            w_s_list = []
            best_arm_list = []
           
            # Populate the lists with data from the JSON file
            for instance in data:
                T_list.append(instance["T"])
                best_arm_list.append(instance["best_arm"])
                w_s_list.append(instance["w_s"])
                mu_hat_list.append(instance["mu_hats"])
                N_times_seens_list.append(instance["N_times_seens"])
            
            return mu_hat_list, N_times_seens_list, T_list, w_s_list, best_arm_list
            
    except FileNotFoundError as err:
        print(err)
        return [], [], [], []
