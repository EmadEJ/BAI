import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import time
from tqdm import tqdm

from utils import *
from io_utils import *
from environment import *

def simulate(verbose=False):
    args = get_arguments()
    index = args.instance_index
    
    instance_path = f"instances/instance{index}.json"
    n, k, confidence, mus, A = read_instance_from_json(instance_path)
    mode = {
        'average_w': args.average_w,
    }
    best_arm = np.argmax(np.dot(A, mus))

    if args.cnt == 1:  
        st_time = time.time()
        env = Environment(mus, A, n, k)
        result = env.run(confidence, args.algorithm, args.tracking, mode, verbose=True)
        fn_time = time.time()
        
        if verbose:
            x = range(0, len(result['lambdas']) * env.log_period, env.log_period)
            if args.algorithm == "STS":
                plt.plot(x, result['lambda_lbs'], label="lambda lower bounds")
                plt.plot(x, result['beta2s'], label="union bound stopping threshold")
            plt.plot(x, result['lambdas'], label="lambdas")
            plt.plot(x, result['betas'], label="stopping threshold")
            plt.xlabel("Arm pulls")
            plt.ylabel("GLR value")
            plt.legend()
            plt.show()
        
        if args.store:
            add_output_to_json(index, args, result)
        
        print("#"*60)
        print("The number of arm pulls is:", result['T'])
        print("The actual best arm is:", best_arm)
        print("The best arm identified is:", result['best_arm'])
        print(f"Simulation process took {fn_time - st_time} seconds.")
        print("#"*60)
        
    else:
        st_time = time.time()
        
        Ts = []
        best_arms = []
        for _ in tqdm(range(args.cnt), desc="Simulations"):
            env = Environment(mus, A, n, k)
            result = env.run(confidence, args.algorithm, args.tracking, mode)
            
            Ts.append(result['T'])
            best_arms.append(result['best_arm'])
            
            if args.store:
                print(args.store)
                add_output_to_json(index, args, result)
            
        fn_time = time.time()
        
        print("#"*60)
        print("Average number of arm pulls is:", np.mean(Ts))
        print("Correctly identified ratio:", best_arms.count(best_arm) / args.cnt)
        print("The actual best arm is:", best_arm)
        print(f"Simulation process took {fn_time - st_time} seconds.")
        print("#"*60)
        
        fig = px.box(x=Ts, title=f"{args.algorithm} arm pulls on instance {index}")
        fig.update_layout(xaxis_title="Arm Pulls")
        fig.show()
    

if __name__ == "__main__":
    simulate(verbose=True)