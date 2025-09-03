import numpy as np
import matplotlib.pyplot as plt
import time

from utils import *
from io_utils import *
from environment import *

def simulate(verbose=True):
    args = get_arguments()
    index = args.instance_index
    
    instance_path = f"instances/instance{index}.json"
    n, k, confidence, mus, A = read_instance_from_json(instance_path)
    
    mode = {
        'average_w': args.average_w,
    }

    st_time = time.time()
    env = Environment(mus, A, args.Algorithm, args.tracking, n, k, confidence, mode=mode)
    best_arm, mu_hats, N_times_seens, w_s, lambda_lbs, lambdas, betas, T = env.loop()
    fn_time = time.time()
    
    if verbose:
        x = range(0, len(lambdas) * env.log_period, env.log_period)
        plt.plot(x, lambda_lbs, label="lambda lower bounds")
        plt.plot(x, lambdas, label="lambdas (upper bound)")
        plt.plot(x, betas, label="stopping threshold")
        plt.legend()
        plt.show()
    
    if args.store:
        add_output_to_json(index, args, mu_hats, N_times_seens, w_s, T, best_arm)
    
    print("#"*60)
    print("The number of time steps is:", T)
    print("The actual best arm is:", np.argmax(np.dot(A, mus)))
    print("The best arm identified is:", best_arm)
    print(f"Simulation process took {fn_time - st_time} seconds.")
    print("#"*60)
    

if __name__ == "__main__":
    simulate()