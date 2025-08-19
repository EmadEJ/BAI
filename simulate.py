import numpy as np
from utils import *
from io_utils import *
from environment import *
import matplotlib.pyplot as plt
import time

def simulate(verbose=True):
    args = get_arguments()
    index = args.instance_index
    
    instance_path = f"instances/instance{index}.json"
    n, k, confidence, mus, A, w_star, T_star = read_instance_from_json(instance_path)
    
    mode = {
        'use_optimized_p': args.use_optimized_p,
        'average_w': args.average_w,
        'average_points_played': args.average_points_played
    }

    st_time = time.time()
    env = Environment(mus, A, args.Algorithm, n, k, confidence, mode=mode, stopping_rule=args.stopping_rule)
    best_arm, mu_hats, N_times_seens, w_s, lambdas, betas, T = env.loop()
    fn_time = time.time()
    
    plt.plot(range(len(lambdas)), lambdas)
    plt.plot(range(len(betas)), betas)
    plt.show()
    
    if args.store:
        add_output_to_json(index, args, mu_hats, N_times_seens, w_s, T, best_arm)
    
    print("The number of time steps is :", T)
    print("The actual best arm is :", np.argmax(np.dot(A, mus)))
    print("The best arm identified is :", best_arm)
    print(f"Process took {fn_time - st_time} seconds.")
    print("#"*60)
    

if __name__ == "__main__":
    simulate()