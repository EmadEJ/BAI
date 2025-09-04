import numpy as np
import matplotlib.pyplot as plt
import time

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

    st_time = time.time()
    env = Environment(mus, A, n, k)
    result = env.run(confidence, args.algorithm, args.tracking, mode)
    fn_time = time.time()
    
    if verbose:
        x = range(0, len(result['lambdas']) * env.log_period, env.log_period)
        if args.algorithm == "STS":
            plt.plot(x, result['lambda_lbs'], label="lambda lower bounds")
        plt.plot(x, result['lambdas'], label="lambdas")
        plt.plot(x, result['betas'], label="stopping threshold")
        plt.legend()
        plt.show()
    
    if args.store:
        add_output_to_json(index, args, result)

    print("#"*60)
    print("The number of arm pulls is:", result['T'])
    print("The actual best arm is:", np.argmax(np.dot(A, mus)))
    print("The best arm identified is:", result['best_arm'])
    print(f"Simulation process took {fn_time - st_time} seconds.")
    print("#"*60)
    

if __name__ == "__main__":
    simulate(verbose=True)