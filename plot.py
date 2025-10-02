import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils import *
from io_utils import *
from environment import *

# --- Configuration ---
# This path points to the directory with your JSON files.
DATA_DIRECTORY = "results/simulation/"
INSTANCES = ["instance_12"]
ALGORITHMS = ["ASTS_G", "MuSTS_G", "STS_G", "SGTS_G"]
ALG_NAMES = [r"GTS-$\mathcal{A}$", r"GTS-$\mu$", "GTS-Unknown", "GTS-SG"]
INS_NAMES = [r"Hard $\mu$"]
# Number of runs to read from each JSON file
RUN_COUNT = 100
# --- End of Configuration ---

def create_boxplots():
    """
    Reads simulation data from JSON files and generates faceted boxplots with mean markers.
    """
    FONT_SIZE = 16
    all_data = []    

    # Loop through each instance and algorithm to read the data
    for ins_idx, instance in enumerate(INSTANCES):
        for alg_idx, algorithm in enumerate(ALGORITHMS):
            filename = f"{instance}_{algorithm}.json"
            filepath = os.path.join(DATA_DIRECTORY, filename)

            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                ts = [run['T'] for run in data[:RUN_COUNT]]
                for t_value in ts:
                    all_data.append({
                        "Instance": INS_NAMES[ins_idx],
                        "Algorithm": ALG_NAMES[alg_idx],
                        "T": t_value
                    })
                print(f"Successfully loaded {len(ts)} runs from {filename}")
            except FileNotFoundError:
                print(f"⚠️ Warning: File not found at '{filepath}'. Skipping.")
            except (json.JSONDecodeError, KeyError) as e:
                print(f"❌ Error reading {filepath}: {e}. Skipping.")

    if not all_data:
        print("\nNo data was loaded. Cannot generate plot.")
        return

    df = pd.DataFrame(all_data)

    # Create faceted boxplots using seaborn's catplot
    g = sns.catplot(
        data=df,
        x="Algorithm",
        y="T",
        col="Instance",
        kind="box",
        palette="deep",
        height=6,
        aspect=1,
        showmeans=True,
        meanprops={"marker":"*", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":FONT_SIZE}
    )
    
    # Set axis labels font sizes
    g.set_xlabels(size=FONT_SIZE-1)
    g.set_ylabels(size=FONT_SIZE-1)

    # Optional: Set tick label sizes too
    g.set_xticklabels(size=FONT_SIZE-2)  # Slightly smaller than titles
    g.set_yticklabels(size=FONT_SIZE-2)

    # --- Add Mean Annotations ---
    # means = df.groupby(['Instance', 'Algorithm'])['T'].mean().reset_index()

    # for i, ax in enumerate(g.axes.flat):
    #     instance_name = df['Instance'].unique()[i]
    #     instance_means = means[means['Instance'] == instance_name]
    #     mean_map = instance_means.set_index('Algorithm')['T'].to_dict()
    #     xtick_labels = [label.get_text() for label in ax.get_xticklabels()]

    #     for j, algorithm_name in enumerate(xtick_labels):
    #         if algorithm_name in mean_map:
    #             mean_val = mean_map[algorithm_name]
    #             # --- MODIFIED THIS LINE ---
    #             ax.text(j, mean_val, f'        {mean_val:.2f}',
    #                     ha='left', va='center', fontsize=8, color='black', fontweight='normal')

    # Set titles and labels
    g.set_axis_labels("Algorithm", "Sample complexity", fontsize=FONT_SIZE)
    g.set_titles("{col_name}", size=FONT_SIZE)
    g.set(yscale="log")

    plt.show()
    
def create_convergeplots(instance, algorithm, log_period=10):
    filename = f"{instance}_{algorithm}.json"
    filepath = os.path.join(DATA_DIRECTORY, filename)

    try:
        with open(filepath, "r") as f:
            data = json.load(f)[-1]
        print(f"Successfully loaded from {filename}")
    except FileNotFoundError:
        print(f"⚠️ Warning: File not found at '{filepath}'. Skipping.")
    except (json.JSONDecodeError, KeyError) as e:
        print(f"❌ Error reading {filepath}: {e}. Skipping.")
    
    LINE_WIDTH = 4
    FONT_SIZE = 18
    plt.figure(figsize=(10, 6))
    x = range(0, len(data['lambdas']) * log_period, log_period)
    plt.plot(x, data['lambdas'], label="GLR-heuristic", linewidth=LINE_WIDTH)
    plt.plot(x, data['lambda_lbs'], label="GLR-relaxation", linewidth=LINE_WIDTH)
    
    plt.plot(x, data['true_lambdas'], ":", label="GLR-ground truth", color="k", linewidth=LINE_WIDTH)
    
    plt.plot(x, data['betas'], label=r"Our $\beta$", linewidth=LINE_WIDTH)
    plt.plot(x, data['beta2s'], label=r"Naive $\beta$", linewidth=LINE_WIDTH)
    
    plt.rcParams.update({'font.size': FONT_SIZE})
    plt.tick_params(axis='both', which='major', labelsize=FONT_SIZE)
    plt.xlabel("timestep t", fontsize=FONT_SIZE)
    plt.legend()
    plt.show()

def create_simplex_heatmaps(instance, div=101):
    instance_path = f"instances/{instance}.json"
    n, k, confidence, mus, A = read_instance_from_json(instance_path)
    
    fig, axis = plt.subplots(1, 4, figsize=(32, 8))
    
    FONT_SIZE = 24
    print("plotting ASTS ...")
    alg = ASTS(n, k, A, confidence, "G")
    alg.plot_w(mus, A, div=div, ax=axis[0])
    axis[0].set_title('ASTS', fontsize=FONT_SIZE)

    print("plotting MuSTS ...")    
    alg = MuSTS(n, k, mus, confidence, "G")
    alg.plot_w(mus, A, div=div, ax=axis[1])
    axis[1].set_title('MuSTS', fontsize=FONT_SIZE)
    
    print("plotting STS ...")
    alg = STS(n, k, confidence, "G")
    alg.plot_w(mus, A, div=div, ax=axis[2])
    axis[2].set_title('STS', fontsize=FONT_SIZE)
    
    print("plotting SGTS ...")
    alg = SGTS(n, k, confidence, "G")
    alg.plot_w(mus, A, div=div, ax=axis[3])
    axis[3].set_title('SGTS', fontsize=FONT_SIZE)
    
    plt.show()

if __name__ == "__main__":
    # create_boxplots()
    create_convergeplots("instance_8", "STS_G")
    # create_simplex_heatmaps("instance10")