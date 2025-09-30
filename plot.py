import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
# This path points to the directory with your JSON files.
DATA_DIRECTORY = "results/simulation/"
INSTANCES = ["instance_31", "instance_32"]
ALGORITHMS = ["ASTS_G", "MuSTS_G", "STS_G", "SGTS_G"]
# Number of runs to read from each JSON file
RUN_COUNT = 50
# --- End of Configuration ---

def create_boxplots():
    """
    Reads simulation data from JSON files and generates faceted boxplots with mean markers.
    """
    all_data = []
    print("Starting data processing...")

    # Loop through each instance and algorithm to read the data
    for instance in INSTANCES:
        for algorithm in ALGORITHMS:
            filename = f"{instance}_{algorithm}.json"
            filepath = os.path.join(DATA_DIRECTORY, filename)

            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                ts = [run['T'] for run in data[:RUN_COUNT]]
                for t_value in ts:
                    all_data.append({
                        "Instance": instance.replace("_", " ").title(),
                        "Algorithm": algorithm.replace("_G", ""),
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
        palette="viridis",
        height=6,
        aspect=0.8,
        showmeans=True,
        meanprops={"marker":"*", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"10"}
    )

    # --- Add Mean Annotations ---
    means = df.groupby(['Instance', 'Algorithm'])['T'].mean().reset_index()

    for i, ax in enumerate(g.axes.flat):
        instance_name = df['Instance'].unique()[i]
        instance_means = means[means['Instance'] == instance_name]
        mean_map = instance_means.set_index('Algorithm')['T'].to_dict()
        xtick_labels = [label.get_text() for label in ax.get_xticklabels()]

        for j, algorithm_name in enumerate(xtick_labels):
            if algorithm_name in mean_map:
                mean_val = mean_map[algorithm_name]
                # --- MODIFIED THIS LINE ---
                ax.text(j, mean_val, f'        {mean_val:.2f}',
                        ha='left', va='center', fontsize=8, color='black', fontweight='normal')

    # Set titles and labels
    g.fig.suptitle('Algorithm Performance Comparison', y=1.03, fontsize=16)
    g.set_axis_labels("Algorithm", "Sample complexity")
    g.set_titles("{col_name}")
    g.set(yscale="log")

    # Save the plot
    output_filename = "simulation_boxplots_subtle_mean.png"
    plt.savefig(output_filename, bbox_inches='tight')
    print(f"\n✅ Successfully generated boxplots with subtle mean values and saved as '{output_filename}'")

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
    
    plt.plot(x, data['true_lambdas'], ":", label="GLR-ground trouth", color="k", linewidth=LINE_WIDTH)
    
    plt.plot(x, data['betas'], label=r"Our $\beta$", linewidth=LINE_WIDTH)
    plt.plot(x, data['beta2s'], label=r"Naive $\beta$", linewidth=LINE_WIDTH)
    
    plt.rcParams.update({'font.size': FONT_SIZE})
    plt.tick_params(axis='both', which='major', labelsize=FONT_SIZE)
    plt.xlabel("Arm pulls", fontsize=FONT_SIZE)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # create_boxplots()
    create_convergeplots("instance_7", "STS_G")