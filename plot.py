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

    # This section checks for the data directory and creates dummy data if it's missing.
    if not os.path.exists(DATA_DIRECTORY):
        os.makedirs(DATA_DIRECTORY)
        for instance in INSTANCES:
            for algorithm in ALGORITHMS:
                filename = f"{instance}_{algorithm}.json"
                filepath = os.path.join(DATA_DIRECTORY, filename)
                # Generating some sample data for demonstration purposes
                if algorithm == 'ASTS_G':
                    dummy_data = [{'T': 100 + i*2 + (i%5 - 2)*5} for i in range(RUN_COUNT)]
                elif algorithm == 'MuSTS_G':
                    dummy_data = [{'T': 200 + i*5 + (i%7 - 3)*10} for i in range(RUN_COUNT)]
                else: # SGTS_G
                    dummy_data = [{'T': 300 + i*10 + (i%10 - 5)*15} for i in range(RUN_COUNT)]
                with open(filepath, 'w') as f:
                    json.dump(dummy_data, f)

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

if __name__ == "__main__":
    create_boxplots()