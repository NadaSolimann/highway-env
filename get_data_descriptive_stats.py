import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans


# ! not working at desired
def fix_traj_nums(df, old_filename):
    prev_traj = 0
    for i, row in df.iterrows():
        curr_traj = int(row['traj'])
        if prev_traj > curr_traj:
            df.at[i, 'traj'] = prev_traj + 1
            curr_traj = prev_traj + 1
        prev_traj = curr_traj

    # Write the updated DataFrame back to a CSV file
    df.to_csv(f'updated_{old_filename}', index=False)


def plot_feature(df, feature, agg_func, axis, colors):
    # calculate the feature aggregation of each trajectory
    trajs_grouped = df.groupby(['intention', 'traj'])
    agg_traj_speeds = trajs_grouped[feature].agg(agg_func)

    # calculate the feature aggregation of each intention
    intents_grouped = agg_traj_speeds.groupby(['intention'])
    agg_intents_speeds = intents_grouped.agg(agg_func)

    # Plot the speed aggregation of each intention with labels
    agg_intents_speeds.plot(kind='bar', ax=axis, color=colors)
    axis.set_title(f"{agg_func.title()} {feature} per Intention")
    axis.set_xlabel('Intention')
    axis.set_ylabel(f"{agg_func.title()} {feature}")
    axis.tick_params(axis='x', labelrotation=0)

    # Add labels to the bars
    for i, v in enumerate(agg_intents_speeds):
        axis.text(i, v, f"{v:.2f}", ha='center', fontsize=8)

    return axis


# TODO run kmeans on trajs stats and see if can recover clusters as intentions
if __name__ == "__main__":
    # TODO: generate data again
    data_file = "generated_highway_data.csv"
    data_df = pd.read_csv(data_file)
    # fix_traj_nums(data_df, data_file)

    hists_path = "./histograms/"
    Path(hists_path).mkdir(parents=True, exist_ok=True)
    colors = ['gold', 'salmon', 'lightgreen', 'cornflowerblue']
    
    for feature in ['speed', 'num_offroad_visits', 'num_collisions']:
        # Create a figure with subplots for each aggregation function
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

        # Call the function for each aggregation function and pass the appropriate subplot
        plot_feature(data_df, feature, 'min', axes[0], colors)
        plot_feature(data_df, feature, 'mean', axes[1], colors)
        plot_feature(data_df, feature, 'max', axes[2], colors)

        plt.tight_layout()
        plt.savefig(hists_path + feature + "_stats.png")

    # get_kmeans_clusters()