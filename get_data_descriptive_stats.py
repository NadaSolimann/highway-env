import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans


def plot_feature(df, feature, agg_func, axis, colors):
    # calculate the feature aggregation of each trajectory
    trajs_grouped = df.groupby(['intention', 'traj'])
    agg_traj_feature = trajs_grouped[feature].agg(agg_func)

    # calculate the feature aggregation of each intention
    intents_grouped = agg_traj_feature.groupby(['intention'])
    agg_intents_feature = intents_grouped.agg(agg_func)

    # Plot the speed aggregation of each intention with labels
    agg_intents_feature.plot(kind='bar', ax=axis, color=colors)
    axis.set_title(f"{agg_func.title()} {feature} per Intention")
    axis.set_xlabel('Intention')
    axis.set_ylabel(f"{agg_func.title()} {feature}")
    axis.tick_params(axis='x', labelrotation=0)

    # Add labels to the bars
    for i, v in enumerate(agg_intents_feature):
        axis.text(i, v, f"{v:.2f}", ha='center', fontsize=8)

    return axis


def plot_scatterplot(df, agg_func, axis, colors, cum_data):
    trajs_grouped = df.groupby(['intention', 'traj'])
    features = trajs_grouped["num_collisions", "num_offroad_visits"]
    agg_traj_feature = features.agg(agg_func)
    if not cum_data:
        cumulative_state = features.sum()
        if (agg_func == 'mean'):
            agg_traj_feature = cumulative_state / features.count()
        elif (agg_func == 'max'):
            agg_traj_feature = cumulative_state

    intents_grouped = agg_traj_feature.groupby(['intention'])

    for intent, data in intents_grouped:
        axis.plot(data["num_collisions"], data["num_offroad_visits"],
        marker='o', linestyle='', ms=5, label=intent, color=colors[intent])
        print(f"passed")
    axis.legend()
    axis.set_title(f"{agg_func.title()} Data per Intention")
    axis.set_xlabel('num_collisions')
    axis.set_ylabel('num_offroad_visits')


if __name__ == "__main__":
    # data_file = "cumulative_highway_data.csv"
    data_file = "binary_highway_data.csv"
    data_df = pd.read_csv(data_file)

    plots_path = "./plots/"
    Path(plots_path).mkdir(parents=True, exist_ok=True)
    colors = {
        "Safe" : "lightgreen",
        "Student" : "cornflowerblue",
        "Demolition" : "gold",
        "Nasty" : "salmon"
    }

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
    plot_scatterplot(data_df, 'min', axes[0], colors, False) # cumulative tag
    plot_scatterplot(data_df, 'mean', axes[1], colors, False)
    plot_scatterplot(data_df, 'max', axes[2], colors, False)
    plt.tight_layout()
    plt.savefig(plots_path + "bin_data_scatterplot.png")


    # for feature in ['speed', 'num_offroad_visits', 'num_collisions']:
    #     # Create a figure with subplots for each aggregation function
    #     fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

    #     # Call the function for each aggregation function and pass the appropriate subplot
    #     plot_feature(data_df, feature, 'min', axes[0], colors)
    #     plot_feature(data_df, feature, 'mean', axes[1], colors)
    #     plot_feature(data_df, feature, 'max', axes[2], colors)

    #     plt.tight_layout()
    #     plt.savefig(hists_path + feature + "_stats.png")

    # get_kmeans_clusters(data_df)