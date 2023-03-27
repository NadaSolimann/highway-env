import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans


def plot_feature(df, feature, agg_func, axis, colors, cum_data):
    # calculate the feature aggregation of each trajectory
    trajs_grouped = df.groupby(['intention', 'traj'])

    if not cum_data:
        cumulative_state = feature.sum()
        if (agg_func == 'mean'):
            agg_traj_feature = cumulative_state / feature.count()
        elif (agg_func == 'max'):
            agg_traj_feature = cumulative_state
    if not cum_data:
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


def calculate_state_action_probs(df, states, actions):
    # Define a dictionary to hold the counts for each state-action pair
    counts = {state: {action: 0 for action in actions} for state in states}

    # Loop over your data and count the number of times each action was taken in each state
    for row in df.itertuples():
        num_collisions = row.num_collisions
        num_offroad = row.num_offroad_visits
        action = row.action
        state = ''
        if num_collisions > 0 and num_offroad == 0:
            state = 'collisions_only'
        elif num_collisions == 0 and num_offroad > 0:
            state = 'offroad_only'
        elif num_collisions > 0 and num_offroad > 0:
            state = 'collisions_and_offroad'
        else:
            state = 'no_collisions_no_offroad'
        counts[state][action] += 1

    # Compute the probabilities for each state-action pair
    probs = {}
    for state in states:
        total = sum(counts[state].values())
        probs[state] = {action: count / total
            for action, count in counts[state].items()}
    return probs


def plot_policy_distribution(df, plots_path):
    actions = [1, -1, 0]
    actions_labels = ["Left", "Right", "Stay"]
    states = ['collisions_only', 'offroad_only', 'collisions_and_offroad', 'no_collisions_no_offroad']

    action_probs = calculate_state_action_probs(df, states, actions)
    fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(12, 8), sharey=True)

    # Loop over each state and action pair and plot a histogram of action probabilities on the corresponding subplot
    for i, state in enumerate(states):
        for j, action in enumerate(actions):
            row = i
            col = j
            
            # Get the probability of taking this action in this state
            prob = action_probs[state][action]
            
            # Plot the histogram of action probabilities on this subplot
            axs[row, col].bar([action], [prob])
            axs[row, col].set_xticks(actions)
            axs[row, col].set_xticklabels(actions_labels)
            axs[row, col].set_xlim(action - 0.5, action + 0.5)
            axs[row, col].set_ylim(0, 1)
            axs[row, col].set_title(state)
            axs[row, col].text(action, prob, f'{prob:.2f}', ha='center', va='bottom')

    fig.suptitle('Distribution over actions for each state')
    plt.tight_layout()
    plt.savefig(plots_path + "data_policy_pi.png")


if __name__ == "__main__":
    # data_file = "cumulative_highway_data.csv"
    data_file = "binary_highway_data.csv"
    data_df = pd.read_csv(data_file)
    data_df = data_df.dropna()

    plots_path = "./plots/"
    Path(plots_path).mkdir(parents=True, exist_ok=True)
    plot_policy_distribution(data_df, plots_path)
    exit(-1)

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


    for feature in ['speed', 'num_offroad_visits', 'num_collisions']:
        # Create a figure with subplots for each aggregation function
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

        # Call the function for each aggregation function and pass the appropriate subplot
        plot_feature(data_df, feature, 'min', axes[0], colors)
        plot_feature(data_df, feature, 'mean', axes[1], colors)
        plot_feature(data_df, feature, 'max', axes[2], colors)

        plt.tight_layout()
        plt.savefig(plots_path + "/distributions/" + feature + "_stats.png")
