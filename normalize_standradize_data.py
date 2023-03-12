import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def normalize_data(df, features):
    for feature in features:
        max_val = df[feature].max()
        df[feature + '_normalized'] = df[feature] / max_val


def standradize_data(df, features):
    mean = df.mean()
    std = df.std()
    standradized = (df - mean) / std

    for feature in ['num_collisions', 'num_offroad_visits', 'speed']:
        df[feature + '_standradized'] = standradized[feature]


def plot_distributions(df, features, output_path):
    for data_type in ["", "_normalized", "_standradized"]:
        fig, axes = plt.subplots(nrows=1, ncols=len(features), figsize=(12, 4))
        for i, feature in enumerate(features):
            df[feature + data_type].plot(kind='kde', ax=axes[i])

            axes[i].set_title(f"{feature} {data_type} Distribution")
            axes[i].set_xlabel(f"{feature} {data_type}")
            axes[i].set_ylabel(f"Frequency")
            axes[i].tick_params(axis='x', labelrotation=0)

        plt.tight_layout()
        plt.savefig(output_path + "data" + data_type + ".png")


if __name__ == "__main__":
    data_file = "old_generated_highway_data.csv"
    df = pd.read_csv(data_file)
    features = ['num_collisions', 'num_offroad_visits', 'speed']
    normalize_data(df, features)
    standradize_data(df, features)

    output_path = "./plots/distributions/"
    Path(output_path).mkdir(parents=True, exist_ok=True)
    plot_distributions(df, features, output_path)
    df.to_csv("normalized_standardized_data.csv")