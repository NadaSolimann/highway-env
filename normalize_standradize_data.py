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
            plt.subplot(1, len(features), i+1)
            plt.hist(df[feature + data_type], bins=50)

            plt.title(f"{feature} {data_type} Distribution")
            plt.xlabel(f"{feature} {data_type}")
            plt.ylabel(f"Frequency")
            plt.tick_params(axis='x', labelrotation=0)

        plt.tight_layout()
        plt.savefig(output_path + "data" + data_type + ".png")


if __name__ == "__main__":
    data_file = "binary_highway_data.csv"
    df = pd.read_csv(data_file)
    features = ['num_collisions', 'num_offroad_visits', 'speed']
    df = df[df['step'] == 39]
    normalize_data(df, features)
    standradize_data(df, features)

    output_path = "./plots/distributions/"
    Path(output_path).mkdir(parents=True, exist_ok=True)
    plot_distributions(df, features, output_path)
    df.to_csv("normalized_standardized_data.csv")