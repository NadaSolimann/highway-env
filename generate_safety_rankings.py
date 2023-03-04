import pandas as pd


if __name__ == "__main__":
    # TODO: currently generates ranking per trans, do for traj or not needed?
    data_file = "old_generated_highway_data.csv"
    df = pd.read_csv(data_file)

    df["safety_score"] = df[["num_collisions","num_offroad_visits"]].apply(tuple,axis=1)\
                .rank(method='dense',ascending=False).astype(int)

    df.sort_values("safety_score")
    df.to_csv("ranked_data_norm.csv")