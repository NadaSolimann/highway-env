import pandas as pd


if __name__ == "__main__":
    # ! currently generates ranking for each trans, do for traj or not needed?
    # ! why needed to stand/norm before ranking, ranking is unchanged
    data_file = "generated_highway_data.csv"
    df = pd.read_csv(data_file)

    df["safety_score"] = df[["num_collisions", "num_offroad_visits"]]\
        .apply(tuple,axis=1).rank(method='dense',ascending=False).astype(int)

    # for data_type in ["", "_normalized", "_standradized"]:
    #     num_collisions = "num_collisions" + data_type
    #     num_offroad_visits = "num_offroad_visits" + data_type

    #     df["safety_score" + data_type] = df[[num_collisions, num_offroad_visits]]\
    #         .apply(tuple,axis=1).rank(method='dense',ascending=False).astype(int)

    df.to_csv("data_safety_rankings.csv")