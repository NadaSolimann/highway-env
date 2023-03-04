import gym
import highway_env
import os
import argparse
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description='Highway Multiple Intentions Example')
    parser.add_argument('--intention', default='Safe', choices=["Safe","Student","Demolition","Nasty"], help='the driver intention')
    args = parser.parse_args()
    return args


def setup_env():
    env = gym.make('highway-v0')
    env.config = {
        'duration': 60,
        'manual_control': True,
        'offroad_terminal': True,
        'collision_reward': -1,
        'lanes_count': 4
    }
    env.reset()
    return env


def get_transitions(env):
    trans = []
    done = False
    while not done:
        action = env.action_type.actions_indexes["IDLE"]
        obs, reward, done, info = env.step(action)
        trans.append(info)
        env.render()
    return trans


def add_headers(out_file):
    columns_arr = ["traj", "step", "speed", "num_collisions", "num_offroad_visits", "intention"]
    data = ''
    for i in range(len(columns_arr)):
        data += '{},'.format(columns_arr[i])
    out_file.write(data[0 : len(data) - 1] + "\n")
    return out_file


def add_traj_data(out_file, trajs, trajs_intention, traj_offset):
    for traj_idx, traj in enumerate(trajs):
        for trans_idx in range(len(traj)):
            data = '' # new transition
            data += '{},{},'.format(traj_idx + traj_offset, trans_idx) # traj_num, step_num
            data += '{},'.format(traj[trans_idx]['speed'])
            data += '{},'.format(traj[trans_idx]['num_collisions'])
            data += '{},'.format(traj[trans_idx]['num_offroad_visits'])
            data += '{},'.format(trajs_intention)
            out_file.write(data[0 : len(data) - 1] + "\n")
        print('added traj %d', traj_idx + traj_offset)


if __name__ == "__main__":
    args = parse_args()
    highway_env.register_highway_envs()

    trajs = []
    for i in range(2):
        env = setup_env()
        trans = get_transitions(env)
        trajs.append(trans)

    data_file = "generated_highway_data.csv"
    data_path = "./" + data_file
    if not os.path.exists(data_path):
        out_file = open(data_path, "wt")
        add_headers(out_file)
        out_file.close()

    out_file = open(data_path, "a+")
    try:
        traj_offset = pd.read_csv(data_file)['traj'].iloc[-1] + 1
    except:
        traj_offset = 0
    add_traj_data(out_file, trajs, args.intention, traj_offset)
    out_file.close()


    # sample of info content (updated)
    # {'speed': 29.999999004618502, 'num_collisions': 3, 'num_offroad_visits': 4}

    # ! num_collisions not perfectly working (but good enough?)
    # ! num_offroad_visist if done very fast, not registered
    # ! can't change number of lanes on road
