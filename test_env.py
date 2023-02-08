import gym
import highway_env
from matplotlib import pyplot as plt
# %matplotlib inline

highway_env.register_highway_envs()

env = gym.make('highway-v0')
env.config = {
    'duration': 2,
    'manual_control': True,
    'offroad_terminal': False,
    'collision_reward': -1,
    'lanes_count': 4
}
env.reset()

done = False
while not done:
    action = env.action_type.actions_indexes["IDLE"]
    obs, reward, done, info = env.step(action)
    env.render()
