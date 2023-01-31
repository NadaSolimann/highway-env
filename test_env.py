import gym
import highway_env
from matplotlib import pyplot as plt
# %matplotlib inline

highway_env.register_highway_envs()

env = gym.make('highway-v0')
env.config = {
    'duration': 10,
    'manual_control': True,
    'offroad_terminal': False,
    'collision_reward': -1,
    'lanes_count': 4
}
env.reset()

done = False
while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    env.render()

    # TODO: figure out why 4 params returned, not 5? Where is truncated?
    # TODO: make session end when duration ends only  (i.e. not when collision happens)
    # TODO: make cars continue playing after collision
    # TODO: make it possible for car to go off-road
