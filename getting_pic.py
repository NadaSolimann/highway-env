import gym
from matplotlib import pyplot as plt
import highway_env


highway_env.register_highway_envs()
env = gym.make('highway-v0')
env.config = {
    'duration': 60,
    'manual_control': True,
    'offroad_terminal': True,
    'collision_reward': -1,
    'lanes_count': 4
}
env.reset()

for _ in range(10):
    action = env.action_type.actions_indexes["IDLE"]
    obs, reward, done, info = env.step(action)
    env.render()
    plt.imshow(env.render(mode='rgb_array'))
    plt.show()

