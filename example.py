import sys
import gym
import gym_snake
import time
import numpy as np

env = gym.make('Snake-v0')
env.reset()
for t in range(100):
    env.render()
    time.sleep(0.05)
    # action = env.action_space.sample()
    # action = np.clip(round(np.random.normal(1, 0.4)), 0, 2)
    line = sys.stdin.readline()
    if line != '1\n' and line != '2\n':
        action = 0
    else:
        action = int(line)
    observation, reward, done, info = env.step(action)
    print(observation, reward)
    if done:
        print("Episode finished after {} timesteps".format(t + 1))
        break