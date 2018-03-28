import gym
import gym_snake
import time

env = gym.make('Snake-v0')
env.reset()
for t in range(100):
    env.render()
    time.sleep(0.05)
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    print(observation)
    if done:
        print("Episode finished after {} timesteps".format(t + 1))
        break
