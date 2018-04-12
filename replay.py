from os import path
from sys import argv
import gym
import gym_snake
import time
import numpy as np
import model_utils
from memory import Episode


def replay(model_path, num_games, timestamps_delay):
    env = gym.make('Snake-v0')
    model = model_utils.load_keras_model(model_path)
    for i in range(num_games):
        env.reset()
        state = env.current_state()
        cum_reward = 0
        done = False
        episode_memory = Episode(state, num_last_frames=4)
        num_timestep_without_improve = 0
        while not done and num_timestep_without_improve < 50:
            env.render()
            time.sleep(timestamps_delay)
            input_data = np.array([episode_memory.get_last_state()])
            prediction = model.predict(input_data, batch_size=1)[0]
            action = np.argmax(prediction)
            next_state, reward, done, info = env.step(action)
            episode_memory.add_timestep(next_state, action, reward, done)
            if cum_reward + reward < cum_reward:
                num_timestep_without_improve += 1
            else:
                num_timestep_without_improve = 0
            cum_reward += reward


def main():
    if len(argv) != 4:
        print("Incorrect number of arguments. Usage: python replay <model path> <num games> <timestamps delay>")
        exit()

    model_path = argv[1]
    num_games = int(argv[2])
    timestamps_delay = float(argv[3]) / 1000

    if path.exists(model_path + '.h5') and path.exists(
                    model_path + '.yaml') and num_games > 0 and timestamps_delay > 0.0:
        replay(model_path, num_games, timestamps_delay)
    else:
        print("Incorrect values of parameters!")


if __name__ == '__main__':
    main()
