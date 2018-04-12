from sys import argv
import gym
import os

import gym_snake
from keras.layers import Conv2D, Activation, Flatten, Dense
from keras.optimizers import RMSprop
from keras.models import Sequential, model_from_yaml
import matplotlib.pyplot as plt

import numpy as np
from time import strftime

from memory import ExperienceMemory, Episode
from model_utils import save_model


class QLearning(object):
    def __init__(self, environment, num_episodes, p_range_exploration=(1.0, 0.1), exploration_phase=0.6, alpha=0.8,
                 gammma=0.8, num_last_frames=4, train_batch=100, experience_memory_size=1000, num_stored_models=1,
                 max_timestep_without_improve=1000):
        print("Starting training - Num episodes: {}, Range exploration: {}, Exploration phase: {}, Alpha: {}, "
              "Gamma: {},  Number of last frames: {}, Train batch: {}, Experience memory size: {}"
              .format(num_episodes, p_range_exploration, exploration_phase, alpha, gammma, num_last_frames, train_batch,
                      experience_memory_size))
        print("Rewards: {}".format(environment.metadata.get('rewards')))
        self.environment = environment
        self.num_episodes = num_episodes
        self.p_max_exploration, self.p_min_exploration = p_range_exploration
        self.exploration_phase = exploration_phase
        self.p_exploration = self.p_max_exploration
        self.p_exploration_decay = (self.p_max_exploration - self.p_min_exploration) / (
        self.num_episodes * exploration_phase)
        self.alpha = alpha
        self.gamma = gammma
        self.num_last_frames = num_last_frames
        self.train_batch = train_batch
        self.num_stored_models = num_stored_models
        self.memory = ExperienceMemory(memory_size=experience_memory_size, num_last_frames=num_last_frames)
        self.model = self._create_model()
        self.experiment_identifier = strftime("%Y%m%d%H%M")
        self.max_timestep_without_improve = max_timestep_without_improve

    def _train(self):
        states = []
        predictions = []
        experience = self.memory.get_batch(self.train_batch)

        for state, next_state, action, reward, done in experience:
            next_prediction = self.predict_one(next_state)
            max_next_reward = np.max(next_prediction)

            prediction = self.predict_one(state)
            prediction[action] = (1 - self.alpha) * prediction[action] \
                                 + self.alpha * (reward + self.gamma * max_next_reward)
            states.append(state)
            predictions.append(prediction)

        input_data = np.array(states)
        # input_data = np.array([np.array([state]) for state in states])
        return float(self.model.train_on_batch(input_data, np.array(predictions)))

    def _create_model(self):
        model = Sequential()
        model.add(Conv2D(
            16,
            kernel_size=(3, 3),
            strides=(1, 1),
            data_format='channels_first',
            input_shape=(self.num_last_frames,) + self.environment.observation_space.shape
        ))
        model.add(Activation('relu'))
        model.add(Conv2D(
            32,
            kernel_size=(3, 3),
            strides=(1, 1),
            data_format='channels_first'
        ))
        model.add(Activation('relu'))

        # Dense layers.
        model.add(Flatten())
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dense(3))  # Number of actions

        model.summary()
        model.compile(RMSprop(), 'MSE')

        return model

    def take_action(self, state):
        action = None
        if np.random.rand() < self.p_exploration:
            # Completely random
            action = self.environment.action_space.sample()
        else:
            action = np.argmax(self.predict_one(state))

        return action

    def execute(self):
        # cum_rewards = []
        for i in range(self.num_episodes):
            # print("Executing episode {}".format(i + 1))
            self.environment.reset()
            state = self.environment.current_state()
            cum_reward = 0
            done = False
            episode_memory = Episode(state, num_last_frames=self.num_last_frames)
            num_timestep_without_improve = 0
            while not done and num_timestep_without_improve < self.max_timestep_without_improve:
                # self.environment.render()
                state_frames = episode_memory.get_last_state()
                action = self.take_action(state_frames)
                next_state, reward, done, info = self.environment.step(action)
                episode_memory.add_timestep(next_state, action, reward, done)
                if cum_reward + reward < cum_reward:
                    num_timestep_without_improve += 1
                else:
                    num_timestep_without_improve = 0
                cum_reward += reward
            self.memory.remember(episode_memory)
            loss = self._train()
            print("{} - Episode {}/{}. Accumulated reward: {:2.4f}. Loss: {:8.4f}. Exploration rate: {:.2f}. Fruits: "
                  "{}. Timesteps: {}"
                  .format(strftime("%Y-%m-%d %H:%M:%S"), (i + 1), self.num_episodes, cum_reward, loss,
                          self.p_exploration, info['fruits_eaten'], episode_memory.num_timesteps()))
            if self.p_exploration > self.p_min_exploration:
                self.p_exploration -= self.p_exploration_decay
                # cum_rewards.append(cum_reward)
                # self._plot_learning_curve(cum_rewards)
            if (i + 1) % (self.num_episodes // self.num_stored_models) == 0:
                save_model(self.model, 'models/'+self.experiment_identifier+'-'+str((float(i+1) / self.num_episodes)))

    def _plot_learning_curve(self, rewards):
        episodes = range(1, self.num_episodes + 1)
        plt.figure()
        plt.plot(episodes, rewards)
        plt.xlabel('Episodes')
        plt.ylabel('Reward')
        plt.show()
        return

    def predict_one(self, state):
        input_data = np.array([state])
        return self.model.predict(input_data, batch_size=1)[0]


def main():
    if len(argv) <= 1:
        print('Number of episodes expected!')
    else:
        env = gym.make('Snake-v0')
        learner = QLearning(env, int(argv[1]), p_range_exploration=(0.99, 0.05), exploration_phase=0.9, alpha=0.85,
                            gammma=0.75, train_batch=64, experience_memory_size=256, num_stored_models=4)
        learner.execute()


if __name__ == '__main__':
    main()
