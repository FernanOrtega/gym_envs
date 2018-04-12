import random


class TIMESTEP:
    STATE = 0,
    NEW_STATE = 1,
    ACTION = 2,
    REWARD = 3,
    DONE = 4


class ExperienceMemory(object):
    def __init__(self, num_last_frames=4, memory_size=100):
        self.num_last_frames = num_last_frames
        self.memory = []
        self.memory_size = memory_size

    def remember(self, episode_memory):
        self.memory.extend(episode_memory.get_all_timesteps())
        self.memory = self.memory[-self.memory_size:]

    def reset(self):
        self.memory.clear()

    def get_batch(self, batch_size):
        batch_size = min(batch_size, len(self.memory))
        if len(self.memory) > batch_size:
            experience = random.sample(self.memory, batch_size)
        else:
            experience = self.memory

        return experience


class Episode(object):
    def __init__(self, init_state, num_last_frames):
        self.num_last_frames = num_last_frames
        self.states = []
        for _ in range(self.num_last_frames):
            self.states.append(init_state)
        self.actions = []
        self.rewards = []
        self.done = []

    def add_timestep(self, next_state, action, reward, done):
        self.states.append(next_state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.done.append(done)

    def get_last_state(self):
        return self.states[-self.num_last_frames:]

    def get_last_timestep(self):
        return [self.states[-self.num_last_frames - 1:-1],
                self.states[-self.num_last_frames:], self.actions[-1], self.rewards[-1], self.done[-1]]

    def get_all_timesteps(self):
        result = []
        for i in range(len(self.done)):
            state = self.states[i:i + 4]
            next_state = self.states[i + 1:i + 5]
            action = self.actions[i]
            reward = self.rewards[i]
            done = self.done[i]
            result.append([state, next_state, action, reward, done])

        return result

    def num_timesteps(self):
        return len(self.done)