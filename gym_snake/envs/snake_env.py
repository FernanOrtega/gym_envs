"""
Snake game using Gym framework.

OpenAI - Request for Research 2.0
"""

import gym
from gym import spaces
import numpy as np


class STATUS(object):
    DIED = 0
    ALIVE = 1

class GRID(object):
    N_ROWS = 50
    N_COLUMNS = 50
    CELL_WIDTH = 10


class SnakeEnv(gym.Env):
    def __init__(self):
        self.__version__ = "0.0.1"
        print("SnakeEnv - Version {}".format(self.__version__))

        self.screen_width = GRID.N_COLUMNS * GRID.CELL_WIDTH
        self.screen_height = GRID.N_ROWS * GRID.CELL_WIDTH

        self.state = STATUS.ALIVE

        self.viewer = None
        self.state = None

        self.steps_beyond_done = None
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.screen_width, self.screen_height, 3),
            dtype=np.uint8)
        self.viewer = None

    def step(self, action):
        """

        Parameters
        ----------
        action :

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        self._take_action(action)
        reward = self._get_reward()
        return self.observation_space, reward, self.state == STATUS.DIED, {}

    def _take_action(self, action):
        pass

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.
        Returns
        -------
        observation (object): the initial observation of the space.
        """
        pass

    def render(self, mode='human'):


        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(self.screen_width, self.screen_height)
            cell = rendering.FilledPolygon([])
            cell.set_color(100, 0, 0)
            self.viewer.add_geom(cell)

        if self.state is STATUS.DIED: return None

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def _get_reward(self):
        return self.state

    def close(self):
        if self.viewer: self.viewer.close()
