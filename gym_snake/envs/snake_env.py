"""
Snake game using Gym framework.

OpenAI - Request for Research 2.0
"""

import gym
from gym import spaces
import numpy as np


class ACTION(object):
    T_LEFT = 0
    CONTINUE = 1
    T_RIGHT = 2


class GRID(object):
    N_ROWS = 50
    N_COLUMNS = 50
    CELL_WIDTH = 10


class DIRECTION(object):
    N = (0, 1)
    W = (-1, 0)
    E = (1, 0)
    S = (0, -1)

    T_DIRECTION = [N, E, S, W]


class SnakeEnv(gym.Env):
    def __init__(self):
        self.__version__ = "0.0.1"
        print("SnakeEnv - Version {}".format(self.__version__))

        self.screen_width = GRID.N_COLUMNS * GRID.CELL_WIDTH
        self.screen_height = GRID.N_ROWS * GRID.CELL_WIDTH

        self.state = None

        self.viewer = None
        self.state = None

        self.steps_beyond_done = None
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=0,
            high=2,
            shape=(GRID.N_COLUMNS, GRID.N_ROWS),
            dtype=np.uint8)

        # Snake attributes
        self.init_size = 10
        self.direction = None
        self.snake = []

        self.snake_trans = []

        # Grid attributes
        self.fruits = []

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
        return [], reward, False, {}

    def _take_action(self, action):
        if action == ACTION.CONTINUE:
            pass
        elif action == ACTION.T_LEFT:
            self.direction = (self.direction - 1) % len(DIRECTION.T_DIRECTION)
        elif action == ACTION.T_RIGHT:
            self.direction = (self.direction + 1) % len(DIRECTION.T_DIRECTION)
        else:
            #Wrong action
            pass

        t_direction = DIRECTION.T_DIRECTION[self.direction]

        for i in range(len(self.snake) - 1, 0, -1):
            self.snake[i] = self.snake[i-1]

        self.snake[0] = (self.snake[0][0] + t_direction[0], self.snake[0][1] + t_direction[1])


    def _sample(self):
        return np.clip(round(np.random.normal(1,0.5)), 0, 2)

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.
        Returns
        -------
        observation (object): the initial observation of the space.
        """

        # Init Snake position (We ensure a secure position)
        secure_limit = GRID.N_ROWS * 0.10
        snake_head = (np.random.randint(0 + secure_limit, GRID.N_ROWS - secure_limit),
                      np.random.randint(0 + secure_limit, GRID.N_COLUMNS - secure_limit))
        self.direction = np.random.randint(0, 4)
        opposite_dir = DIRECTION.T_DIRECTION[(self.direction + 2) % 4]
        self.snake = [snake_head] \
                     + [(snake_head[0] + opposite_dir[0] * i, snake_head[1] + opposite_dir[1] * i) for i in range(1, self.init_size)]

    def render(self, mode='human'):

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(self.screen_width, self.screen_height)

            for _ in self.snake:
                cell = rendering.FilledPolygon([
                    (0, 0),
                    (0, GRID.CELL_WIDTH),
                    (GRID.CELL_WIDTH, GRID.CELL_WIDTH),
                    (GRID.CELL_WIDTH, 0),
                ])
                cell.set_color(100, 0, 0)
                celltrans = rendering.Transform()
                cell.add_attr(celltrans)
                self.viewer.add_geom(cell)
                self.snake_trans.append(celltrans)

        for cell, celltrans in zip(self.snake, self.snake_trans):
            celltrans.set_translation(cell[0] * GRID.CELL_WIDTH, cell[1] * GRID.CELL_WIDTH)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def _get_reward(self):
        return 0

    def close(self):
        if self.viewer: self.viewer.close()
