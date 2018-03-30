"""
Snake game using Gym framework.

OpenAI - Request for Research 2.0
"""

import gym
from gym import spaces
import numpy as np
from gym.envs.classic_control import rendering


class ACTION(object):
    CONTINUE = 0
    T_LEFT = 1
    T_RIGHT = 2


class GRID(object):
    N_ROWS = 50
    N_COLUMNS = 50
    CELL_WIDTH = 10
    WALL = -1
    EMPTY = 0
    FRUIT = 1
    SNAKE = 2


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
        self.num_fruits = 4

        self.viewer = None

        # self.steps_beyond_done = None
        self.action_space = spaces.Discrete(3)
        # -1: Wall, 0: Empty, 1: Fruit, 2: Snake
        self.observation_space = spaces.Box(
            low=-1,
            high=3,
            shape=(GRID.N_COLUMNS, GRID.N_ROWS),
            dtype=np.uint8)

        # Snake attributes
        self.init_size = 5
        self.grow_size = 2
        self.direction = None
        self.snake = None
        self.fruits = None
        self.snake_trans = []
        self.fruit_trans = []
        self.cells_to_grow = 0

        self.empty_state = self._create_empty_state()

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
        dead, fruit_eaten = self._take_action(action)
        reward = self._get_reward(dead, fruit_eaten)
        return self._create_state(), reward, dead, {'head_position': self.snake[0],
                                                    'head_direction': DIRECTION.T_DIRECTION[self.direction]}

    def _take_action(self, action):
        if action == ACTION.CONTINUE:
            pass
        elif action == ACTION.T_LEFT:
            self.direction = (self.direction - 1) % len(DIRECTION.T_DIRECTION)
        elif action == ACTION.T_RIGHT:
            self.direction = (self.direction + 1) % len(DIRECTION.T_DIRECTION)
        else:
            # Wrong action
            pass

        t_direction = DIRECTION.T_DIRECTION[self.direction]

        new_head_position = (self.snake[0][0] + t_direction[0], self.snake[0][1] + t_direction[1])

        fruit_eaten = new_head_position in self.fruits
        # A fruit was eaten
        if fruit_eaten:
            self.fruits.remove(new_head_position)
            self.fruits.append(self._create_fruit())
            self.cells_to_grow += self.grow_size

        self.snake.insert(0, new_head_position)

        if self.cells_to_grow == 0:
            self.snake.pop()
        else:
            self.cells_to_grow -= 1

        # for i in range(len(self.snake) - 1, 0, -1):
        #     self.snake[i] = self.snake[i - 1]
        #
        # self.snake[0] = new_head_position

        return self._is_dead(), fruit_eaten

    def _is_dead(self):
        head_position = self.snake[0]
        result = head_position[0] < 1 or head_position[0] >= GRID.N_COLUMNS - 1 \
                 or head_position[1] < 1 or head_position[1] >= GRID.N_ROWS - 1 or head_position in self.snake[1:]
        return result

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
                     + [(snake_head[0] + opposite_dir[0] * i, snake_head[1] + opposite_dir[1] * i) for i in
                        range(1, self.init_size)]
        self.fruits = [self._create_fruit() for _ in range(self.num_fruits)]

    def _create_fruit(self):
        while True:
            fruit = (np.random.randint(1, GRID.N_ROWS - 1), np.random.randint(1, GRID.N_COLUMNS - 1))
            if fruit not in self.snake or fruit not in self.fruits:
                return fruit

    def render(self, mode='human'):

        if self.viewer is None:
            self.viewer = rendering.Viewer(self.screen_width, self.screen_height)

            self._draw_walls()

            for _ in self.snake:
                self.snake_trans.append(self._add_cell(1, 0, 0))

            for _ in self.fruits:
                self.fruit_trans.append(self._add_cell(0, 1, 0))

        for _ in range(len(self.snake_trans), len(self.snake)):
            self.snake_trans.append(self._add_cell(1, 0, 0))

        for snake_cell, snake_trans in zip(self.snake, self.snake_trans):
            snake_trans.set_translation(snake_cell[0] * GRID.CELL_WIDTH, snake_cell[1] * GRID.CELL_WIDTH)

        for fruit, fruit_trans in zip(self.fruits, self.fruit_trans):
            fruit_trans.set_translation(fruit[0] * GRID.CELL_WIDTH, fruit[1] * GRID.CELL_WIDTH)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def _add_cell(self, r, g, b):
        current_cell = rendering.FilledPolygon([
            (0, 0),
            (0, GRID.CELL_WIDTH),
            (GRID.CELL_WIDTH, GRID.CELL_WIDTH),
            (GRID.CELL_WIDTH, 0),
        ])
        current_cell.set_color(r, g, b)
        current_celltrans = rendering.Transform()
        current_cell.add_attr(current_celltrans)
        self.viewer.add_geom(current_cell)
        return current_celltrans

    def _get_reward(self, dead, fruit_eaten):
        result = 10
        if dead:
            result = -100
        elif fruit_eaten:
            result = 100

        return result

    def _create_state(self):
        state = np.copy(self.empty_state)
        for fruit in self.fruits:
            state[fruit[0], fruit[1]] = GRID.FRUIT
        for cell_snake in self.snake:
            state[cell_snake[0], cell_snake[1]] = GRID.SNAKE

        return state

    def close(self):
        if self.viewer: self.viewer.close()

    def _draw_walls(self):
        left = rendering.FilledPolygon([
            (0, 0),
            (0, self.screen_height),
            (GRID.CELL_WIDTH, self.screen_height),
            (GRID.CELL_WIDTH, 0)
        ])
        left.set_color(.41, .41, .41)
        self.viewer.add_geom(left)

        right = rendering.FilledPolygon([
            (self.screen_width - GRID.CELL_WIDTH, 0),
            (self.screen_width - GRID.CELL_WIDTH, self.screen_height),
            (self.screen_width, self.screen_height),
            (self.screen_width, 0)
        ])
        right.set_color(.41, .41, .41)
        self.viewer.add_geom(right)

        upper = rendering.FilledPolygon([
            (GRID.CELL_WIDTH, self.screen_height - GRID.CELL_WIDTH),
            (GRID.CELL_WIDTH, self.screen_height),
            (self.screen_width - GRID.CELL_WIDTH, self.screen_height),
            (self.screen_width - GRID.CELL_WIDTH, self.screen_height - GRID.CELL_WIDTH)
        ])
        upper.set_color(.41, .41, .41)
        self.viewer.add_geom(upper)

        bottom = rendering.FilledPolygon([
            (GRID.CELL_WIDTH, 0),
            (GRID.CELL_WIDTH, GRID.CELL_WIDTH),
            (self.screen_width - GRID.CELL_WIDTH, GRID.CELL_WIDTH),
            (self.screen_width - GRID.CELL_WIDTH, 0)
        ])
        bottom.set_color(.41, .41, .41)
        self.viewer.add_geom(bottom)

    # Method to create an empty state to speed up every step
    def _create_empty_state(self):
        state = np.zeros(shape=(GRID.N_COLUMNS, GRID.N_ROWS))
        for i in range(0, GRID.N_COLUMNS):
            state[i, 0] = GRID.WALL
            state[i, GRID.N_COLUMNS - 1] = GRID.WALL

        for i in range(1, GRID.N_ROWS - 1):
            state[0, i] = GRID.WALL
            state[GRID.N_ROWS - 1, i] = GRID.WALL

        return state
