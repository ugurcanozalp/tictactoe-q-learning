
from os import path
from typing import List, Optional

import numpy as np
from gym import Env, spaces, utils
from gym.envs.toy_text.utils import categorical_sample


class TicTacToe(Env):

    def __init__(self, player = "X"):
        assert player not in {"X", "O"}, "Player must be X or O"
        self.player = 1 if player == "X" else 2
        self.observation_space = spaces.Discrete(3**9)
        self.action_space = spaces.Discrete(9)
        self.state = None

    def build_board(self):
        self.board = [
            [" ", " ", " "],
            [" ", " ", " "],
            [" ", " ", " "]
        ]

    def render(self):
        self.build_board()
        print(self.board)

    @staticmethod
    def board_state_to_state(board_state):
        pass

    def reset(self):
        self.board_state = np.zeros((3, 3), dtype=np.int)
        return self.board_state_to_state(self.board_state)

    def step(self, a):
        if self.board_state(a) != 0:
            reward = -1000
            next_state = self.board_state_to_state(self.board_state)
            done = True
        else:
            self.board_state[a] = self.player
            self._adversary_step()
            next_state = self.board_state_to_state(self.board_state)
            reward = 0 # will be implemented
            done = False
        return next_state, reward, done, {}