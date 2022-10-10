
from os import path
from typing import List, Optional

import numpy as np
from gym import Env, spaces, utils
from gym.envs.toy_text.utils import categorical_sample

# board -> configuration -> state

class TicTacToeBoard():
    
    BASE_3 = 3**np.arange(9)
    SYMBOLS = [" ", "X", "O"]

    def __init__(self, player=1):
        self.player, self.adversary_player = player, player % 2 + 1
        self.num_configurations = 3**9
        
        self.configurations = np.arange(self.num_configurations)
        
        self.configuration_feasible_map = np.array([TicTacToeBoard.is_configuration_feasible(c) for c in self.configurations])
        
        self.num_feasible_configurations = self.configuration_feasible_map.sum()
        
        self.state_to_configuration,  = np.where(self.configuration_feasible_map)
        
        self.configuration_to_state = {c: i for i, c in enumerate(self.state_to_configuration)} 
        
        self.board = np.zeros((3, 3), dtype=np.int8)

    def print_board(self):
        values = [self.SYMBOLS[i] for i in self.board.flatten() ]
        print("\n")
        print("\t     |     |")
        print("\t  {}  |  {}  |  {}".format(values[0], values[1], values[2]))
        print('\t_____|_____|_____')
    
        print("\t     |     |")
        print("\t  {}  |  {}  |  {}".format(values[3], values[4], values[5]))
        print('\t_____|_____|_____')
    
        print("\t     |     |")
    
        print("\t  {}  |  {}  |  {}".format(values[6], values[7], values[8]))
        print("\t     |     |")
        print("\n")

    @property
    def state(self):
        configuration = self.board_to_configuration(self.board)
        return self.configuration_to_state[configuration]

    def play(self, a, player):
        assert player in [1, 2]
        i = a // 3
        j = a - 3*i
        if self.board[i, j] != 0:
            return False
        else:
            self.board[i, j] = player
            return True

    def feasible_actions(self):
        af, = np.where(self.board.flatten() == 0)
        return af

    def check_winner(self):
        winner = self.check_won_lost(self.board)
        return winner
    
    def is_full(self):
        return (self.board != 0).all()

    @staticmethod
    def board_to_configuration(board):
        return (board.flatten() * TicTacToeBoard.BASE_3).sum()

    @staticmethod
    def configuration_to_board(configuration):
        repr = np.base_repr(configuration, 3)
        repr = "0"*(9 - len(repr)) + repr
        repr = repr[::-1]
        return np.array(list(map(int, repr))).reshape(3, 3)

    @staticmethod
    def check_won_lost(board):
        wons = np.zeros((2, 8)) # 8 possible winning for both players
        for p in [1, 2]: # players
            for k in range(3):
                wons[p-1, 2*k] = (board[k] == p).all()
                wons[p-1, 2*k+1] = (board[:, k] == p).all()
            wons[p-1, 6] = (board.diagonal() == p).all()
            wons[p-1, 7] = (np.flip(board, 0) == p).all()
        no_double_win = (wons != 0).sum() < 2 # there are no 2 winning position
        if wons[0].any() and no_double_win:
            winner = 1
        elif wons[1].any() and no_double_win:
            winner = 2
        elif no_double_win:
            winner = 0
        else: # the board is not feasible
            winner = -1
        return winner 
    
    @staticmethod
    def is_board_feasible(board, player):
        # only non-finished configurations are feasible, draw, win, lose cases are implemented seperately.
        feasible_1 = TicTacToeBoard.check_won_lost(board) == 0
        feasible_2 = (not (board == 1).sum() != (board == 2).sum() + (player - 1))
        return feasible_1 and feasible_2 

    @staticmethod
    def is_configuration_feasible(configuration, player=1):
        board = TicTacToeBoard.configuration_to_board(configuration)
        return TicTacToeBoard.is_board_feasible(board, player)

class TicTacToe(Env):
    
    PLAYERS = [1, 2]

    def __init__(self, player = 1):
        assert player in self.PLAYERS, "Player must be 1 ( X ) or 2 (O) !"
        self.player, self.adversary_player = player, player % 2 + 1
        self.board = TicTacToeBoard(player=self.player)
        self.observation_space = spaces.Discrete(self.board.num_feasible_configurations + 1)
        self.action_space = spaces.Discrete(9)

    def reset(self):
        return self.board.state, {}

    def step(self, a):

        success = self.board.play(a, player=self.player)
        
        if not success:
            next_state = self.board.num_feasible_configurations # extra error state
            reward = -10
            done, truncated = True, False
        else:
            feasible_adverasry_actions = self.board.feasible_actions()
            if len(feasible_adverasry_actions) > 0: # otherwise, no play for adversary
                a_adv = np.random.choice(feasible_adverasry_actions)
                self.board.play(a_adv, player=self.adversary_player)
            next_state = self.board.state
            reward = (self.board.check_winner() == self.player)*1
            done = (reward != 0) # if lost or won
            truncated = self.board.is_full()
        return next_state, reward, done, truncated, {}

    def adversary_step(self):
        feasible_adverasry_actions = self.board.feasible_actions()
        if len(feasible_adverasry_actions) > 0: # otherwise, no play for adversary
            a_adv = np.random.choice(feasible_adverasry_actions)
            self.board.play(a_adv, player=self.adversary_player)
            return True
        else:
            return False

    def render(self):
        self.board.print_board()

if __name__=="__main__":
    env = TicTacToe()
    state, _ = env.reset()