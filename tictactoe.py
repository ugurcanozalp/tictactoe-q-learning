
import numpy as np
import random

from gym import Env, spaces

# board -> configuration -> state

class TicTacToeBoard():
	
	BASE_3 = 3**np.arange(9)
	SYMBOLS = [" ", "X", "O"]
	DRAW = 0
	X_WINS = 1
	O_WINS = 2
	ERROR = 3
	NOT_FINISHED = 4
	STATUS = ["DRAW", "X_WINS", "O_WINS", "ERROR", "NOT_FINISHED"]

	def __init__(self, player=1):
		self.player, self.adversary_player = player, player % 2 + 1
		self.num_configurations = 3**9
		self.configurations = np.arange(self.num_configurations)
		self.configuration_status = np.array([TicTacToeBoard.configuration_to_status(c) for c in self.configurations])
		self.nonterminal_to_configuration, = np.where(self.configuration_status == TicTacToeBoard.NOT_FINISHED)
		self.configuration_to_nonterminal = {c: i for i, c in enumerate(self.nonterminal_to_configuration)}
		self.num_nonterminal_states = len(self.nonterminal_to_configuration)
		self.num_states = TicTacToeBoard.NOT_FINISHED + self.num_nonterminal_states
		self.reset_board()
		self.turn = 1

	def reset_board(self):
		self.board = np.zeros((3, 3), dtype=np.int8)
		self.failed = False
		self.turn = 1

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
		if self.failed:
			return TicTacToeBoard.ERROR
		configuration = self.board_to_configuration(self.board)
		status = self.configuration_status[configuration]
		if status != TicTacToeBoard.NOT_FINISHED: # if game is finished somehow
			return status
		else:
			return TicTacToeBoard.NOT_FINISHED  + self.configuration_to_nonterminal[configuration]

	def play(self, a, player):
		assert player == self.turn
		i = a // 3
		j = a - 3*i
		if self.board[i, j] != 0:
			self.failed = True
			self.current_status = TicTacToeBoard.ERROR
			self.turn = None
		else:
			self.board[i, j] = player
			self.current_status = self.check_status()
			self.turn = player % 2 + 1

	def feasible_actions(self):
		af, = np.where(self.board.flatten() == 0)
		return af

	def check_status(self):
		return self.board_to_status(self.board)

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
	def board_to_status(board):
		if (board == 1).sum() - (board == 2).sum() not in [0, 1]:
			return TicTacToeBoard.ERROR # error status
		wons = np.zeros((2, 8)) # 8 possible winning for both players
		for p in [1, 2]: # players
			for k in range(3):
				wons[p-1, 2*k] = (board[k] == p).all()
				wons[p-1, 2*k+1] = (board[:, k] == p).all()
			wons[p-1, 6] = (board.diagonal() == p).all()
			wons[p-1, 7] = (np.flip(board, 0).diagonal() == p).all()
		if wons[0].any():
			return TicTacToeBoard.X_WINS # X wins
		elif wons[1].any():
			return TicTacToeBoard.O_WINS # O wins
		elif (board.flatten()==0).sum()==0:
			return TicTacToeBoard.DRAW # table filled, no winner = draw
		else: # game did not finished yet
			return TicTacToeBoard.NOT_FINISHED

	@staticmethod
	def configuration_to_status(configuration):
		board = TicTacToeBoard.configuration_to_board(configuration)
		return TicTacToeBoard.board_to_status(board)

class TicTacToe(Env):
	
	PLAYERS = [1, 2]

	def __init__(self, player = 1):
		assert player in self.PLAYERS, "Player must be 1 ( X ) or 2 (O) !"
		self.player, self.adversary_player = player, player % 2 + 1
		self.board = TicTacToeBoard(player=self.player)
		self.observation_space = spaces.Discrete(self.board.num_states)
		self.action_space = spaces.Discrete(9)

	def reset(self):
		self.board.reset_board()
		if self.player == 2:
			self.adversary_step()
		return self.board.state, {}

	def step(self, a):
		self.board.play(a, player=self.player)
		if self.board.current_status == 4: # if not finished, let adversary play
			self.adversary_step()
		next_state = self.board.state # extra error state
		done = self.board.current_status != 4
		if self.board.current_status == 3: # error state 
			reward = -1
		elif self.board.current_status == self.player: # player wins
			reward = 1 
		elif self.board.current_status == self.adversary_player: # opponent wins
			reward = -1
		elif self.board.current_status == 0: # draw
			reward = 0
		else: # game not finished
			reward = 0
		return next_state, reward, done, False, {}

	# def adversary_step(self):
	# 	feasible_adversary_actions = self.board.feasible_actions()
	# 	if len(feasible_adversary_actions) > 0: # otherwise, no play for adversary
	# 		a_adv = np.random.choice(feasible_adversary_actions)
	# 		self.board.play(a_adv, player=self.adversary_player)

	def render(self):
		self.board.print_board()

	def adversary_step(self):
		a_adv = self.adversary_policy()
		self.board.play(a_adv, player=self.adversary_player)

	def adversary_policy(self):
		feasible_actions = self.board.feasible_actions()
		best_a_adv = None
		allowed_a_advs = []
		for a_adv in feasible_actions:
			board_tmp = self.board.board.copy()
			i = a_adv // 3
			j = a_adv - 3*i
			board_tmp[i, j] = self.adversary_player
			status_tmp = self.board.board_to_status(board_tmp)
			if status_tmp == self.adversary_player:
				best_a_adv = a_adv
				break
			player_wins=False
			for a_ply in feasible_actions:
				if a_ply == a_adv:
					continue
				i = a_ply // 3
				j = a_ply - 3*i	
				board_tmp[i, j] = self.player
				status_tmp = self.board.board_to_status(board_tmp)
				if status_tmp == self.player:
					player_wins = True
					break
			if not player_wins:
				allowed_a_advs.append(a_adv)
		if best_a_adv is not None: # if there is a final action that ends game, play it
			return best_a_adv
		elif len(allowed_a_advs)>0: # else, play one action that do not allow the player wins
			return random.choice(allowed_a_advs)
		else: # just play random, you will player will win probably.
			return random.choice(feasible_actions)

if __name__=="__main__":
	env = TicTacToe()
	state, _ = env.reset()