
import numpy as np
import matplotlib.pyplot as plt 

from tictactoe import TicTacToe
from q_learning import QLearningAgent

def experiment(player=1):
	env = TicTacToe(player=player)
	agent = QLearningAgent(env)

	gamma = 0.9
	alpha = 0.8
	grid_n_step = [1, 2, 3]
	grid_epsilon = [0.0, 0.1, 0.2, 0.3]
	num_experiment = 10

	fig, axs = plt.subplots(3, 4, figsize=(16, 9))

	for i, n_step in enumerate(grid_n_step):
		for j, epsilon in enumerate(grid_epsilon):
			exp_result = []
			for k in range(num_experiment):
				agent.reset_q_table()
				agent.set_parameters(gamma, alpha, n_step, epsilon)
				result = agent.train()
				result = np.convolve(result, np.ones(20)/20, mode='valid')
				exp_result.append(result)
			exp_result = np.stack(exp_result)
			mu = exp_result.mean(axis=0)
			std = exp_result.std(axis=0)
			#
			train_step_idx = np.arange(len(mu))
			axs[i,j].plot(train_step_idx, mu,'b-',alpha=0.8, label="score")
			axs[i,j].fill_between(train_step_idx, mu - std, mu + std, facecolor='b', alpha=0.4)
			axs[i,j].set_title(f"n_step: {n_step}, epsilon: {epsilon}")
	fig.savefig(f"learning_curve_{player}.png")
	# fig.show()

if __name__=="__main__":
	experiment(1)
	experiment(2)