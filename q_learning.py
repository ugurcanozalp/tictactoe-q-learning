
import random
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from tictactoe import TicTacToe

class QLearningAgent():
	def __init__(self, env, gamma=0.9, alpha=0.5, n_step = 1, epsilon = 0.1):
		super(QLearningAgent, self).__init__()
		self._env = env
		self.reset_q_table()
		self.set_parameters(gamma, alpha, n_step, epsilon)

	def reset_q_table(self):
		self._Q = 0.1*np.random.randn(self._env.observation_space.n, self._env.action_space.n) # Q table

	def set_parameters(self, gamma=0.9, alpha=0.5, n_step = 1, epsilon = 0.1):
		self._gamma = gamma
		self._alpha = alpha
		self._n_step = n_step
		self._epsilon = epsilon

	def act(self, s):
		if np.random.rand() > self._epsilon:
			a = self._Q[s].argmax()
		else:
			a = self._env.action_space.sample()
		return a

	def train(self, num_episodes=2000):
		episode_scores = np.zeros(num_episodes)
		for e in range(num_episodes):
			s, _ = self._env.reset()
			d, truncated = False, False
			s_buf = deque(maxlen=self._n_step)
			a_buf = deque(maxlen=self._n_step)
			r_buf = deque(maxlen=self._n_step)
			t = 0
			while not (d or truncated):
				a = self.act(s)
				sp, r, d, truncated, _ = self._env.step(a)
				# self._env.render()
				t += 1
				# fill buffer
				s_buf.append(s)
				a_buf.append(a)
				r_buf.append(r)
				# TD Update
				if t < self._n_step:
					continue
				else:
					ap_best = self._Q[sp].argmax()
					s0, a0 = s_buf[0], a_buf[0]
					td_target = (1-d) * self._gamma**self._n_step * self._Q[sp][ap_best]
					for i in range(self._n_step):
						td_target += self._gamma**i * r_buf[i]
					td_error = td_target - self._Q[s0][a0] 
					self._Q[s0][a0] = self._Q[s0][a0] + self._alpha * td_error
				s = sp # state update
				episode_scores[e] += r
		return episode_scores

if __name__=="__main__":
	env = TicTacToe()
	agent = QLearningAgent(env)
	res = agent.train()
	plt.plot(res, linestyle=" ", marker="."); plt.show()