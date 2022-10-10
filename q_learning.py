
import random
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import gym

class QLearningAgent():
	def __init__(self, env, gamma=0.9, alpha=0.2, n_step = 3, epsilon = 0.3):
		super(QLearningAgent, self).__init__()
		self._env = env
		self._gamma = gamma
		self._alpha = alpha
		self._n_step = n_step
		self._Q = np.zeros((self._env.observation_space.n, self._env.action_space.n)) # Q table
		self._epsilon = epsilon

	def act(self, s):
		if np.random.rand() > self._epsilon:
			a = self._Q[s].argmax()
		else:
			a = self._env.action_space.sample()
		return a

	def train(self, num_episodes=1000):
		episode_scores = np.zeros(num_episodes)
		for e in range(num_episodes):
			s, _ = env.reset()
			d, truncated = False, False
			s_buf = deque(maxlen=self._n_step)
			a_buf = deque(maxlen=self._n_step)
			r_buf = deque(maxlen=self._n_step)
			while not (d or truncated):
				a = self.act(s)
				sp, r, d, truncated, _ = env.step(a)
				# fill buffer
				s_buf.append(s)
				a_buf.append(a)
				r_buf.append(r)
				# TD Update
				if len(s_buf) < self._n_step:
					continue
				ap_best = self._Q[sp].argmax()    
				td_target = sum([(self._gamma**i)*rew for i, rew in enumerate(r_buf)]) + (self._gamma**self._n_step) * self._Q[sp][ap_best]
				td_error = td_target - self._Q[s_buf[0]][a_buf[0]]
				self._Q[s_buf[0]][a_buf[0]] = (1-self._alpha)*self._Q[s_buf[0]][a_buf[0]] + self._alpha * td_error
				# state update
				s = sp
				# add score
				episode_scores[e] += r
		return episode_scores

if __name__=="__main__":
	env = gym.make('FrozenLake-v1', is_slippery=True)
	agent = QLearningAgent(env)
	res = agent.train()
	plt.plot(res, linestyle=" ", marker="."); plt.show()