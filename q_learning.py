
import random
import numpy as np
import matplotlib.pyplot as plt
import gym

class QLearningAgent():
	def __init__(self, env, gamma=0.9, alpha=0.5, n_step = 1, epsilon = 0.3):
		super(QLearningAgent, self).__init__()
		self._env = env
		self._gamma = gamma
		self._alpha = alpha
		self._n_step = n_step
		self._Q = np.zeros((self._env.observation_space.n, self._env.action_space.n)) # Q table
		self._epsilon = epsilon
		# self._Q = np.random.randn(self._env.observation_space.n, self._env.action_space.n) # Q table

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
			while not (d or truncated):
				a = self.act(s)
				sp, r, d, truncated, _ = env.step(a)
				# TD Update
				ap_best = self._Q[sp].argmax()    
				td_target = r + self._gamma * self._Q[sp][ap_best]
				td_delta = td_target - self._Q[s][a]
				self._Q[s][a] += self._alpha * td_delta
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