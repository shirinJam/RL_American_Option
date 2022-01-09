"""A trading environment"""
import gym
import logging
import numpy as np
from gym.utils import seeding

logger = logging.getLogger("root")


class OptionEnvironment(gym.Env):
    def __init__(self, S0, K, r, sigma, T, N):
        self.S0 = S0
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.N = N

        self.S1 = 0
        self.reward = 0
        self.day_step = 0  # from day 0 taking N steps to day N

        self.action_space = gym.spaces.Discrete(2)  # 0: hold, 1:exercise
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0]), high=np.array([np.inf, 1.0]), dtype=np.float32
        )  # S in [0, inf], tao in [0, 1]

        # seed and start
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        if action == 1:  # exercise
            reward = max(self.K - self.S1, 0.0) * np.exp(
                -self.r * self.T * (self.day_step / self.N)
            )
            done = True
        else:  # hold
            if self.day_step == self.N:  # at maturity
                reward = max(self.K - self.S1, 0.0) * np.exp(-self.r * self.T)
                done = True
            else:  # move to tomorrow
                reward = 0
                # lnS1 - lnS0 = (r - 0.5*sigma^2)*t + sigma * Wt
                self.S1 = self.S1 * np.exp(
                    (self.r - 0.5 * self.sigma ** 2) * (self.T / self.N)
                    + self.sigma * np.sqrt(self.T / self.N) * np.random.normal()
                )
                self.day_step += 1
                done = False

        tao = 1.0 - self.day_step / self.N  # time to maturity, in unit of years
        return np.array([self.S1, tao]), reward, done, {}

    def reset(self):
        self.day_step = 0
        self.S1 = self.S0
        tao = 1.0 - self.day_step / self.N  # time to maturity, in unit of years
        return [self.S1, tao]

    def render(self):
        """
        make video
        """
        pass

    def close(self):
        pass
