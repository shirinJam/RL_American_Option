"""A trading environment"""
import gym
import logging
import numpy as np
from gym.utils import seeding

logger = logging.getLogger("root")


class OptionEnvironment(gym.Env):
    """
    Environment
    This class instantiates the parameters and methods required for setting up the environment
    """

    def __init__(self, S0, K, r, sigma, T, N, sabr_flag=False, option_type="put"):

        # initialising the required option parameters
        self.S0 = S0
        self.K = K
        self.r = r
        self.sigma = sigma
        self.sigma0 = sigma
        self.T = T
        self.N = N  # number of days in a year
        self.sabr_flag = sabr_flag  # Stochastic-volatality (True or False)
        self.option_type = option_type  # Type of option (Put or Call)

        # definig the required state variables
        self.S1 = 0
        self.reward = 0
        self.day_step = 0  # from day 0 taking N steps to day N

        # defining the action space
        self.action_space = gym.spaces.Discrete(2)  # 0: hold, 1:exercise
        
        # defining the state space - shape=2 (asset price, day step)
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0]), high=np.array([np.inf, 1.0]), dtype=np.float32
        )  # S in [0, inf], tao in [0, 1]

        # seed and start
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """Applies an action to the environment and returns a TimeStep tuple

        Args:
            action (int): action which is either to exercise(1) or to hold(0)

        Returns:
            tuple: returns a tuple with observation(object), reward(float), done(boolean) and info(dict)
        """

        if action == 1:  # exercise
            if self.option_type == "put":
                reward = max(self.K - self.S1, 0.0) * np.exp(
                    -self.r * self.T * (self.day_step / self.N)
                )
            if self.option_type == "call":
                reward = max(self.S1 - self.K, 0.0) * np.exp(-self.r * self.T * (self.day_step/self.N))
            done = True
        else:  # hold
            if self.day_step == self.N:  # at maturity
                if self.option_type == "put":
                    reward = max(self.K - self.S1, 0.0) * np.exp(-self.r * self.T)
                if self.option_type == "call":
                    reward = max(self.S1 - self.K, 0.0) * np.exp(-self.r * self.T)
                done = True
            else: # move to tomorrow (Monte Carlo Simulation)
                if self.sabr_flag=="True":  # GBM with stochastic volatality
            
                    # SABR parameters
                    beta = 1
                    rho = -0.4
                    volvol = 0.6

                    qs = np.random.normal()
                    qi = np.random.normal()
                    qv = rho * qs + np.sqrt(1 - rho * rho) * qi

                    gvol = self.sigma * (self.S1 ** (beta - 1))
                    self.S1 = self.S1 * np.exp(
                        (self.r - 0.5 * gvol ** 2) * (self.T / self.N)
                        + gvol * np.sqrt(self.T / self.N) * qs
                    )

                    self.sigma = self.sigma * np.exp(
                        (-0.5 * volvol ** 2) * (self.T / self.N)
                        + volvol * np.sqrt(self.T / self.N) * qv
                    )

                    self.day_step += 1
                    reward = 0
                    done = False

                else:  # GBM with constant volatality
                    reward = 0
                    self.S1 = self.S1 * np.exp(
                        (self.r - 0.5 * self.sigma ** 2) * (self.T / self.N)
                        + self.sigma * np.sqrt(self.T / self.N) * np.random.normal()
                    )
                    self.day_step += 1
                    done = False

        tao = 1.0 - self.day_step / self.N  # time to maturity, in unit of years
        return np.array([self.S1, tao]), reward, done, {}

    def reset(self):
        """Resets the environment when the episode ends or when the function is called

        Returns:
            tuple: observation-space including asset price and day step
        """
        self.day_step = 0
        self.S1 = self.S0
        self.sigma = self.sigma0
        tao = 1.0 - self.day_step / self.N  # time to maturity, in unit of years
        return [self.S1, tao]