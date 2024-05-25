import numpy as np
from Environment import UAVEnv
from Usernum_record import user_name_set


class ObserNormalization(object):
    def __init__(self):
        env = UAVEnv()
        obj = user_name_set()
        M = obj.get_user_num()
        self.high_observation = np.array(
            [1e6, env.ground_length, env.ground_width, 200 * 1048576])
        self.high_observation = np.append(self.high_observation, np.ones(M * 2) * env.ground_length)
        self.high_observation = np.append(self.high_observation, np.ones(M) * 6291458)
        self.high_observation = np.append(self.high_observation, np.ones(M))
        self.low_observation = np.zeros(4 * M + 4)  # uav loc, ue loc, task size, block_flag

    def obser_normal(self, observation):
        observation = observation / (self.high_observation - self.low_observation)
        observation = np.array(observation, dtype=float)
        return observation
