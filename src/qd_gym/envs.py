"""Environment classes for QDgym."""
import logging

import gym
import numpy as np
from pybullet_envs.gym_locomotion_envs import (AntBulletEnv,
                                               HalfCheetahBulletEnv,
                                               HopperBulletEnv,
                                               HumanoidBulletEnv,
                                               Walker2DBulletEnv)

#  from pybullet_envs.robot_bases import MJCFBasedRobot

gym.logger.set_level(40)
logger = logging.getLogger(__name__)


class QDAntBulletEnv(AntBulletEnv):

    def __init__(self, render=False):
        super().__init__(render=render)
        self.T = 0
        self.tot_reward = 0.0
        self.desc = np.array([0.0 for _ in range(4)])
        # acc = accumulation (i.e. desc_acc is an integer count of foot
        # contacts).
        self.desc_acc = np.array([0.0 for _ in range(4)])

        logger.info(
            "The behavioural descriptor is %d-dimensional and defined as "
            "proportion of feet contact time with the ground in the order %s",
            len(self.desc), self.robot.foot_list)

    def reset(self):
        r = super().reset()
        self.T = 0
        self.tot_reward = 0.0
        self.desc = np.array([0.0 for _ in range(4)])
        self.desc_acc = np.array([0.0 for _ in range(4)])

        return r

    def step(self, a):
        state, reward, done, info = super().step(a)
        self.desc_acc += self.robot.feet_contact
        self.tot_reward += reward
        self.T += 1
        self.alive = (self.__dict__["_alive"] >= 0.0)
        self.desc = self.desc_acc / self.T
        # self.desc gets a new array when created above, so no need to copy.
        info["bc"] = self.desc
        info["x_pos"] = None
        return state, reward, done, info


class QDHalfCheetahBulletEnv(HalfCheetahBulletEnv):

    def __init__(self, render=False):
        super().__init__(render=render)
        self.T = 0
        self.tot_reward = 0.0
        self.desc = np.array([0.0 for _ in range(2)])
        self.desc_acc = np.array([0.0 for _ in range(2)])

        logger.info(
            "The behavioural descriptor is %d-dimensional and defined as "
            "proportion of feet contact time with the ground in the order %s",
            len(self.desc), [self.robot.foot_list[0], self.robot.foot_list[3]])

    def reset(self):
        r = super().reset()
        self.T = 0
        self.tot_reward = 0.0
        self.desc = np.array([0.0 for _ in range(2)])
        self.desc_acc = np.array([0.0 for _ in range(2)])

        return r

    def step(self, a):
        state, reward, done, info = super().step(a)
        self.desc_acc[0] += self.robot.feet_contact[0]
        self.desc_acc[1] += self.robot.feet_contact[3]
        self.tot_reward += reward
        self.T += 1
        self.alive = (self.__dict__["_alive"] >= 0.0)
        self.desc = self.desc_acc / self.T
        info["bc"] = self.desc
        info["x_pos"] = None
        return state, reward, done, info


class QDWalker2DBulletEnv(Walker2DBulletEnv):

    def __init__(self, render=False):
        super().__init__(render=render)
        self.T = 0
        self.tot_reward = 0.0
        self.desc = np.array([0.0 for _ in range(2)])
        self.desc_acc = np.array([0.0 for _ in range(2)])

        logger.info(
            "The behavioural descriptor is %d-dimensional and defined as "
            "proportion of feet contact time with the ground in the order %s",
            len(self.desc), self.robot.foot_list)

    def reset(self):
        r = super().reset()
        self.T = 0
        self.tot_reward = 0.0
        self.desc = np.array([0.0 for _ in range(2)])
        self.desc_acc = np.array([0.0 for _ in range(2)])

        return r

    def step(self, a):
        state, reward, done, info = super().step(a)
        self.desc_acc += self.robot.feet_contact
        self.tot_reward += reward
        self.T += 1
        self.alive = (self.__dict__["_alive"] >= 0.0)
        self.desc = self.desc_acc / self.T
        info["bc"] = self.desc
        info["x_pos"] = None
        return state, reward, done, info


class QDHumanoidBulletEnv(HumanoidBulletEnv):

    def __init__(self, render=False):
        super().__init__(render=render)
        self.T = 0
        self.tot_reward = 0.0
        self.desc = np.array([0.0 for _ in range(2)])
        self.desc_acc = np.array([0.0 for _ in range(2)])

        logger.info(
            "The behavioural descriptor is %d-dimensional and defined as "
            "proportion of feet contact time with the ground in the order %s",
            len(self.desc), self.robot.foot_list)

    def reset(self):
        r = super().reset()
        self.T = 0
        self.tot_reward = 0.0
        self.desc = np.array([0.0 for _ in range(2)])
        self.desc_acc = np.array([0.0 for _ in range(2)])

        return r

    def step(self, a):
        state, reward, done, info = super().step(a)
        self.desc_acc += self.robot.feet_contact
        self.tot_reward += reward
        self.T += 1
        self.alive = (self.__dict__["_alive"] >= 0.0)
        self.desc = self.desc_acc / self.T
        info["bc"] = self.desc
        info["x_pos"] = None
        return state, reward, done, info


class QDHopperBulletEnv(HopperBulletEnv):

    def __init__(self, render=False):
        super().__init__(render=render)
        self.T = 0
        self.tot_reward = 0.0
        self.desc = np.array([0.0 for _ in range(1)])
        self.desc_acc = np.array([0.0 for _ in range(1)])

        logger.info(
            "The behavioural descriptor is %d-dimensional and defined as "
            "proportion of feet contact time with the ground in the order %s",
            len(self.desc), self.robot.foot_list)

    def reset(self):
        r = super().reset()
        self.T = 0
        self.tot_reward = 0.0
        self.desc = np.array([0.0 for _ in range(1)])
        self.desc_acc = np.array([0.0 for _ in range(1)])

        return r

    def step(self, a):
        state, reward, done, info = super().step(a)
        self.desc_acc += self.robot.feet_contact
        self.tot_reward += reward
        self.T += 1
        self.alive = (self.__dict__["_alive"] >= 0.0)
        self.desc = self.desc_acc / self.T
        info["bc"] = self.desc
        info["x_pos"] = None
        return state, reward, done, info


if __name__ == "__main__":
    env = QDHalfCheetahBulletEnv()
    env.reset()
    a = env.action_space.sample()
    env.step(a)
    print(env.alive)
    print(env.desc)
