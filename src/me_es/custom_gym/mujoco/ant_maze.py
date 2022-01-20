# Modifications Copyright (c) 2020 Uber Technologies Inc.
#
# ------------------------------------------------------------------------
#
# THIS IS NOT THE ORIGINAL VERSION OF THE FILE.
#
# Last modified 2021-12-02
"""Ant Maze environment.

Modified from https://github.com/openai/mlsh from the "Meta-Learning Shared
Hierarchies" paper: https://arxiv.org/abs/1710.09767.
"""
import os

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class AntMazeEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """Entry point for the ant maze environment."""

    def __init__(self):
        self.goal = np.array([35, -25])
        this_path = os.path.dirname(os.path.abspath(__file__))
        mujoco_env.MujocoEnv.__init__(self, this_path + '/assets/ant_maze.xml',
                                      5)
        utils.EzPickle.__init__(self)

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        reward = -np.sqrt(np.sum(np.square(self.data.qpos[:2] - self.goal)))
        done = False
        ob = self._get_obs()
        # Note: There was previously a bug where self.data.qpos was not being
        # copied, so the same bc would show up if this environment was run a few
        # times.
        return ob, reward, done, dict(bc=np.array(self.data.qpos[:2]),
                                      x_pos=self.data.qpos[0])

    def _get_obs(self):
        """Concatenates the ant's position and velocities."""
        qpos = self.data.qpos.flatten()
        qpos[:2] = (qpos[:2] - 5) / 70  # Normalize xy to [-0.5, 0.5].
        return np.concatenate([
            qpos,
            self.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.8
        self.viewer.cam.elevation = -60
        self.viewer.cam.lookat[0] = 5
        self.viewer.cam.lookat[1] = 5
        self.viewer.opengl_context.set_buffer_size(1280, 1024)
