"""QDgym."""
import logging

import gym
from gym.envs.registration import register

# ------------QDgym-------------

register(id='QDWalker2DBulletEnv-v0',
         entry_point='src.qd_gym.envs:QDWalker2DBulletEnv',
         max_episode_steps=1000,
         reward_threshold=2500.0)

register(id='QDHalfCheetahBulletEnv-v0',
         entry_point='src.qd_gym.envs:QDHalfCheetahBulletEnv',
         max_episode_steps=1000,
         reward_threshold=3000.0)

register(id='QDAntBulletEnv-v0',
         entry_point='src.qd_gym.envs:QDAntBulletEnv',
         max_episode_steps=1000,
         reward_threshold=2500.0)

register(id='QDHopperBulletEnv-v0',
         entry_point='src.qd_gym.envs:QDHopperBulletEnv',
         max_episode_steps=1000,
         reward_threshold=2500.0)

register(id='QDHumanoidBulletEnv-v0',
         entry_point='src.qd_gym.envs:QDHumanoidBulletEnv',
         max_episode_steps=1000)


def get_list():
    btenvs = [
        '- ' + spec.id
        for spec in gym.envs.registry.all()
        if spec.id.find('QD') >= 0
    ]
    return btenvs
