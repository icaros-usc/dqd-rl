"""Registers all the custom gym environments."""
from gym.envs.registration import register

register(
    id='HumanoidDeceptive-v2',
    entry_point='src.me_es.custom_gym.mujoco:HumanoidDeceptive',
    max_episode_steps=1000,
)

register(
    id='AntMaze-v3',
    entry_point='src.me_es.custom_gym.mujoco:AntMazeEnv',
    max_episode_steps=1500,
)

register(
    id='DamageAnt-v2',
    entry_point='src.me_es.custom_gym.mujoco:AntEnv',
    max_episode_steps=1000,
)

# Register a bunch of ant environments with different joints broken.
for joints in [[0], [1], [2], [3], [4], [5], [6], [7], [0, 1], [2, 3], [4, 5],
               [6, 7], [2, 3, 4, 5], [0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 4, 5],
               [0, 1, 6, 7], [2, 3, 6, 7], [0, 2], [4, 6], [0, 2, 4, 6], [1, 3],
               [5, 7]]:
    register(
        id='DamageAntBrokenJoint{}-v2'.format(''.join(map(str, joints))),
        entry_point='src.me_es.custom_gym.mujoco:AntEnv',
        max_episode_steps=1000,
        kwargs=dict(modif='joint', param=joints),
    )
