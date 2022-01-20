"""Help debug an env by rendering it.

Usage:
    python -m src.gym_envs.watch
"""
import time

import fire
import gym

import src.gym_envs  # pylint: disable = unused-import
import src.me_es.custom_gym  # pylint: disable = unused-import
import src.qd_gym  # pylint: disable = unused-import


def main(env: str = "QDAntBulletEnv-v0", mode: str = "still"):
    """Renders an env.

    Args:
        mode: "still" -> just show the env; "action" -> make the agent take
            random actions
    """
    env = gym.make(env)
    env.render()  # Needed so that pybullet envs work.
    print("Observation Space:", env.observation_space)
    print("Action Space:", env.action_space)
    if mode == "still":
        # Reset every 100 iters to see different initial positions.
        while True:
            env.reset()
            for _ in range(100):
                env.render()
                time.sleep(1 / 100)
    elif mode == "action":
        done = False
        env.reset()
        while True:
            env.render()
            a = env.action_space.sample()
            ob, r, done, info = env.step(a)  # pylint: disable = unused-variable
            if done:
                ob = env.reset()
            time.sleep(1 / 100)
    else:
        raise ValueError(f"Unrecognized mode '{mode}'")


if __name__ == "__main__":
    fire.Fire(main)
