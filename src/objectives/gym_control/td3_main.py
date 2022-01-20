"""Demo TD3 in QD Ant for debugging.

Usage:
    python -m src.objectives.gym_control.td3_main train
    python -m src.objectives.gym_control.td3_main rollout
"""
import json
from typing import Union

import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
from alive_progress import alive_bar
from logdir import LogDir

from src.objectives.gym_control import GymControl, GymControlConfig
from src.objectives.gym_control.td3 import TD3, TD3Config
from src.utils.logging import setup_logging


def setup(env_id, layer_sizes, train, seed):
    gym_control = GymControl(
        GymControlConfig(
            env_id=env_id,
            return_type="sum",
            obj_result_opts={"aggregation": "mean"},
            init="xavier",
            layer_sizes=layer_sizes,
            action_noise=0.1,  # TD3 has exploration noise.
            activation="tanh",
            obs_collect_prob=0.001,
            use_norm_obs=False,
        ))

    td3 = TD3(
        TD3Config(
            buffer_size=1_000_000,
            train_critics_itrs=300,
            batch_size=256,
            pg_batch_size=256,
            discount=0.99,
            target_update_rate=0.005,
            target_update_freq=2,
            smoothing_noise_variance=0.2,
            smoothing_noise_clip=0.5,
            adam_learning_rate=3e-4,
            train=train,
        ),
        gym_control,
        behavior_dim=4,
        seed=seed + 1,
    )

    return gym_control, td3


def train_td3(
    env_id: str = "QDAntBulletEnv-v0",
    total_itrs: int = 3333,
    layer_sizes: tuple = (128, 128),
    train: Union[str, int] = "objective",  # See TD3Config.train
    seed: int = 42,
):
    assert train != "all", "train=all not supported"
    params = locals()

    logdir = LogDir(f"td3-{env_id}")
    logdir.save_data(params, "params.json")

    setup_logging(on_worker=False)

    torch.manual_seed(seed)
    gym_control, td3 = setup(env_id, layer_sizes, train, seed)

    # With 300 training steps for the critic, this gives ~1M training steps
    # total. Note that comparison with the original TD3 is not entirely
    # possible, since we run one episode and then train the critic, instead of
    # training and rolling out simultaneously.
    rng = np.random.default_rng(seed)
    evaluations = {"x": [], "y": []}
    with alive_bar(total_itrs) as progress:
        for itr in range(1, total_itrs + 1):
            progress()  # pylint: disable = not-callable

            res = gym_control.evaluate(td3.actors[0][0].serialize(),
                                       n_evals=1,
                                       seed=rng.integers(1e6),
                                       experience_evals=1)
            td3.add_experience(res.experience)
            td3.train_critics()

            if itr % 25 == 0 or itr == total_itrs:
                res = gym_control.evaluate(td3.actors[0][0].serialize(),
                                           n_evals=10,
                                           seed=rng.integers(1e6),
                                           disable_noise=True)

                if train == "objective":
                    agg = res.agg_return
                    name = "Objective"
                else:
                    agg = res.agg_bc[train]
                    name = f"bc_{train}"

                print(f"Average evaluation over 10 episodes: {agg}")
                evaluations["x"].append(itr)
                evaluations["y"].append(agg)
                logdir.save_data(evaluations, "evaluations.json")
                td3.save(logdir.pfile("td3.pkl"), logdir.pfile("td3.pth"))

                # Plot evaluation score.
                plt.close()
                plt.figure()
                plt.plot(evaluations["x"], evaluations["y"])
                plt.xlabel("Itrs")
                plt.ylabel(name)
                plt.savefig(logdir.file("evaluation.pdf"))


def rollout(
    logdir: str,
    n_evals: int = 5,
    render: bool = True,
):
    """Reads results from logdir and rolls out policies from the archive.

    Args:
        logdir: Path to a logging directory output by an experiment.
        n_evals: Number of rollouts to perform.
        render: Whether to render the rollout.
    """
    logdir = LogDir("TD3 Rollout", custom_dir=logdir)

    with logdir.pfile("params.json").open("r") as file:
        params = json.load(file)

    torch.manual_seed(params["seed"])
    gym_control, td3 = setup(params["env_id"], params["layer_sizes"],
                             params["train"], params["seed"])
    td3.load(logdir.pfile("td3.pkl"), logdir.pfile("td3.pth"))

    res = gym_control.evaluate(
        td3.actors[0][0].serialize(),
        n_evals,
        params["seed"],
        render=render,
    )

    print(res)


if __name__ == "__main__":
    fire.Fire({
        "train": train_td3,
        "rollout": rollout,
    })
