"""GymControl objective and config."""
import time
from dataclasses import dataclass
from typing import Callable, Literal, Optional, Sequence

import gin
import gym
import numpy as np
import pybullet_envs  # pylint: disable = unused-import
import torch
from torch import nn

import src.gym_envs  # pylint: disable = unused-import
import src.me_es.custom_gym  # pylint: disable = unused-import
import src.qd_gym  # pylint: disable = unused-import
from src.objectives.gym_control.actors import MLPActor
from src.objectives.gym_control.replay_buffer import Experience
from src.objectives.objective_base import ObjectiveBase
from src.objectives.objective_result import ObjectiveResult
from src.objectives.obs_stats import ObsStats


@gin.configurable(denylist=["use_norm_obs"])
@dataclass
class GymControlConfig:
    """Configuration for GymControl."""

    ## Environment ##

    # Name of the gym environment.
    env_id: str = gin.REQUIRED
    # Amount of time (seconds) to wait after calling env.render().
    render_delay: float = 0.0
    # How to accumulate the rewards into the return during each episode.
    # Options are to "sum" the rewards, take the "last" reward.
    return_type: Literal["sum", "last"] = "sum"
    # Options for ObjectiveResult.from_raw (see there for more info),
    obj_result_opts: dict = None

    ## Network ##

    # How to initialize network params. See self._init in GymControl for
    # options.
    init: str = "xavier"
    # Standard deviation when using normc initialization. The default value
    # should be fine.
    normc_stdev: float = 0.01

    # List of sizes of hidden layers of the neural net.
    layer_sizes: Sequence[int] = ()
    # Noise in the action space of the neural network.
    action_noise: float = 0.0
    # Activation for the hidden layers (input layer has no activation, output
    # layer has tanh). See self._activation in GymControl for valid values.
    activation: str = "tanh"

    ## Observation Normalization ##

    # Whether to use observation normalization. No need to configure in gin
    # because the Manager handles this.
    use_norm_obs: bool = None
    # Probability of collecting observation statistics from any given episode.
    # This is small for computational efficiency.
    obs_collect_prob: float = 0.001


class GymControl(ObjectiveBase):
    """Objectives involving rolling out a neural network policy in a gym env.

    Only operates with float32.
    """

    def __init__(self, config: GymControlConfig):
        super().__init__(config)

        ## Environment ##

        self._env = gym.make(config.env_id)

        if not isinstance(self._env.observation_space, gym.spaces.Box):
            raise ValueError("Observation space must be Box")
        if not isinstance(self._env.action_space, gym.spaces.Box):
            raise ValueError("Action space must be Box")

        ## Network ##

        layers = ([np.product(self._env.observation_space.shape)] +
                  list(config.layer_sizes) +
                  [np.product(self._env.action_space.shape)])
        self._layer_shapes = list(zip(layers[:-1], layers[1:]))

        self._activation = {
            "tanh": nn.Tanh,
            "relu": nn.ReLU,
        }[config.activation]

        self._init = {
            "xavier": nn.init.xavier_uniform_,
            "zeros": nn.init.zeros_,
        }[config.init]

        self._n_params = len(self.new_actor().serialize())

    @property
    def env(self):
        return self._env

    @property
    def n_params(self):
        """Number of parameters in the neural network."""
        return self._n_params

    @property
    def obs_shape(self):
        """Shape of environment observation space."""
        return self._env.observation_space.shape

    @property
    def action_shape(self):
        """Shape of environment action space."""
        return self._env.action_space.shape

    def new_actor(self):
        """Fresh actor."""
        return MLPActor(self._layer_shapes, self._activation)

    def initial_solution(self, seed: Optional[int] = None) -> np.ndarray:
        """1D array with weights of an initialized network."""
        return self.new_actor().initialize(self._init).serialize()

    def evaluate(
        self,
        solution: np.ndarray,
        n_evals: int = 1,
        seed: Optional[int] = None,
        obs_stats: ObsStats = None,
        render: bool = False,
        render_callable: Callable = None,
        disable_noise: bool = False,
        experience_evals: int = 0,
    ) -> ObjectiveResult:
        """Rolls out solution n_evals times in the environment.

        BCs are taken from info["bc"] from the environment. If info["bc"] does
        not exist, the BC is set to an empty array.

        See ObjectiveBase.evaluate for other args.

        This method guarantees that ObjectiveResult.experience will always be a
        list, though the list may be empty (in particular, if no experience was
        collected).

        Args:
            obs_stats: Current statistics for observation noramlization, only
                matters if config.use_norm_obs was True.
            render: Whether to render the environment.
            render_callable: A function which should be called with no args when
                the environment is rendered.
            disable_noise: Pass True to turn off action noise.
            experience_evals: Number of episodes from which to collect
                experience. Must be <= n_evals. Episodes are selected randomly.
        """
        rng = np.random.default_rng(seed)
        actor = self.new_actor().deserialize(solution)

        # New seed on each eval. Use tolist b/c seeds must be int, not np.int64.
        env_seeds = rng.integers(1e6, size=n_evals).tolist()

        # Determine which eval episodes should be collected.
        if experience_evals > n_evals:
            raise ValueError(f"experience_evals ({experience_evals}) must "
                             f"be <= n_evals ({n_evals})")
        collect_eval = set(rng.permutation(n_evals)[:experience_evals])

        # Data to collect.
        returns = np.zeros(n_evals, dtype=float)
        lengths = np.zeros(n_evals, dtype=int)
        final_xpos = [None for _ in range(n_evals)]  # env may not have this.
        bcs = [None for _ in range(n_evals)]  # BC shape not known in advance.
        if self.config.use_norm_obs:
            obs_sum = np.zeros(self.obs_shape, dtype=np.float32)
            obs_sumsq = np.zeros(self.obs_shape, dtype=np.float32)
            obs_count = 0
        experience = []

        with torch.inference_mode():  # Supposed to make things faster?
            for eval_i in range(n_evals):
                self._env.seed(env_seeds[eval_i])

                # pybullet envs need a render() here. See:
                # https://github.com/benelot/pybullet-gym#installing-pybullet-gym
                if render:
                    self._env.render()
                obs = self._env.reset()

                # Potentially collect observations from this eval.
                itr_obs = ([obs] if
                           (self.config.use_norm_obs and
                            rng.random() < self.config.obs_collect_prob) else
                           None)

                done = False
                collect = eval_i in collect_eval
                while not done:
                    lengths[eval_i] += 1
                    old_obs = obs  # Store obs before normalization.

                    # Apply observation normalization.
                    if self.config.use_norm_obs:
                        obs = (obs - obs_stats.mean) / obs_stats.std

                    # Compute action and add noise if necessary.
                    action = actor.action(obs)
                    if not disable_noise and self.config.action_noise > 0.0:
                        action += (
                            self.config.action_noise *
                            rng.standard_normal(len(action), dtype=np.float32))

                    obs, reward, done, info = self._env.step(action)

                    # BCs are assumed to come from the last step only, so
                    # intermediate steps set the BCs to 0.
                    bcs[eval_i] = info.get("bc", [])
                    if not done:
                        bcs[eval_i] = np.zeros_like(bcs[eval_i])

                    if collect:
                        experience.append(
                            Experience(
                                old_obs,
                                action,
                                reward,
                                bcs[eval_i],
                                obs,
                                done,
                            ))

                    # Record data.
                    returns[eval_i] += reward
                    final_xpos[eval_i] = info.get("x_pos", None)
                    if itr_obs is not None:
                        itr_obs.append(obs)

                    if render:
                        self._env.render()
                        if render_callable is not None:
                            render_callable()
                        time.sleep(self.config.render_delay)

                # Rewards are summed above by default (which covers the "sum"
                # return_type).
                if self.config.return_type == "last":
                    returns[eval_i] = reward

                if itr_obs is not None:
                    itr_obs = np.asarray(itr_obs)
                    obs_sum += itr_obs.sum(axis=0)
                    obs_sumsq += np.square(itr_obs).sum(axis=0)
                    obs_count += len(itr_obs)

        return ObjectiveResult.from_raw(
            returns=returns,
            bcs=np.asarray(bcs),
            lengths=lengths,
            final_xpos=(None
                        if final_xpos[0] is None else np.asarray(final_xpos)),
            obs_sum=obs_sum if self.config.use_norm_obs else None,
            obs_sumsq=obs_sumsq if self.config.use_norm_obs else None,
            obs_count=obs_count if self.config.use_norm_obs else None,
            experience=experience,
            opts=self.config.obj_result_opts,
        )
