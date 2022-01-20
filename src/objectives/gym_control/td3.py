"""TD3 implementation."""
import copy
import logging
import pickle as pkl
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import cloudpickle
import gin
import numpy as np
import torch
import torch.nn.functional as F

from src.objectives.gym_control.critics import MLPCritic
from src.objectives.gym_control.gym_control import GymControl
from src.objectives.gym_control.replay_buffer import Experience, ReplayBuffer

logger = logging.getLogger(__name__)


@gin.configurable
@dataclass
class TD3Config:

    # Size of replay buffer.
    buffer_size: int = gin.REQUIRED

    # Number of iterations to train the critics on each call to train_critics().
    train_critics_itrs: int = gin.REQUIRED

    # Batch size for sampling from replay buffer.
    batch_size: int = gin.REQUIRED

    # Batch size for calculating policy gradient.
    pg_batch_size: int = gin.REQUIRED

    # Discount factor.
    discount: float = gin.REQUIRED

    # Known as tau in TD3 paper.
    target_update_rate: float = gin.REQUIRED

    # How often (in terms of training itrs) to update target networks.
    target_update_freq: int = gin.REQUIRED

    # Noise for smoothing the critic training - see section 5.3 of TD3 paper.
    smoothing_noise_variance: float = gin.REQUIRED
    smoothing_noise_clip: float = gin.REQUIRED

    # Learning rate for Adam optimizers for training critics and greedy actors.
    adam_learning_rate: float = gin.REQUIRED

    # Number of steps to take when improving a policy with gradient_ascent.
    gradient_steps: int = gin.REQUIRED

    # Learning rate for Adam optimizer in gradient_ascent.
    gradient_learning_rate: float = gin.REQUIRED

    # Which actors/critics to train.
    # - all: Objective and BCs
    # - objective: Just objective
    # - {i}: Just BC `i` (0-indexed)
    train: Union[str, int] = "all"


class TD3:
    """TD3 implementation modified for DQD-RL.

    Trains a greedy actor and critics for the objective and for each BC.

    Adapted from TD3 code by Scott Fujimoto:
    https://github.com/sfujim/TD3/blob/master/TD3.py
    """

    def __init__(self,
                 config: TD3Config,
                 gym_control: GymControl,
                 behavior_dim: int,
                 seed: int = None):
        self.config = config
        self.gym_control = gym_control

        # Technically, the actions could have different bounds for different
        # dims, but this works most of the time.
        self.max_action = float(gym_control.env.action_space.high[0])

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        logger.info("TD3 Device: %s", self.device)

        self.behavior_dim = behavior_dim

        self.n_actors_critics = (
            1 + self.behavior_dim  # 1 for objective.
            if self.config.train == "all" else 1)

        #
        # Attributes below need to be saved in save(), and they get replaced by
        # load().
        #

        self.rng = np.random.default_rng(seed)
        self.buffer = ReplayBuffer(config.buffer_size, gym_control.obs_shape,
                                   gym_control.action_shape, behavior_dim, seed)

        # Total training iters so far.
        self.total_it = 0

        # The first actor/critic trains for reward, and the others train for a
        # BC.
        self.actors = []
        self.critics = []
        for _ in range(self.n_actors_critics):
            actor = gym_control.new_actor().to(self.device)
            actor_target = copy.deepcopy(actor)
            actor_opt = torch.optim.Adam(actor.parameters(),
                                         lr=self.config.adam_learning_rate)
            self.actors.append((actor, actor_target, actor_opt))

            critic = MLPCritic(
                np.product(gym_control.obs_shape),
                np.product(gym_control.action_shape),
            ).to(self.device)
            critic_target = copy.deepcopy(critic)
            critic_opt = torch.optim.Adam(critic.parameters(),
                                          lr=self.config.adam_learning_rate)
            self.critics.append((critic, critic_target, critic_opt))

        logger.info("Created %d actors and critics", len(self.actors))

    def first_actor(self) -> np.ndarray:
        return self.actors[0][0].serialize()

    def add_experience(self, experience: List[Experience]):
        """Adds experience to the buffer."""
        for e in experience:
            self.buffer.add(e)

    def train_critics(self):
        """Trains the critics for config.train_critics_itrs iterations."""
        logger.info("Training critics for %d itrs",
                    self.config.train_critics_itrs)
        logger.info("Replay buffer size: %d", len(self.buffer))

        for _ in range(self.config.train_critics_itrs):
            self.total_it += 1

            # Sample replay buffer.
            batch = self.buffer.sample_tensors(self.config.batch_size)

            for (i, ((actor, actor_target, actor_opt),
                     (critic, critic_target,
                      critic_opt))) in enumerate(zip(self.actors,
                                                     self.critics)):
                with torch.no_grad():
                    # Select action according to policy and add clipped noise.
                    # We use numpy's rng since it is harder to reproduce PyTorch
                    # randomness.
                    noise = (self.rng.standard_normal(size=batch.action.shape,
                                                      dtype=np.float32) *
                             self.config.smoothing_noise_variance)
                    noise = torch.from_numpy(noise).clamp(
                        -self.config.smoothing_noise_clip,
                        self.config.smoothing_noise_clip,
                    ).to(self.device)

                    next_action = (actor_target(batch.next_obs) + noise).clamp(
                        -self.max_action, self.max_action)

                    # Compute the target Q value.
                    target_q1, target_q2 = critic_target(
                        batch.next_obs, next_action)
                    target_q = torch.min(target_q1, target_q2)

                    if self.config.train == "all":
                        # First actor/critic trains for reward, others for a BC.
                        reward = batch.reward if i == 0 else batch.bcs[:, i - 1]
                    elif self.config.train == "objective":
                        reward = batch.reward
                    else:
                        reward = batch.bcs[:, self.config.train]

                    target_q = (reward[:, None] + (1.0 - batch.done[:, None]) *
                                self.config.discount * target_q)

                # Get current Q estimates.
                current_q1, current_q2 = critic(batch.obs, batch.action)

                # Compute critic loss.
                critic_loss = (F.mse_loss(current_q1, target_q) +
                               F.mse_loss(current_q2, target_q))

                # Optimize the critic.
                critic_opt.zero_grad()
                critic_loss.backward()
                critic_opt.step()

                # Delayed policy updates.
                if self.total_it % self.config.target_update_freq == 0:

                    # Compute actor losses.
                    actor_loss = -critic.q1(batch.obs, actor(batch.obs)).mean()

                    # Optimize the actor.
                    actor_opt.zero_grad()
                    actor_loss.backward()
                    actor_opt.step()

                    # Update the frozen target models.
                    tau = self.config.target_update_rate
                    for param, target_param in zip(critic.parameters(),
                                                   critic_target.parameters()):
                        target_param.data.copy_(tau * param.data +
                                                (1 - tau) * target_param.data)

                    for param, target_param in zip(actor.parameters(),
                                                   actor_target.parameters()):
                        target_param.data.copy_(tau * param.data +
                                                (1 - tau) * target_param.data)

        logger.info("Finished training critics - total itrs: %d", self.total_it)

    def jacobian_batch(self, batch_solutions: np.ndarray) -> np.ndarray:
        """Calculates jacobian for each solution on the basis of one step.

        Each jacobian matrix consists of the jacobian for the objective followed
        by the jacobians for the BCs.
        """
        try:
            batch = self.buffer.sample_tensors(self.config.pg_batch_size)
            logger.info("Sampled %d experience", len(batch[0]))
        except ValueError:  # Buffer is empty.
            # Send back random gradient directions (normal distribution is
            # isotropic so the directions will be chosen u.a.r.).
            return self.rng.standard_normal(
                (
                    batch_solutions.shape[0],
                    self.n_actors_critics,
                    batch_solutions.shape[1],
                ),
                dtype=np.float32,
            )

        batch_jacobians = []
        for sol in batch_solutions:
            actor = self.gym_control.new_actor().deserialize(sol).to(
                self.device)
            jacobian = []
            for critic, *_ in self.critics:
                critic.zero_grad()
                actor_loss = -critic.q1(batch.obs, actor(batch.obs)).mean()
                actor_loss.backward()
                jacobian.append(actor.gradient())
            batch_jacobians.append(jacobian)

        return np.asarray(batch_jacobians)

    def gradient_ascent(self, sol: np.ndarray) -> np.ndarray:
        """Performs config.gradient_steps steps of gradient ascent on sol.

        The critic used is the first critic (usually the objective critic).

        Adapted from PGA-ME:
        https://github.com/ollenilsson19/PGA-MAP-Elites/blob/master/variational_operators.py#L24
        """
        actor = self.gym_control.new_actor().deserialize(sol).to(self.device)
        actor_opt = torch.optim.Adam(actor.parameters(),
                                     lr=self.config.gradient_learning_rate)
        batch = self.buffer.sample_tensors(self.config.pg_batch_size *
                                           self.config.gradient_steps)
        critic = self.critics[0][0]

        for i in range(self.config.gradient_steps):
            cur_slice = slice(i * self.config.pg_batch_size,
                              (i + 1) * self.config.pg_batch_size)
            obs = batch.obs[cur_slice]
            actor_loss = -critic.q1(obs, actor(obs)).mean()
            actor_opt.zero_grad()
            actor_loss.backward()
            actor_opt.step()

        return actor.serialize()

    def save(self, pickle_path: Path, pytorch_path: Path):
        """Saves data to a pickle file and a PyTorch file.

        The PyTorch file holds the actors and critics, and the pickle file holds
        all the other attributes.

        See here for more info:
        https://pytorch.org/tutorials/beginner/saving_loading_models.html#save
        """
        logger.info("Saving TD3 pickle data")
        with pickle_path.open("wb") as file:
            cloudpickle.dump(
                {
                    "rng": self.rng,
                    "buffer": self.buffer,
                    "total_it": self.total_it,
                },
                file,
            )

        logger.info("Saving TD3 PyTorch data")
        torch.save(
            {
                "actors": [(actor.state_dict(), actor_target.state_dict(),
                            actor_opt.state_dict())
                           for (actor, actor_target, actor_opt) in self.actors],
                "critics": [
                    (critic.state_dict(), critic_target.state_dict(),
                     critic_opt.state_dict())
                    for (critic, critic_target, critic_opt) in self.critics
                ],
            },
            pytorch_path,
        )

    def load(self, pickle_path: Path, pytorch_path: Path):
        """Loads data from files saved by save()."""
        with open(pickle_path, "rb") as file:
            pickle_data = pkl.load(file)
            self.rng = pickle_data["rng"]
            self.buffer = pickle_data["buffer"]
            self.total_it = pickle_data["total_it"]

        pytorch_data = torch.load(pytorch_path)
        for ((actor, actor_target, actor_opt),
             (actor_dict, actor_target_dict,
              actor_opt_dict)) in zip(self.actors, pytorch_data["actors"]):
            actor.load_state_dict(actor_dict)
            actor_target.load_state_dict(actor_target_dict)
            actor_opt.load_state_dict(actor_opt_dict)
        for ((critic, critic_target, critic_opt),
             (critic_dict, critic_target_dict,
              critic_opt_dict)) in zip(self.critics, pytorch_data["critics"]):
            critic.load_state_dict(critic_dict)
            critic_target.load_state_dict(critic_target_dict)
            critic_opt.load_state_dict(critic_opt_dict)

        return self
