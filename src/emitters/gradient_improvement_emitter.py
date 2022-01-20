"""Provides the GradientImprovementEmitter used in CMA-MEGA and its variants."""
import itertools
import logging

import gin
import numpy as np
from ribs.archives import AddStatus
from ribs.emitters import EmitterBase

from src.emitters.opt.cma_es import CMAEvolutionStrategy
from src.emitters.opt.gradients import AdamOpt, GradientAscentOpt
from src.objectives.gym_control.td3 import TD3

logger = logging.getLogger(__name__)


@gin.configurable
class GradientImprovementEmitter(EmitterBase):
    """Adapts a covariance matrix towards changes in the archive.

    Adapted from:
    https://github.com/icaros-usc/dqd/blob/main/ribs/emitters/_gradient_improvement_emitter.py

    To use, call ask() and tell() twice. First, call ask() with
    `grad_estimate=True`, and depending on `gradient_source`, pass Jacobians to
    tell() or let them be calculated automatically.  Then, call ask() and tell()
    without gradients. If using TD3, ask() and tell() also need a `td3`
    argument.

    If using an Optimizer (src/optimizers/optimizer.py), `grad_estimate` and
    `td3` are passed in through `emitter_kwargs` in the Optimizer's ask() and
    tell() methods.

    If running CMA-MEGA (ES), set gradient_source="sample", and if running
    CMA-MEGA (TD3, ES), set gradient_source="td3_sample".

    Args:
        archive (ribs.archives.ArchiveBase): An archive to use when creating and
            inserting solutions. For instance, this can be
            :class:`ribs.archives.GridArchive`.
        x0 (np.ndarray): Initial solution.
        sigma_g (float): Initial gradient learning rate.
        stepsize (float): Step size for gradient optimizer (e.g. Adam or
            gradient ascent).
        selection_rule ("mu" or "filter"): Method for selecting solutions in
            CMA-ES. With "mu" selection, the first half of the solutions will be
            selected, while in "filter", any solutions that were added to the
            archive will be selected.
        restart_rule ("no_improvement" or "basic"): Method to use when checking
            for restart. With "basic", only the default CMA-ES convergence rules
            will be used, while with "no_improvement", the emitter will restart
            when none of the proposed solutions were added to the archive.
        weight_rule ("truncation" or "active"): Method for generating weights in
            CMA-ES. Either "truncation" (positive weights only) or "active"
            (include negative weights).
        gradient_optimizer: Gradient ascent optimizer for stepping the mean
            solution. We only use gradient ascent in this work, though the
            original CMA-MEGA also considered adam.
        normalize_gradients: Whether to normalize the objective and measure
            gradients such that they have equal magnitude. We always set this to
            True.
        bounds (None or array-like): Bounds of the solution space. Solutions are
            clipped to these bounds. Pass None to indicate there are no bounds.
            Alternatively, pass an array-like to specify the bounds for each
            dim. Each element in this array-like can be None to indicate no
            bound, or a tuple of ``(lower_bound, upper_bound)``, where
            ``lower_bound`` or ``upper_bound`` may be None to indicate no bound.
        batch_size (int): Number of solutions to return in :meth:`ask`. If not
            passed in, a batch size will automatically be calculated.
        seed (int): Value to seed the random number generator. Set to None to
            avoid a fixed seed.
        gradient_source: Where to get jacobians in tell() - None to get it
            passed in, "td3" to get all gradients from the TD3 object, "sample"
            to estimate from gradient samples with ES, "td3_and_sample" to
            estimate from gradient samples and also add an objective gradient
            from TD3 (the objective gradient from ES is still kept),
            "td3_sample" is like "td3_and_sample" but tosses out the objective
            gradient from ES.
        polyak: If passed in, jacobians will be accumulated with polyak *
            jacobian + (1 - polyak) * previous accumulation (i.e. we will take
            an exponential moving average). This was an idea that we ended up
            not using in our paper.
        sample_insert: Whether to insert solutions from gradient samples into
            the archive. We always set this to True.
        sample_sigma: Std of the distribution for sampling gradients.
        sample_mirror: Whether to use mirror sampling.
        sample_batch_size: Number of samples to make.
        sample_rank_norm: Whether to use rank normalization.
        greedy_eval: Whether to evaluate the greedy TD3 solution. Only applies
            when gradient_source is "td3_sample" or "td3_and_sample".
        greedy_insert: Whether to insert the greedy TD3 solution. Only applies
            when gradient_source is "td3_sample" or "td3_and_sample".
    Raises:
        ValueError: If any of ``selection_rule``, ``restart_rule``, or
            ``weight_rule`` is invalid.
    """

    def __init__(
        self,
        archive,
        x0,
        sigma_g,
        stepsize,
        selection_rule="mu",
        restart_rule="no_improvement",
        weight_rule="truncation",
        gradient_optimizer="adam",
        normalize_gradients=True,
        bounds=None,
        batch_size=None,
        seed=None,
        gradient_source: str = None,
        polyak: float = None,
        # Arguments for gradient sampling -- may move somewhere else like a
        # config object.
        sample_insert: bool = None,
        sample_sigma: float = None,
        sample_mirror: bool = None,
        sample_batch_size: int = 0,
        sample_rank_norm: bool = None,
        # Arguments when using TD3.
        greedy_eval: bool = None,
        greedy_insert: bool = None,
    ):
        self._rng = np.random.default_rng(seed)
        self._batch_size = batch_size
        self._x0 = np.array(x0, dtype=archive.dtype)
        self._sigma_g = sigma_g
        self._normalize_gradients = normalize_gradients
        EmitterBase.__init__(
            self,
            archive,
            len(self._x0),
            bounds,
        )

        self._gradient_source = gradient_source
        self._polyak = polyak
        assert polyak is None or 0.0 <= polyak <= 1.0, \
            "polyak must satisfy 0.0 <= polyak <= 1.0"

        self._gradient_opt = None
        if gradient_optimizer not in ["adam", "gradient_ascent"]:
            raise ValueError(
                f"Invalid Gradient Ascent Optimizer {gradient_optimizer}")
        if gradient_optimizer == "adam":
            self._gradient_opt = AdamOpt(self._x0, stepsize)
        elif gradient_optimizer == "gradient_ascent":
            self._gradient_opt = GradientAscentOpt(self._x0, stepsize)

        if selection_rule not in ["mu", "filter"]:
            raise ValueError(f"Invalid selection_rule {selection_rule}")
        self._selection_rule = selection_rule

        if restart_rule not in ["basic", "no_improvement"]:
            raise ValueError(f"Invalid restart_rule {restart_rule}")
        self._restart_rule = restart_rule

        opt_seed = None if seed is None else self._rng.integers(10_000)
        measure_dim = archive.behavior_dim
        self._num_coefficients = (
            measure_dim + 1 +
            int(self._gradient_source == "td3_and_sample"
               )  # Sample gradients and also add in the TD3 gradient
        )

        self._use_td3 = self._gradient_source in [
            "td3_sample", "td3_and_sample"
        ]
        self._greedy_eval = self._use_td3 and greedy_eval
        self._greedy_insert = self._greedy_eval and greedy_insert
        self._greedy_obj = None
        measure_x0 = np.zeros(self._num_coefficients)
        self.opt = CMAEvolutionStrategy(
            sigma_g,
            # Subtract 1 for greedy solution.
            batch_size - 1 if self._greedy_eval else batch_size,
            self._num_coefficients,
            weight_rule,
            opt_seed,
            self.archive.dtype,
        )
        self.opt.reset(measure_x0)
        logger.info("opt.batch_size: %d", self.opt.batch_size)
        self._num_parents = (self.opt.batch_size //
                             2 if selection_rule == "mu" else None)
        self._batch_size = (self.opt.batch_size +
                            1 if self._greedy_eval else self.opt.batch_size)
        self._restarts = 0  # Currently not exposed publicly.

        self._grad_coefficients = None  # Defined in ask().
        self._jacobians = None  # Defined in tell().

        # Gradient sampling settings.
        assert self._gradient_source is None or self._gradient_source in [
            "td3", "sample", "td3_and_sample", "td3_sample"
        ]
        if self._gradient_source in ["sample", "td3_and_sample", "td3_sample"]:
            self._sample_insert = sample_insert
            self._sample_sigma = sample_sigma
            self._sample_mirror = sample_mirror
            self._sample_batch_size = sample_batch_size
            self._sample_rank_norm = sample_rank_norm

            if self._sample_mirror:
                assert self._sample_batch_size % 2 == 0, \
                    "If using mirror_sampling, batch_size must be even."

            self._sample_noise = None

    @property
    def x0(self):
        """numpy.ndarray: Initial solution for the optimizer."""
        return self._x0

    @property
    def sigma_g(self):
        """float: Initial gradient learning rate."""
        return self._sigma_g

    @property
    def batch_size(self):
        """int: Number of solutions to return in :meth:`ask`."""
        return self._batch_size

    @property
    def sample_batch_size(self):
        """int: Number of solutions to use for estimating gradient."""
        return self._sample_batch_size

    @property
    def restarts(self):
        """int: Number of times the emitter has restarted."""
        return self._restarts

    @property
    def greedy_eval(self):
        """Whether this emitter evaluates the greedy TD3 solution."""
        return self._greedy_eval

    @property
    def greedy_obj(self):
        """Performance of the last evaluated greedy solution.

        Only applicable if self._greedy_eval is True.
        """
        return self._greedy_obj

    def ask(self, grad_estimate=False, td3=None):
        """Samples new solutions from a multivariate Gaussian.

        The multivariate Gaussian is parameterized by the CMA-ES optimizer.

        Args:
            grad_estimate: Whether this ask() is being called for a gradient
                estimate (i.e. the user will return the jacobians in tell()).
        Returns:
            ``(batch_size, solution_dim)`` array -- contains ``batch_size`` new
            solutions to evaluate.
        """

        if grad_estimate:
            mean = self._gradient_opt.theta

            if self._gradient_source in [
                    "sample", "td3_and_sample", "td3_sample"
            ]:
                logger.info("Generating samples for estimating gradient")
                if self._sample_mirror:
                    logger.info("Mirror sampling")
                    noise_half = self._rng.standard_normal(
                        (self.sample_batch_size // 2, self.solution_dim),
                        dtype=self.archive.dtype,
                    )
                    self._sample_noise = np.concatenate(
                        (noise_half, -noise_half))
                    samples = (mean[None] +
                               self._sample_sigma * self._sample_noise)
                else:
                    logger.info("No mirror sampling")
                    self._sample_noise = self._rng.standard_normal(
                        (self.sample_batch_size, self.solution_dim),
                        dtype=self.archive.dtype,
                    )
                    samples = (mean[None] +
                               self._sample_sigma * self._sample_noise)
            else:
                samples = []

            return [mean] + list(samples)

        # NOTE: These bounds are for the gradient coefficients. We currently do
        # not apply the solution bounds (i.e. self.lower_bounds and
        # self.upper_bounds).
        lower_bounds = np.full(self._num_coefficients,
                               -np.inf,
                               dtype=self._archive.dtype)
        upper_bounds = np.full(self._num_coefficients,
                               np.inf,
                               dtype=self._archive.dtype)
        noise = self.opt.ask(lower_bounds, upper_bounds)
        self._grad_coefficients = noise
        noise = np.expand_dims(noise,
                               axis=2)  # Size: (batch, behavior_dim + 1, 1)
        offset = np.sum(np.multiply(self._jacobians, noise), axis=1)
        sols = offset + self._gradient_opt.theta

        if self._greedy_eval:
            logger.info("Deploying greedy solution")
            greedy = td3.first_actor()
            sols = [greedy] + list(sols)

        logger.info("Emitting %d sols", len(sols))

        return sols

    def _check_restart(self, num_parents):
        """Emitter-side checks for restarting the optimizer.

        The optimizer also has its own checks.
        """
        if self._restart_rule == "no_improvement":
            return num_parents == 0
        return False

    def _calc_jacobian_from_samples(self, objs, bcs):
        jacobian = []

        for i in range(1 + bcs.shape[1]):
            if i == 0:
                logger.info("Gradient for obj")
            else:
                logger.info("Gradient for bc %d", i - 1)

            # Rank normalization.
            if self._sample_rank_norm:
                logger.info("Rank normalization")
                ranking_indices = (np.argsort(objs)
                                   if i == 0 else np.argsort(bcs[:, i - 1]))

                # Assign ranks -- ranks[i] tells the rank of noise[i].
                ranks = np.empty(self.sample_batch_size, dtype=np.int32)
                ranks[ranking_indices] = np.arange(self.sample_batch_size)

                # Normalize ranks to [-0.5, 0.5].
                ranks = (ranks / (self.sample_batch_size - 1)) - 0.5
                logger.info("Normalized min & max rank: %f %f", np.min(ranks),
                            np.max(ranks))

                objs = ranks
            else:
                logger.info("No rank normalization")

            # Compute the gradient.
            logger.info("Computing gradient")
            if self._sample_mirror:
                half_batch = self.sample_batch_size // 2
                gradient = np.sum(
                    self._sample_noise[:half_batch] *
                    (objs[:half_batch] - objs[half_batch:])[:, None],
                    axis=0)
                gradient /= half_batch * self._sample_sigma
            else:
                gradient = np.sum(self._sample_noise * objs[:, None], axis=0)
                gradient /= self.sample_batch_size * self._sample_sigma

            jacobian.append(gradient)

        self._sample_noise = None  # Reset.
        return np.asarray(jacobian)

    def tell(self,
             solutions,
             objective_values,
             behavior_values,
             metadata=None,
             jacobians=None,
             grad_estimate=None,
             td3: TD3 = None):
        """Gives the emitter results from evaluating solutions.

        Args:
            solutions (numpy.ndarray): Array of solutions generated by this
                emitter's :meth:`ask()` method.
            objective_values (numpy.ndarray): 1D array containing the objective
                function value of each solution.
            behavior_values (numpy.ndarray): ``(n, <behavior space dimension>)``
                array with the behavior space coordinates of each solution.
            metadata (numpy.ndarray): 1D object array containing a metadata
                object for each solution.
            grad_estimate: Whether this tell() is being called for a gradient
                estimate.
            jacobians (numpy.ndarray): For each solution, this array holds the
                jacobian of the objectives followed by the jacobians of the
                BCs / measures. Only use with grad_estimate=True.
            td3 (TD3): TD3 object for calculating policy gradients.
        """
        # If not inserting samples, only insert first solution (i.e. the mean).
        limiter = (range(1) if grad_estimate and not self._sample_insert else
                   itertools.repeat(None))

        # Handle the greedy solution.
        if not grad_estimate and self._greedy_eval:
            self._greedy_obj = objective_values[0]

            if self._greedy_insert:
                logger.info("Inserting greedy solution.")
                # Insert greedy solution here so it does not interfere with
                # optimizer solutions.
                self.archive.add(solutions[0], objective_values[0],
                                 behavior_values[0], metadata[0])
            else:
                logger.info("NOT inserting greedy solution.")

            solutions = solutions[1:]
            objective_values = objective_values[1:]
            behavior_values = behavior_values[1:]
            metadata = metadata[1:]

        ranking_data = []
        new_sols = 0
        metadata = itertools.repeat(None) if metadata is None else metadata
        for i, (sol, obj, beh, meta, _) in enumerate(
                zip(solutions, objective_values, behavior_values, metadata,
                    limiter)):
            status, value = self.archive.add(sol, obj, beh, meta)
            ranking_data.append((status, value, i))
            if status in (AddStatus.NEW, AddStatus.IMPROVE_EXISTING):
                new_sols += 1

        if grad_estimate:
            # Record jacobians but avoid updating optimizers.
            if self._gradient_source == "sample":
                logger.info("Calculating jacobian from samples")
                # First solution was the mean, so we exclude it.
                jacobians = self._calc_jacobian_from_samples(
                    objective_values[1:],
                    behavior_values[1:],
                )[None]
            elif self._gradient_source == "td3_and_sample":
                logger.info("Calculating jacobian from samples and TD3")
                sample_jacobians = self._calc_jacobian_from_samples(
                    objective_values[1:],
                    behavior_values[1:],
                )[None]
                td3_jacobians = td3.jacobian_batch(solutions[0][None])
                jacobians = np.concatenate((sample_jacobians, td3_jacobians),
                                           axis=1)
            elif self._gradient_source == "td3_sample":
                logger.info("Calculating jacobian from samples and replacing "
                            "objective gradient with TD3")
                jacobians = self._calc_jacobian_from_samples(
                    objective_values[1:],
                    behavior_values[1:],
                )[None]  # (1, num_coefficients, sol_dim)
                td3_jacobians = td3.jacobian_batch(
                    solutions[0][None])  # (1, 1, sol_dim)
                jacobians[0, 0, :] = td3_jacobians[0, 0, :]
            elif self._gradient_source == "td3":
                logger.info("Calculating jacobian from TD3")
                jacobians = td3.jacobian_batch(solutions[0][None])
            else:
                assert jacobians is not None, \
                    "jacobians must be provided b/c there is no gradient source"

            if self._normalize_gradients:
                norms = np.linalg.norm(jacobians, axis=2, keepdims=True)
                norms += 1e-8  # Make this configurable later
                jacobians /= norms

            if self._polyak is None:
                self._jacobians = jacobians
            else:
                # With polyak, the new jacobians are accumulated into the
                # current jacobians.
                if self._jacobians is None:
                    self._jacobians = jacobians
                else:
                    self._jacobians = (self._polyak * jacobians +
                                       (1.0 - self._polyak) * self._jacobians)
            return

        # New solutions sort ahead of improved ones, which sort ahead of ones
        # that were not added.
        ranking_data.sort(reverse=True)
        indices = [d[2] for d in ranking_data]
        logger.info("Ranking data: %s", ranking_data)

        num_parents = (new_sols if self._selection_rule == "filter" else
                       self._num_parents)

        self.opt.tell(self._grad_coefficients[indices], num_parents)

        # Calculate a new mean in solution space.
        parents = solutions[indices]
        parents = parents[:num_parents]
        weights = (np.log(num_parents + 0.5) -
                   np.log(np.arange(1, num_parents + 1)))
        total_weights = np.sum(weights)
        weights = weights / total_weights
        new_mean = np.sum(parents * np.expand_dims(weights, axis=1), axis=0)

        # Use the mean to calculate a gradient step and step the optimizer.
        gradient_step = new_mean - self._gradient_opt.theta
        self._gradient_opt.step(gradient_step)

        # Check for reset.
        if (self.opt.check_stop([value for status, value, i in ranking_data]) or
                self._check_restart(new_sols)):
            self._gradient_opt.reset(self.archive.get_random_elite()[0])
            measure_x0 = np.zeros(self._num_coefficients)
            self.opt.reset(measure_x0)
            self._jacobians = None
            self._restarts += 1
