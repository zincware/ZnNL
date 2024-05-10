"""
ZnNL: A Zincwarecode package.

License
-------
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincwarecode Project.

Contact Information
-------------------
email: zincwarecode@gmail.com
github: https://github.com/zincware
web: https://zincwarecode.com/

Citation
--------
If you use this module please cite us with:

Summary
-------
"""

import jax
import jax.numpy as np

from znnl.agents.agent import Agent
from znnl.analysis.entropy import EntropyAnalysis
from znnl.data import DataGenerator
from znnl.models import JaxModel
from znnl.utils.prng import PRNGKey
from znnl.ntk_computation.jax_ntk import JAXNTKComputation


class ApproximateMaximumEntropy(Agent):
    """Class for the approximate maximum entropy data selection agent."""

    def __init__(
        self,
        target_network: JaxModel,
        data_generator: DataGenerator,
        samples: int = 10,
        seed: int = None,
    ):
        """
        Constructor for the Approximate maximum entropy agent.

        Parameters
        ----------
        target_network : Jax_Model
                Model of the target network.
        samples : int (default=10)
                Number of samples to try to get the maximum entropy.
        data_generator : DataGenerator
                Data generator from which samples are taken.
        seed : int, default None
                Random seed for the RNG.
        """
        self.target_network = target_network
        self.data_generator = data_generator
        self.samples = samples
        self.rng = PRNGKey(seed)

        self.target_set: np.ndarray
        self.target_indices: list
        self.ntk_computation = JAXNTKComputation(target_network.ntk_apply_fn, trace_axes=(-1,))

    def _compute_entropy(self, dataset: np.ndarray):
        """
        Compute the entropy of the dataset.

        Parameters
        ----------
        dataset : np.ndarray
                Dataset on which to compute the entropy.

        Returns
        -------
        entropy : float
                Entropy pf the dataset.
        """
        ntk = self._compute_ntk(dataset)

        entropy_calculator = EntropyAnalysis(matrix=ntk)

        return entropy_calculator.compute_von_neumann_entropy()

    def _compute_ntk(self, dataset: np.ndarray):
        """
        Compute the neural tangent kernel.

        Returns
        -------
        empirical_ntk : np.ndarray
                The empirical NTK matrix of the target network.
        """
        return self._compute_ntk.compute_ntk({"params": self.target_network.model_state.params}, dataset)

    def build_dataset(
        self, target_size: int = None, visualize: bool = False, report: bool = True
    ):
        """
        Run the random network distillation methods and build the target set.

        Parameters
        ----------
        target_size : int
                Target size of the operation.
        visualize : bool (default=False)
                If true, a t-SNE visualization will be performed on the final models.
        report : bool (default=True)
                If true, print a report about the RND performance.

        Returns
        -------
        target_set : list
                Returns the newly constructed target set.
        """
        max_index = int(len(self.data_generator) - 1)

        samples = jax.random.choice(
            key=self.rng(),
            a=max_index,
            shape=(self.samples, target_size),
            replace=False,
        )

        entropy_array = np.zeros(self.samples)
        for idx, sample in enumerate(samples):
            data = np.take(self.data_generator.data_pool, sample, axis=0)
            entropy = self._compute_entropy(data)
            entropy_array = entropy_array.at[idx].set(entropy)

        max_set = samples[np.argmax(entropy_array)]

        self.target_set = np.take(self.data_generator.data_pool, max_set, axis=0)
        self.target_indices = max_set

        return self.target_set
