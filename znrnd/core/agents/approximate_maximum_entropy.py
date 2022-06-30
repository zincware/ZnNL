"""
ZnRND: A Zincwarecode package.

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
Module for the approximate maximum entropy agent.
"""
import jax
import jax.numpy as np
import numpy as onp

from znrnd.core.agents.agent import Agent
from znrnd.core.analysis.entropy import EntropyAnalysis
from znrnd.core.data.data_generator import DataGenerator
from znrnd.core.models.model import Model


class ApproximateMaximumEntropy(Agent):
    """
    Class for the approximate maximum entropy data selection agent.

    Attributes
    ----------
    """

    def __init__(
        self, target_network: Model, data_generator: DataGenerator, samples: int = 10
    ):
        """
        Constructor for the Approximate maximum entropy agent.

        Parameters
        ----------
        target_network : Model
                Model of the target network.
        samples : int (default=10)
                Number of samples to try to get the maximum entropy.
        data_generator : DataGenerator
                Data generator from which samples are taken.

        """
        self.target_network = target_network
        self.data_generator = data_generator
        self.samples = samples

        self.target_set: np.ndarray
        self.target_indices: list

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
        return self.target_network.compute_ntk(dataset)["empirical"]

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

        rng = jax.random.PRNGKey(onp.random.randint(684518))
        samples = jax.random.randint(
            rng, shape=(self.samples, target_size), minval=0, maxval=max_index
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
