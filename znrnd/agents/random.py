"""
ZnRND: A zincwarecode package.

License
-------
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the zincwarecode Project.

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
Module for the random agent.
"""
import jax
import jax.numpy as np

from znrnd.agents.agent import Agent
from znrnd.data import DataGenerator
from znrnd.utils.prng import PRNGKey


class RandomAgent(Agent):
    """
    Class for the random agent.
    """

    def __init__(self, data_generator: DataGenerator, seed: int = None):
        """
        Constructor for the random agent.

        Parameters
        ----------
        data_generator : DataGenerator
                Data generator object from which data should be picked.
        seed : int, default None
                Random seed for the RNG.
        """
        self.data_generator = data_generator
        self.target_set: np.ndarray
        self.target_indices: list
        self.rng = PRNGKey(seed)

    def _get_indices(self, n_points: int):
        """
        Get the uniform random indices.

        Parameters
        ----------
        n_points : int
                Number of points to generate.
        """
        indices = jax.random.choice(
            key=self.rng.key,
            a=len(self.data_generator) - 1,
            shape=(n_points,),
            replace=False,
        )

        return indices

    def build_dataset(
        self, target_size: int = None, visualize: bool = False, report: bool = True
    ):
        """
        Build the dataset.

        See parent class for full documentation.
        """
        indices = self._get_indices(target_size)

        self.target_set = np.take(self.data_generator.data_pool, indices, axis=0)
        self.target_indices = indices

        return self.target_set
