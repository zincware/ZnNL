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
import numpy as onp

from znrnd.core.agents.agent import Agent
from znrnd.core.data.data_generator import DataGenerator


class RandomAgent(Agent):
    """
    Class for the random agent.
    """

    target_set: np.ndarray

    def __init__(self, data_generator: DataGenerator):
        """
        Constructor for the random agent.
        """
        self.data_generator = data_generator

    def _get_indices(self, n_points: int):
        """
        Get the uniform random indices.

        Parameters
        ----------
        n_points : int
                Number of points to generate.
        """
        rng = jax.random.PRNGKey(onp.random.randint(1981))

        indices = jax.random.randint(
            rng, shape=(n_points,), minval=0, maxval=len(self.data_generator) - 1
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

        return self.target_set
