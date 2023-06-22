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
from znnl.data import DataGenerator
from znnl.utils.prng import PRNGKey


class RandomAgent(Agent):
    """
    Class for the random agent.
    """

    def __init__(
        self,
        data_generator: DataGenerator,
        seed: int = None,
        class_uniform: bool = False,
    ):
        """
        Constructor for the random agent.

        Parameters
        ----------
        data_generator : DataGenerator
                Data generator object from which data should be picked.
        seed : int, default None
                Random seed for the RNG.
        class_uniform : bool, default False
                Whether to sample uniformly over classes.
                If True, the random agent will sample draw points from each class
                individually, and then concatenate them.
                If True, one-hot encoded labels are required.
        """
        self.data_generator = data_generator
        self.target_set: np.ndarray
        self.target_indices: list
        self.rng = PRNGKey(seed)
        self.class_uniform = class_uniform

    def _get_indices(self, data: np.ndarray, n_points: int):
        """
        Get the uniform random indices.

        Parameters
        ----------
        data : np.ndarray
                Data from which to draw indices.
        n_points : int
                Number of points to generate.
        """
        indices = jax.random.choice(
            key=self.rng(),
            a=len(data),
            shape=(n_points,),
            replace=False,
        )

        return indices

    def build_dataset(
        self, target_size: int, visualize: bool = False, report: bool = True
    ):
        """
        Build the dataset.

        Parameters
        ----------
        target_size : int
                Number of points to be selected by the agent.
        visualize : bool, default False
                Whether to visualize the selected points.
        report : bool, default True
                Whether to print a report of the selected points.

        See parent class for full documentation.
        """
        if self.class_uniform:
            indices = []

            # Get the labels of the data
            labels = self.data_generator.train_ds["targets"]
            n_classes = np.shape(labels)[1]
            labels = np.argmax(labels, axis=-1)

            # Distribute the number of points evenly over classes
            n_points = target_size // n_classes
            # Randomly select classes to the remaining points
            additional_points = target_size % n_classes
            class_additional_points = self._get_indices(
                np.arange(n_classes), additional_points
            )

            for i in range(n_classes):
                # Get the indices of the current class
                idx = np.where(labels == i)[0]
                # Randomly select indices from the current class
                if i in class_additional_points:
                    class_idx = self._get_indices(idx, n_points + 1)
                else:
                    class_idx = self._get_indices(idx, n_points)
                # Add the indices to the list
                indices.extend(idx[class_idx])

            indices = np.array(indices)

        else:
            # Get the indices
            indices = self._get_indices(self.data_generator, target_size)

        self.target_set = np.take(self.data_generator.data_pool, indices, axis=0)
        self.target_indices = indices.tolist()

        return self.target_set
