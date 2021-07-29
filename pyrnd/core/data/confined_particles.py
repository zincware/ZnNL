"""
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincware Project.

Description: Module for the particles in a box generator class.
"""
from abc import ABC
from pyrnd.core.data.data_generator import DataGenerator
import tensorflow as tf
import numpy as np
import random


class ConfinedParticles(DataGenerator, ABC):
    """
    A class to generate data for particles in a box.
    """

    def __init__(self, box_length: float = 2.0, dimension: int = 2.0):
        """
        Constructor for the ConfinedParticles data generator.

        Parameters
        ----------
        box_length : float
                Side length of box
        dimension : int
                Number of dimensions to consider.
        """
        self.box_length = box_length
        self.dimension = dimension

        self.data_pool = None

    def _generate_points(self, n_points: int):
        """
        Generate points in the box.

        Parameters
        ----------
        n_points : int
                Number of points to generate.

        Returns
        -------

        """
        return tf.random.uniform((n_points, int(self.dimension)),
                                 minval=0,
                                 maxval=self.box_length)

    def build_pool(self, n_points: int = 100, data: np.ndarray = None):
        """
        Build a pool of data. Append to an existing one if it already exists.

        Parameters
        ----------
        n_points : int
                Number of points to add to the pool.
        data : np.ndarray
                Data that has been generated elsewhere but should be added to
                the pool.

        Returns
        -------

        """
        if data is None:
            if self.data_pool is None:
                self.data_pool = self._generate_points(n_points).numpy()
            else:
                new_data = self._generate_points(n_points).numpy()
                self.data_pool = np.concatenate((self.data_pool, new_data),
                                                axis=0)
        else:
            self.data_pool = np.concatenate((self.data_pool, data), axis=0)

    def _return_new_data(self, n_points: int):
        """
        Return new data points.

        Parameters
        ----------
        n_points : int
                Number of new points to return.

        Returns
        -------
        points : np.ndarray
                A numpy array of new data points.

        Notes
        -----
        This is only a method as it is called twice.
        """
        points = self._generate_points(n_points)
        self.build_pool(n_points, points)
        return points

    def get_points(self, n_points: int, generate: bool = False):
        """
        Fetch N points from the data pool.

        Parameters
        ----------
        n_points : int
                Number of points to fetch. If -1, all points are returned.
        generate : bool
                If true, generate new data.

        Returns
        -------
        data : np.ndarray
                A numpy array of data points.
        """
        if generate:
            return self._return_new_data(n_points)
        else:
            try:
                indices = random.sample(range(0, len(self.data_pool) - 1),
                                        n_points)
                data = self.data_pool[indices]
            except ValueError:
                data = self._return_new_data(n_points)

            return data
