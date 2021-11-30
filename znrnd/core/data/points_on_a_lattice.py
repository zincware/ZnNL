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
Module to generate points on a lattice.
"""
from abc import ABC
from znrnd.core.data.data_generator import DataGenerator
import random
import numpy as np


class PointsOnLattice(DataGenerator, ABC):
    """
    class to generate points on a circle
    """

    def __init__(self):
        """
        constructor for points on a lattice.
        """
        self.data_pool = None

    def build_pool(self, x_points: int = 10, y_points: int = 10):
        """
        Build the data pool for points on a square lattice with spacing 1.

        Parameters
        ----------
        x_points : int
                Number of points in the x direction.
        y_points : int
                Number of points in the y direction.

        Returns
        -------
        Will call a method which updates the class state.
        """
        x = np.linspace(-x_points / 2, x_points / 2, x_points + 1, dtype=int)
        y = np.linspace(-y_points / 2, y_points / 2, y_points + 1, dtype=int)

        grid = np.stack(np.meshgrid(x, y), axis=2)
        self.data_pool = grid.reshape(-1, grid.shape[-1])

    def get_points(self, n_points: int) -> np.ndarray:
        """
        Fetch N points from the data pool.

        Parameters
        ----------
        n_points : int
                Number of points to fetch.

        Returns
        -------
        data : np.ndarray
                A numpy array of data points.
        """

        if n_points == -1:
            return self.data_pool
        else:
            try:
                indices = random.sample(range(0, len(self.data_pool) - 1), n_points)
                data = self.data_pool[indices].astype(np.float)
            except ValueError:
                indices = random.sample(
                    range(0, len(self.data_pool) - 1), len(self.data_pool)
                )
                data = self.data_pool[indices].astype(np.float)
            return data
