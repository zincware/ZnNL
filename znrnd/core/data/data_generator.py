"""
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincware Project.

Description: Module for the parent class of the data generator.
"""
import abc

import numpy as np


class DataGenerator(metaclass=abc.ABCMeta):
    """
    Parent class for the data generator.
    """

    data_pool: np.array

    @abc.abstractmethod
    def get_points(self, n_points: int):
        """
        Fetch N points from the data pool.

        Parameters
        ----------
        n_points : int
                Number of points to fetch.

        Returns
        -------

        """
        raise NotImplemented("Implemented in the child class.")

    def __len__(self):
        """
        Return the size of the data pool.

        Returns
        -------
        data_pool_length : int
                Number of points in the data pool.
        """
        return len(self.data_pool)
