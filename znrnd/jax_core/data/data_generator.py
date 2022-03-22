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
Data generator module.
"""
import abc
import logging

import jax.numpy as np
import jax.random

logger = logging.getLogger(__name__)


class DataGenerator(metaclass=abc.ABCMeta):
    """
    Parent class for the data generator.
    """

    data_pool: np.array

    def get_points(self, n_points: int, method: str = "first"):
        """
        Fetch data from the data pool.

        Parameters
        ----------
        n_points : int
                Number of points to fetch.
        method : str (default= first)
                How to select the points.
                Method include:
                    * first -- select the first N points
                    * uniform -- uniformly select N points starting from 0
                    * random -- randomly select N points

        Returns
        -------
        dataset : list
                List of selected data.
        """
        if n_points == -1:
            n_points = len(self.data_pool)
        if n_points > len(self.data_pool):
            logger.info("Too many points requested, returning full point cloud.")
            return self.data_pool

        if method == "uniform":
            indices = np.linspace(0, len(self.data_pool) - 1, n_points, dtype=int)
        elif method == "random":
            key = jax.random.PRNGKey(3)
            indices = jax.random.randint(
                key=key,
                shape=(n_points,),
                minval=0,
                maxval=len(self.data_pool) - 1,
                dtype=int,
            )
        elif method == "first":
            indices = [i for i in range(n_points)]
        dataset = []

        for item in indices:
            dataset.append(self[item])

        return dataset

    def __len__(self):
        """
        Return the size of the data pool.

        Returns
        -------
        data_pool_length : int
                Number of points in the data pool.
        """
        return len(self.data_pool)

    def __getitem__(self, idx: int):
        """
        Get item dunder for the generator.

        Parameters
        ----------
        idx

        Returns
        -------
        item : Any
                Data in the data pool at index idx. Can be anything.

        Notes
        -----
        This method will often be overwritten in favour of a more tailored data
        collection method. However for trivial systems, can be used as is.
        """
        return self.data_pool[idx]
