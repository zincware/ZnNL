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
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import jax.numpy as np
from numpy.testing import assert_array_equal

from znnl.distance_metrics.cosine_distance import CosineDistance
from znnl.point_selection import GreedySelection


class Agent:
    """
    A dummy class to mimic an agent.
    """

    @staticmethod
    def generate_points(n_points: int = -1):
        """
        Generate dummy points.

        Parameters
        ----------
        n_points : int
                Number of points to generate.

        Returns
        -------
        points : np.array
        """
        if n_points == -1:
            return np.array([[0, 1], [1, 0]])

    @staticmethod
    def compute_distance(points: np.ndarray):
        """
        Compute the distance between points.

        Parameters
        ----------
        points : np.ndarray
                First set of points to look at

        Returns
        -------

        """
        metric = CosineDistance()
        return metric(points, np.array([[1, 0], [1, 0]]))


class TestGreedySelection:
    """
    A class to test the greedy selection method.
    """

    def test_single_select(self):
        """
        Test the select points methods.
        """
        agent = Agent()
        data = agent.generate_points(-1)
        distances = agent.compute_distance(data)

        self.selector = GreedySelection()
        point = self.selector.select_points(distances)
        assert_array_equal(data[point], np.array([0, 1]))
