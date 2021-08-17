"""
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincware Project.

Description: greedy selection module test.
"""
import unittest
import pyrnd
import tensorflow as tf
import numpy as np


class Agent:
    """
    A dummy class to mimic an agent.
    """

    @staticmethod
    def generate_points(n_points: int = 1):
        """
        Generate dummy points.

        Parameters
        ----------
        n_points : int
                Number of points to generate.

        Returns
        -------
        points : tf.Tensor
        """
        if n_points == -1:
            return tf.convert_to_tensor([[0, 1], [1, 0]])

    @staticmethod
    def compute_distance(points: tf.Tensor):
        """
        Compute the distance between points.

        Parameters
        ----------
        points : tf.Tensor
                First set of points to look at

        Returns
        -------

        """
        return pyrnd.similarity_measures.cosine_similarity(
            points, tf.convert_to_tensor([[1, 0], [1, 0]])
        )


class TestGreedySelection(unittest.TestCase):
    """
    A class to test the greedy selection method.
    """

    def test_single_select(self):
        """
        Test the select points methods.
        """
        self.agent = Agent()
        self.selector = pyrnd.GreedySelection(self.agent)
        point = self.selector.select_points()

        np.testing.assert_array_equal(point.numpy(), np.array([0, 1]))
