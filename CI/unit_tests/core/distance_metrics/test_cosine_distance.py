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
Test the cosine distance module.
"""
import unittest

import numpy as np
import tensorflow as tf

import znrnd


class TestCosineDistance(unittest.TestCase):
    """
    Class to test the cosine distance measure module.
    """

    def test_cosine_distance(self):
        """
        Test the cosine similarity measure.

        Returns
        -------
        Assert the correct answer is returned for orthogonal, parallel, and
        somewhere in between.
        """
        metric = znrnd.distance_metrics.CosineDistance()

        # Test orthogonal vectors
        point_1 = tf.convert_to_tensor([[1, 0, 0, 0]])
        point_2 = tf.convert_to_tensor([[0, 1, 0, 0]])
        self.assertEqual(metric(point_1, point_2), [1])

        # Test parallel vectors
        point_1 = tf.convert_to_tensor([[1, 0, 0, 0]])
        point_2 = tf.convert_to_tensor([[1, 0, 0, 0]])
        self.assertEqual(metric(point_1, point_2), [0])

        # Somewhere in between
        point_1 = tf.convert_to_tensor([[1.0, 0, 0, 0]])
        point_2 = tf.convert_to_tensor([[0.5, 1.0, 0, 3.0]])
        self.assertEqual(metric(point_1, point_2), [0.84382623])

    def test_multiple_distances(self):
        """
        Test the cosine similarity measure.

        Returns
        -------
        Assert the correct answer is returned for orthogonal, parallel, and
        somewhere in between.
        """
        metric = znrnd.distance_metrics.CosineDistance()

        # Test orthogonal vectors
        point_1 = tf.convert_to_tensor([[1, 0, 0, 0], [1, 0, 0, 0], [1.0, 0, 0, 0]])
        point_2 = tf.convert_to_tensor([[0, 1, 0, 0], [1, 0, 0, 0], [0.5, 1.0, 0, 3.0]])
        np.testing.assert_array_almost_equal(
            metric(point_1, point_2), [1, 0, 0.843826], decimal=6
        )
