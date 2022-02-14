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
Test the hyper sphere distance module.
"""
import unittest

import numpy as np
import tensorflow as tf

import znrnd


class TestCosineDistance(unittest.TestCase):
    """
    Class to test the cosine distance measure module.
    """

    def test_hyper_sphere_distance(self):
        """
        Test the hyper sphere distance.

        Returns
        -------
        Assert the correct answer is returned for orthogonal, parallel, and
        somewhere in between.
        """
        metric = znrnd.distance_metrics.HyperSphere(order=2)

        # Test orthogonal vectors
        point_1 = tf.convert_to_tensor([[1, 0, 0, 0]], dtype=tf.float32)
        point_2 = tf.convert_to_tensor([[0, 1, 0, 0]], dtype=tf.float32)
        self.assertEqual(metric(point_1, point_2), [1.41421356])

        # Test parallel vectors
        point_1 = tf.convert_to_tensor([[1, 0, 0, 0]], dtype=tf.float32)
        point_2 = tf.convert_to_tensor([[1, 0, 0, 0]], dtype=tf.float32)
        self.assertEqual(metric(point_1, point_2), [0])

        # Somewhere in between
        point_1 = tf.convert_to_tensor([[1.0, 0, 0, 0]], dtype=tf.float32)
        point_2 = tf.convert_to_tensor([[0.5, 1.0, 0, 3.0]], dtype=tf.float32)
        self.assertEqual(metric(point_1, point_2), [0.84382623 * np.sqrt(10.25)])

    def test_multiple_distances(self):
        """
        Test the hyper sphere distance.

        Returns
        -------
        Assert the correct answer is returned for orthogonal, parallel, and
        somewhere in between.
        """
        metric = znrnd.distance_metrics.HyperSphere(order=2)

        # Test orthogonal vectors
        point_1 = tf.convert_to_tensor(
            [[1, 0, 0, 0], [1, 0, 0, 0], [1.0, 0, 0, 0]], dtype=tf.float32
        )
        point_2 = tf.convert_to_tensor(
            [[0, 1, 0, 0], [1, 0, 0, 0], [0.5, 1.0, 0, 3.0]], dtype=tf.float32
        )
        np.testing.assert_array_almost_equal(
            metric(point_1, point_2),
            [np.sqrt(2), 0, 0.84382623 * np.sqrt(10.25)],
            decimal=6,
        )
