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
Test the order n norm metric.
"""
import unittest
import znrnd
import tensorflow as tf
import numpy as np


class TestOrderNDifference(unittest.TestCase):
    """
    Class to test the cosine distance measure module.
    """

    def test_order_2_distance(self):
        """
        Test the cosine similarity measure.

        Returns
        -------
        Assert the correct answer is returned for orthogonal, parallel, and
        somewhere in between.
        """
        metric = znrnd.distance_metrics.OrderNDifference(order=2)

        # Test orthogonal vectors
        point_1 = tf.convert_to_tensor([[1.0, 7.0, 0.0, 0.0]], dtype=tf.float32)
        point_2 = tf.convert_to_tensor([[1.0, 1.0, 0.0, 0.0]], dtype=tf.float32)
        self.assertEqual(metric(point_1, point_2), [36.0])

    def test_order_3_distance(self):
        """
        Test the cosine similarity measure.

        Returns
        -------
        Assert the correct answer is returned for orthogonal, parallel, and
        somewhere in between.
        """
        metric = znrnd.distance_metrics.OrderNDifference(order=3)

        # Test orthogonal vectors
        point_1 = tf.convert_to_tensor([[1.0, 1.0, 0.0, 0.0]], dtype=tf.float32)
        point_2 = tf.convert_to_tensor([[1.0, 7.0, 0.0, 0.0]], dtype=tf.float32)
        np.testing.assert_almost_equal(metric(point_1, point_2), [-216.0], decimal=4)

    def test_multi_distance(self):
        """
        Test the cosine similarity measure.

        Returns
        -------
        Assert the correct answer is returned for orthogonal, parallel, and
        somewhere in between.
        """
        metric = znrnd.distance_metrics.OrderNDifference(order=3)

        # Test orthogonal vectors
        point_1 = tf.convert_to_tensor(
            [[1.0, 7.0, 0.0, 0.0], [4, 7, 2, 1]], dtype=tf.float32
        )
        point_2 = tf.convert_to_tensor(
            [[1.0, 1.0, 0.0, 0.0], [6, 3, 1, 8]], dtype=tf.float32
        )
        np.testing.assert_almost_equal(
            metric(point_1, point_2), [216.0, -286.0], decimal=4
        )
