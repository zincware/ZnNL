"""
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincware Project.

Description: Test for the distance measures modules.
"""
import unittest
import pyrnd
import tensorflow as tf
import numpy as np


class TestDistanceMetrics(unittest.TestCase):
    """
    Class to test the distance metric module.
    """

    def test_euclidean_distance(self):
        """
        Test that the euclidean distance metric is working correctly

        Returns
        -------
        Asserts distances for cases of 0, 1 and in-between.

        Notes
        -----
        After running the test it is clear that we just cannot use such a
        metric to compare similarity.
        """
        metric = pyrnd.distance_metrics.euclidean_distance

        # Test equal length vectors
        point_1 = tf.convert_to_tensor([1.0, 0, 0, 3.0])
        point_2 = tf.convert_to_tensor([1.0, 0, 0, 3.0])
        self.assertEqual(metric(point_1, point_2), 0)
