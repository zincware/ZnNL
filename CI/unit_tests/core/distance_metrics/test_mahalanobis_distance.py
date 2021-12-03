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
Test the angular distance module.
"""
import unittest

import numpy.random
import scipy.spatial.distance
import znrnd
import tensorflow as tf
import numpy as np


class TestMahalanobisDistance(unittest.TestCase):
    """
    Class to test the cosine distance measure module.
    """

    def test_mahalanobis_distance(self):
        """
        Test the Mahalanobis distance on functionality by comparing results a
        test Mahalanobis distance from scipy.

        Returns
        -------
        Assert if the Mahalanobis distance returns true values for sample set of
        random normal distributed points in two dimensions.
        """
        metric = znrnd.distance_metrics.MahalanobisDistance()

        # Create sample set
        point_1, point_2 = self.create_sample_set()

        # Calculate results from distance metric
        metric_results = metric(
            tf.convert_to_tensor(point_1, dtype=tf.float32),
            tf.convert_to_tensor(point_2, dtype=tf.float32),
        )

        # Calculate test results from numpy distance metric
        test_metric_results = []
        self.calculate_numpy_mahalanobis_distance(point_1, point_2, test_metric_results)

        # Assert results
        np.testing.assert_almost_equal(
            metric_results.numpy(), np.array(test_metric_results), decimal=1
        )

    def test_identity(self):
        """
        Test the identity criterion of a metric, based on a randomly produced sample
        set (used to create the covariance matrix).

        Returns
        -------
        Asserts if the distance of the last point of point_1 and point_2 is equal to 0
        """
        # Create Sample set
        point_1, point_2 = self.create_sample_set()

        # Add point of interest
        point_of_interest = tf.convert_to_tensor([[7.0, 3.0]], dtype=tf.float32)
        point_1 = tf.concat(
            [tf.convert_to_tensor(point_1, dtype=tf.float32), point_of_interest], axis=0
        )
        point_2 = tf.concat(
            [tf.convert_to_tensor(point_2, dtype=tf.float32), point_of_interest], axis=0
        )

        # Assert identity
        metric = znrnd.distance_metrics.MahalanobisDistance()
        self.assertEqual(metric(point_1, point_2)[-1], 0)

    def test_symmetry(self):
        """
        Test the symmetry criterion of a metric, based on a randomly produced sample
        set (used to create the covariance matrix).

        Returns
        -------
        Asserts if the distances of the last two points of point_1 and point_2 are
        identical.
        """
        # Create Sample set
        point_1, point_2 = self.create_sample_set()

        # Add point of interest
        point_1_of_interest = tf.convert_to_tensor(
            [[-2.0, 5.0], [7.0, 3.0]], dtype=tf.float32
        )
        point_2_of_interest = tf.convert_to_tensor(
            [[7.0, 3.0], [-2.0, 5.0]], dtype=tf.float32
        )
        point_1 = tf.concat(
            [tf.convert_to_tensor(point_1, dtype=tf.float32), point_1_of_interest],
            axis=0,
        )
        point_2 = tf.concat(
            [tf.convert_to_tensor(point_2, dtype=tf.float32), point_2_of_interest],
            axis=0,
        )

        # Assert identity
        metric = znrnd.distance_metrics.MahalanobisDistance()
        self.assertEqual(metric(point_1, point_2)[-1], (metric(point_1, point_2)[-2]))

    @staticmethod
    def create_sample_set():
        """
        Returns
        -------
        Creates a random normal distributed sample set
        """
        point_1 = np.array(
            [np.random.normal(0, 10, 100), np.random.normal(0, 20, 100)]
        ).T
        point_2 = np.array(
            [np.random.normal(0, 10, 100), np.random.normal(0, 20, 100)]
        ).T
        return point_1, point_2

    @staticmethod
    def calculate_numpy_mahalanobis_distance(point_1, point_2, result_list):
        """
        Calculates the Mahalanobis distance based on a scipy integration.

        Parameters
        ----------
        point_1 : np.ndarray
                Set of points in the distance calculation.
        point_2 : np.ndarray
                Set of points in the distance calculation.
        result_list : list
                Results for each point are appended to this list.

        Returns
        -------
        Appends all calculated distances to the result_list.
        """
        inv_cov = np.linalg.inv(np.cov(point_1.T))
        for index in range(len(point_1.T[0, :])):
            result_list.append(
                scipy.spatial.distance.mahalanobis(
                    point_1[index], point_2[index], inv_cov
                )
            )
