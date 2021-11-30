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
Module for the Mahalanobis distance.
"""
from .distance_metric import DistanceMetric
import tensorflow as tf
import tensorflow_probability as tfp
from .l_p_norm import LPNorm


class MahalanobisDistance(DistanceMetric):
    """
    Compute the mahalanobis distance between points.
    """

    def __init__(self):
        """
        Constructor for the Mahalanobis distance.
        """
        self.euclidean = LPNorm(order=2)

    @staticmethod
    def _compute_covariance(distribution: tf.Tensor) -> tf.Tensor:
        """
        Compute the covariance on the distribution.

        Parameters
        ----------
        distribution : tf.Tensor
                Distribution on which to compute the covariance.

        Returns
        -------
        covariance: tf.Tensor shape=(n_points, n_points, n_dim)
                Covariance matrix.
        """
        covariance = tfp.stats.covariance(distribution)
        covariance_half = tf.linalg.cholesky(covariance)

        return covariance_half

    def __call__(self, point_1: tf.Tensor, point_2: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Call the distance metric.

        Distance between points in the point_1 tensor will be computed between those in
        the point_2 tensor element-wise. Therefore, we will have:

                point_1[i] - point_2[i] for all i.

        Parameters
        ----------
        point_1 : tf.Tensor (n_points, point_dimension)
            First set of points in the comparison.
        point_2 : tf.Tensor (n_points, point_dimension)
            Second set of points in the comparison.
        kwargs
                Miscellaneous keyword arguments for the specific metric.

        Returns
        -------
        d(point_1, point_2) : tf.tensor : shape=(n_points, 1)
                Array of distances for each point.
        """
        covariance = self._compute_covariance(point_1)
        point_1_maha = tf.matmul(point_1, covariance)
        point_2_maha = tf.matmul(point_2, covariance)
        return self.euclidean(point_1_maha, point_2_maha)
