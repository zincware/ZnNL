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
        Constructor for the Mahalanobis Distance
        Parameters
        ----------
        """
        # Class defined attributes
        self.cov = None
        self.decomposed = None
        self.pool = None
        self.point_1 = None
        self.point_2 = None

        self.euclidean = LPNorm(order=2)

    def __call__(self, point_1: tf.Tensor, point_2: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Call the distance metric.

        Mahalanobis Distance between points in the point_1 tensor will be computed
        between those in the point_2 tensor element-wise. Therefore, we will have:

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
        self.point_1, self.point_2 = point_1, point_2
        self._update_covariance_matrix()
        self._compute_cholesky_decomposition()

        point_1_rescaled = tf.matmul(self.point_1, self.decomposed)
        point_2_rescaled = tf.matmul(self.point_2, self.decomposed)
        return self.euclidean(point_1_rescaled, point_2_rescaled)

    def _update_covariance_matrix(self):
        """
        Updates / computes the inverse covariance matrix of the representation point_1
        point_1 : tf.tensor
                neural network representation
        Returns
        -------
        """
        self.cov = tf.linalg.inv(tfp.stats.covariance(self.point_1))

    def _compute_cholesky_decomposition(self):
        """
        Returns
        -------
        The Cholesky decomposition of the the covariance matrices of both points
        """
        self.decomposed = tf.linalg.cholesky(self.cov)
