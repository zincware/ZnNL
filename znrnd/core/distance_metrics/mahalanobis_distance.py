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
from znrnd.core.data.data_generator import DataGenerator
from znrnd.core.models.model import Model


class MahalanobisDistance(DistanceMetric):
    """
    Compute the mahalanobis distance between points.
    """

    def __init__(self,
                 data_generator: DataGenerator = None,
                 target_network: Model = None,
                 predictor_network: Model = None,
                 ):
        """
        Constructor for the Mahalanobis Distance
        Parameters
        ----------
        data_generator : objector
                Class to generate or select new points from the point cloud
                being studied.
        target_network : Model
                Model class for the target network
        predictor_network : Model
                Model class for the predictor.
        """

        # User defined attributes
        self.generator = data_generator
        self.target = target_network
        self.predictor = predictor_network

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

    def _compute_representation(self, pool):
        """
        Computes the representation of all points in the pool
        Parameters
        ----------
        pool : np.nd_array
                A numpy array of data points.
        Returns
        -------
        point_1 : tf.tensor
                Representation of the target network
        point_2 : tf.tensor
                Representation of the target network
        """
        point_1 = self.target.predict(pool)
        point_2 = self.predictor.predict(pool)
        return point_1, point_2

    def _compute_cholesky_decomposition(self):
        """
        Returns
        -------
        The Cholesky decomposition of the the covariance matrices of both points
        """
        self.decomposed = tf.linalg.cholesky(self.cov)
