"""
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincware Project.

Description: Module for the distance metrics.
"""
import tensorflow as tf
from pyrnd.core.data.data_generator import DataGenerator
from pyrnd.core.models.model import Model
import tensorflow_probability as tfp


def euclidean_distance(point_1: tf.Tensor, point_2: tf.Tensor):
    """
    Compute the Euclidean distance metric.

    Parameters
    ----------
    point_1 : tf.Tensor
            First point in the comparison.
    point_2 : tf.Tensor
            Second point in the comparison.
    """
    return tf.norm(
        tf.cast(point_1, dtype=tf.float64) - tf.cast(point_2, dtype=tf.float64), axis=1
    )


class MahalanobisDist:
    """
    Class to compute the Mahalanobis Distance
    """

    def __init__(self,
                 data_generator: DataGenerator,
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
        self.cov_point_1 = None
        self.cov_point_2 = None
        self.decomposed_point_1 = None
        self.decomposed_point_2 = None
        self.pool = None
        self.point_1 = None
        self.point_2 = None

    def _update_covariance_matrix(self, point_1, point_2):
        """
        Updates the covariance matrix of both representations point_1 and point_2.

        When only the seed point exists, it creates the whole pool to calculate
        the covariance matrices.

        When fitting the model, the covariance matrices still have to contain the
        information of the whole data set, not only the target set.
        Based on all data points, the covariance matrices have to be calculated first
        TODO: not implemented yet - fallunterscheidung wenn man target set einliest

        point_1 : tf.tensor
                neural network representation
        point_2 : tf.tensor
                neural network representation
        Returns
        -------
        The covariance matrices, based on the current representations
        """
        if self.cov_point_1 is None:
            self.pool = self.generator.get_points(-1)
            self.point_1, self.point_2 = self._compute_representation(self.pool)
        else:
            self.point_1, self.point_2 = point_1, point_2
        self.cov_point_1 = tfp.stats.covariance(self.point_1)
        self.cov_point_2 = tfp.stats.covariance(self.point_2)

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
        self.decomposed_point_1 = tf.linalg.cholesky(self.cov_point_1)
        self.decomposed_point_2 = tf.linalg.cholesky(self.cov_point_2)

    def compute_mahalanobis_distance(self, point_1, point_2):
        """
        Computes the Mahalanobis Distance, based on the Cholesky decomposition of
        both representations.
        Returns the Euclidean Distance of the rescaled representations.
        Returns
        -------
        Returns the Mahalanobis distance between the representations of point_1 and
        point_2
        """
        self._update_covariance_matrix(point_1=point_1, point_2=point_2)
        self._compute_cholesky_decomposition()
        point_1_rescaled = tf.matmul(self.point_1, self.decomposed_point_1)
        point_2_rescaled = tf.matmul(self.point_2, self.decomposed_point_2)
        return euclidean_distance(point_1_rescaled, point_2_rescaled)
