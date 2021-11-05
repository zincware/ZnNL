"""
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincware Project.

Description: Module for the similarity measures.

Notes
-----
These similarity measures will return 1 - S(A, B). This is because we need
a quasi-distance for the comparison to occur.
"""
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
import tensorflow.keras


class SimilarityMeasures:
    """
    Parent Class for the similarity measurements module
    """

    def calculate(self, point_1, point_2) -> tf.Tensor:
        """
        Calculate the similarity of the given points

        Parameters
        ----------
        point_1 : tf.Tensor
                first neural network representation of the considered points
        point_2 :
                second neural network representation of the considered points

        Returns
        -------
        Similarity measurement : tf.Tensor
                Similarity measurement of each point individually
        """
        raise NotImplementedError

    def __call__(self, point_1: tf.Tensor, point_2: tf.Tensor) -> float:
        """
        Summation over the tensor of the respective similarity measurement
        Parameters
        ----------
        point_1 : tf.Tensor
                first neural network representation of the considered points
        point_2 :
                second neural network representation of the considered points

        Returns
        -------
        loss : float
                total loss of all points based on the similarity measurement
        """
        return tf.reduce_mean(self.calculate(point_1, point_2))


class CosineSim(SimilarityMeasures):
    """
    Cosine similarity between two representations
    """

    def calculate(self, point_1, point_2):
        """
        Parameters
        ----------
        point_1 : tf.Tensor
                First point in the comparison.
        point_2 : tf.Tensor
                Second point in the comparison.
        TODO: include factor sqrt2 that rescales on a real distance metric (look up)
        """
        numerator = tf.cast(tf.einsum("ij, ij -> i", point_1, point_2), tf.float32)
        denominator = tf.sqrt(
            tf.cast(

                # tf.einsum("ij, ij, ij, ij -> i", point_1, point_1, point_2, point_2)

                tf.einsum("ij, ij -> i", point_1, point_1)
                * tf.einsum("ij, ij -> i", point_2, point_2),
                tf.float32,
            )
        )
        return 1 - abs(tf.divide(numerator, denominator))


class AngleSim(SimilarityMeasures):
    """
    Angle between two representations normalized by pi
    """

    def calculate(self, point_1, point_2):
        """
        Parameters
        ----------
        point_1 : tf.Tensor
                First point in the comparison.
        point_2 : tf.Tensor
                Second point in the comparison.
        """
        numerator = tf.cast(tf.einsum("ij, ij -> i", point_1, point_2), tf.float32)
        denominator = tf.sqrt(
            tf.cast(
                tf.einsum("ij, ij -> i", point_1, point_1)
                * tf.einsum("ij, ij -> i", point_2, point_2),
                tf.float32,
            )
        )
        return tf.acos(abs(tf.divide(numerator, denominator)))/np.pi


class MSE(SimilarityMeasures):
    """
    Mean square error between two representations
    """

    def calculate(self, point_1, point_2):
        """
        Parameters
        ----------
        point_1 : tf.Tensor
                First point in the comparison.
        point_2 : tf.Tensor
                Second point in the comparison.
        """

        diff = point_1 - point_2
        return tf.cast(tf.einsum("ij, ij -> i", diff, diff), tf.float32)


class EuclideanDist(SimilarityMeasures):
    """
    Compute the Euclidean distance metric between two representations
    """

    def calculate(self, point_1, point_2) -> tf.Tensor:
        """
        Parameters
        ----------
        point_1 : tf.Tensor
                First point in the comparison.
        point_2 : tf.Tensor
                Second point in the comparison.
        """
        return tf.cast(tf.norm(point_1 - point_2, axis=1), tf.float32)


class MahalanobisDist(SimilarityMeasures):
    """
    Compute Mahalanobis Distance metric between two representations
    """

    def calculate(self, point_1, point_2) -> tf.Tensor:
        """
        Parameters
        ----------
        point_1 : tf.Tensor
                First point in the comparison.
        point_2 : tf.Tensor
                Second point in the comparison.
        TODO: search for the fehler cause doesn't work
        """

        covariance = tfp.stats.covariance(point_1)
        covariance_half = tf.linalg.cholesky(covariance)
        print(covariance_half)
        point_1_maha = tf.matmul(point_1, covariance_half)
        point_2_maha = tf.matmul(point_2, covariance_half)
        return EuclideanDist().calculate(point_1_maha, point_2_maha)
