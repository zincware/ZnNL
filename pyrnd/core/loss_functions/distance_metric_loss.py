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
Module for the distance metric loss.
"""
from abc import ABC
import tensorflow as tf
from tensorflow.keras.losses import Loss


class DistanceMetricLoss(Loss, ABC):
    """
    Custom loss function for learning a distance metric.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0, gamma: float = 1.0):
        """
        Constructor for the distance metric loss function.

        This metric function computes a loss for the distance metric learning module.
        It will enforce the fundamental criteria of a distance metric e.g.,

        symmetry: d(a, b) == d(b, a)

        and the triangle identity:
            for c = a + b:
            d(a, b) <= d(a, c) + d(c, b)

        The loss function is composed of three parts:

            L = alpha * loss_simple + beta * loss_symmetry + gamma * loss_triangle

        Simple Loss
        ^^^^^^^^^^^
        L_simple = abs(y_predicted - y_true)

        Symmetry Loss
        ^^^^^^^^^^^^^
        L_symmetry = abs(d(a, b) - d(b, a))

        Triangle Loss
        ^^^^^^^^^^^^^
        L_triangle = H(y_pred - z' + z'')
        where z' = d(a, c) and z'' = d(c, b)


        Parameters
        ----------
        alpha : float
                Weighting term for the simple loss
        beta : float
                Weighting term for the symmetry loss
        gamma : float
                Weighting term for the triangle loss.
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    @tf.function
    def simple_loss(self, y_true, y_pred):
        """
        Compute the simple loss.

        Parameters
        ----------
        y_true : tf.Tensor
                True metric value
        y_pred : tf.Tensor
                Predicted metric value.

        Returns
        -------

        """

        return abs(y_true - y_pred)

    @tf.function
    def symmetry_loss(self, y_pred, y_permuted):
        """
        Compute the symmetry loss term.

        Parameters
        ----------
        y_pred : tf.Tensor
                Predicted standard metric.
        y_permuted : tf.Tensor
                Predicted metric for permuted input.

        Returns
        -------

        """
        return abs(y_pred - y_permuted)

    @tf.function
    def triangle_loss(self, y_pred, z_prime, z_prime_prime):
        """
        Compute the triangle loss on the prediction.

        Parameters
        ----------
        y_pred : tf.Tensor
                Predicted metric value
        z_prime : tf.Tensor
                Predicted z_prime value
        z_prime_prime : tf.Tensor
                Predicted z_prime_prime value.

        Returns
        -------

        """
        triangle = y_pred - (z_prime + z_prime_prime)
        return tf.experimental.numpy.heaviside(triangle, 1)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        """
        Call method during which the loss is computed.

        Parameters
        ----------
        y_true : tf.Tensor
                true y value.
        y_pred : tf.Tensor
                predicted y values for different combinations of inputs computed by the
                custom training step.
                looks like [y_pred, y_pred_permuted, z_prime, z_prime_prime]

        Returns
        -------

        """
        simple_loss = self.alpha * self.simple_loss(y_true, y_pred[0])
        symmetry_loss = self.beta * self.symmetry_loss(y_pred[0], y_pred[1])
        triangle_loss = self.gamma * self.triangle_loss(y_pred[0], y_pred[2], y_pred[3])

        return simple_loss + symmetry_loss + triangle_loss
