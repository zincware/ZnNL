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
Raise a difference to a power of order n.

e.g. (a - b)^n
"""
import tensorflow as tf

from .distance_metric import DistanceMetric


class OrderNDifference(DistanceMetric):
    """
    Compute the order n difference between points.
    """

    def __init__(self, order: float = 2):
        """
        Constructor for the order n distance.

        Parameters
        ----------
        order : float (default=2)
                Order to which the difference should be raised.
        """
        self.order = order

    def __call__(self, point_1: tf.Tensor, point_2: tf.Tensor, **kwargs):
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
        diff = point_1 - point_2
        # return tf.cast(tf.einsum("ij, ij -> i", diff, diff), tf.float32)
        return tf.cast(tf.reduce_sum(tf.pow(diff, self.order), axis=1), tf.float32)
