"""
ZnNL: A Zincwarecode package.

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
"""

import jax.numpy as np

from znnl.distance_metrics.distance_metric import DistanceMetric


class ExponentialRepulsion(DistanceMetric):
    """
    Class for the exponential repulsion.

    The exponential repulsion inherits from the DistanceMetric class but
    is not a metric in the mathematical sense.
    It can be seen as a repulsive measure of how close points are to each other and
    therefore rather defines a potential function than a distance metric.

    It is mainly used for the calculation of the contraction loss.

    The default values for alpha and temperature are chosen such that the repulsion is
    of equal strength as the default attraction of the contraction loss.
    """

    def __init__(self, alpha: float = 0.01, temperature: float = 0.1):
        """
        Constructor for the exponential repulsion loss class.

        Parameters
        ----------
        alpha : float (default=0.01)
                Factor defining the strength of the repulsion, i.e. the value of the
                repulsion for zero distance.
        temperature : float (default=0.1)
                Factor defining the length scale on which the repulsion is taking place.
                This can be interpreted as a temperature parameter softening the
                repulsion.

        """
        self.alpha = alpha
        self.temperature = temperature

    def __call__(self, point_1: np.ndarray, point_2: np.ndarray, **kwargs):
        """
        Call the distance metric.

        Distance between points in the point_1 tensor will be computed between those in
        the point_2 tensor element-wise. Therefore, we will have:

                point_1[i] - point_2[i] for all i.

        Parameters
        ----------
        point_1 : np.ndarray (n_points, point_dimension)
            First set of points in the comparison.
        point_2 : np.ndarray (n_points, point_dimension)
            Second set of points in the comparison.
        kwargs
            Miscellaneous keyword arguments for the specific metric.

        Returns
        -------
        d(point_1, point_2) : tf.tensor : shape=(n_points, 1)
                Array of distances for each point.
        """
        absolute = np.abs(point_1 - point_2)
        return np.mean(self.alpha * np.exp(-absolute / self.temperature), axis=1)
