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
from jax.nn import softmax
from scipy.stats import wasserstein_distance

from znnl.distance_metrics.distance_metric import DistanceMetric


class WassersteinDistance(DistanceMetric):
    """
    Compute the 1-Wasserstein Distance between two probabilistic distributions.

    The Wasserstein Distance defines the minimal Cost re-distributing one distribution
    into another one. It is also called Earth-Mover Distance.
    The softmax function is applied to the input points to ensure that they are elements
    of a probability space.

    The Wasserstein Distance is based on a scipy function, which means it cannot be used
    for gradient and other auto-diff calculations.
    """

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
        d(point_1, point_2) : np.ndarray : shape=(n_points, 1)
                Array of distances for each point.
        """
        out_list = list(map(wasserstein_distance, softmax(point_1), softmax(point_2)))
        return np.array(out_list)
