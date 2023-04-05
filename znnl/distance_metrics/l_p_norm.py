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
Module for the L_p norm class.

    r = p1 - p2

    d = (|r[0]|^p + |r[1]|^p + ... + |r[n]|^p)^(1/p)
"""
import jax.numpy as np

from znnl.distance_metrics.distance_metric import DistanceMetric


class LPNorm(DistanceMetric):
    """
    Compute the L_p norm between vectors.
    """

    def __init__(self, order: float):
        """
        Constructor for the LPNorm class.

        Parameters
        ----------
        order : float
                order of the space
        """
        self.order = order

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
        return np.linalg.norm(point_1 - point_2, axis=1, ord=self.order)
