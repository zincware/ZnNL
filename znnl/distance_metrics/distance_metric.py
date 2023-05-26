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

from znnl.observables.observable import Observable


class DistanceMetric(Observable):
    """
    Parent class for a ZnRND distance metric.
    """

    def __name__(self) -> str:
        """
        Name of the class.

        Returns
        -------
        name : str
                The name of the class.
        """
        return "distance_metric"

    def __signature__(self) -> tuple:
        """
        Signature of the class.

        Returns
        -------
        signature : tuple
                The signature of the class.
                For the distance metric, it is (1,).
        """
        return (1,)

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
        raise NotImplementedError("Implemented in child class.")
