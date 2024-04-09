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

from abc import ABC
from typing import Optional

import jax.numpy as np

from znnl.distance_metrics.distance_metric import DistanceMetric


class SimpleLoss(ABC):
    """
    Class for the simple loss.

    Attributes
    ----------
    metric : DistanceMetric
    """

    def __init__(self):
        """
        Constructor for the simple loss parent class.
        """
        super().__init__()
        self.metric: DistanceMetric = None

    def __call__(
        self, point_1: np.ndarray, point_2: np.ndarray, mask: Optional[np.array] = None
    ) -> float:
        """
        Summation over the tensor of the respective similarity measurement.

        Parameters
        ----------
        point_1 : np.ndarray
                Neural network representations of the considered points.
                Represented should have the shape (batch_size, n_features)
        point_2 : np.ndarray
                Neural network representations of the considered points.
                Represented should have the shape (batch_size, n_features)
        mask : np.ndarray, optional
                Mask to be multiplied to the loss.
                It needs to have the shape (batch_size, )

        Returns
        -------
        loss : float
                total loss of all points based on the similarity measurement
        """
        if mask is not None:
            return np.sum(self.metric(point_1, point_2) * mask, axis=0)
        else:
            return np.sum(self.metric(point_1, point_2), axis=0)
