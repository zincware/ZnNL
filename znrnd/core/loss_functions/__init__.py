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
Package containing custom loss functions.
"""
from .absolute_angle_difference import AngleDistanceLoss
from .cosine_distance import CosineDistanceLoss
from .distance_metric_loss import DistanceMetricLoss
from .l_p_norm import LPNormLoss
from .mahalanobis import MahalanobisLoss
from .mean_power_error import MeanPowerLoss
from .simple_loss import SimpleLoss

__all__ = [
    "AngleDistanceLoss",
    "CosineDistanceLoss",
    "DistanceMetricLoss",
    "LPNormLoss",
    "MahalanobisLoss",
    "MeanPowerLoss",
    "SimpleLoss",
]
