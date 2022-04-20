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
from znrnd.core.loss_functions.absolute_angle_difference import AngleDistanceLoss
from znrnd.core.loss_functions.cosine_distance import CosineDistanceLoss
from znrnd.core.loss_functions.l_p_norm import LPNormLoss
from znrnd.core.loss_functions.mahalanobis import MahalanobisLoss
from znrnd.core.loss_functions.mean_power_error import MeanPowerLoss
from znrnd.core.loss_functions.simple_loss import SimpleLoss

__all__ = [
    AngleDistanceLoss.__name__,
    CosineDistanceLoss.__name__,
    LPNormLoss.__name__,
    MahalanobisLoss.__name__,
    MeanPowerLoss.__name__,
    SimpleLoss.__name__,
]
