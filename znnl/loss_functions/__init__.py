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
from znnl.loss_functions.absolute_angle_difference import AngleDistanceLoss
from znnl.loss_functions.contrastive_loss import ContrastiveLoss, ExternalPotential
from znnl.loss_functions.cosine_distance import CosineDistanceLoss
from znnl.loss_functions.cross_entropy_loss import CrossEntropyLoss
from znnl.loss_functions.exponential_repulsion_loss import ExponentialRepulsionLoss
from znnl.loss_functions.l_p_norm import LPNormLoss
from znnl.loss_functions.mahalanobis import MahalanobisLoss
from znnl.loss_functions.mean_power_error import MeanPowerLoss
from znnl.loss_functions.simple_loss import SimpleLoss
from znnl.loss_functions.wasserstein_loss import WassersteinLoss

__all__ = [
    AngleDistanceLoss.__name__,
    CosineDistanceLoss.__name__,
    LPNormLoss.__name__,
    MahalanobisLoss.__name__,
    MeanPowerLoss.__name__,
    SimpleLoss.__name__,
    CrossEntropyLoss.__name__,
    WassersteinLoss.__name__,
    ExponentialRepulsionLoss.__name__,
    ContrastiveLoss.__name__,
    ExternalPotential.__name__,
]
