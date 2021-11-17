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
distance metric module
"""
from .distance_metric import DistanceMetric
from .cosine_distance import CosineDistance
from .angular_distance import AngularDistance
from .l_p_norm import LPNorm
from .order_n_difference import OrderNDifference
from .mahalanobis_distance import MahalanobisDistance
from .mlp import MLPMetric

__all__ = [
    "DistanceMetric",
    "CosineDistance",
    "MLPMetric",
    "AngularDistance",
    "LPNorm",
    "OrderNDifference",
    "MahalanobisDistance"
]
