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
from znnl.distance_metrics.angular_distance import AngularDistance
from znnl.distance_metrics.cosine_distance import CosineDistance
from znnl.distance_metrics.distance_metric import DistanceMetric
from znnl.distance_metrics.hyper_sphere_distance import HyperSphere
from znnl.distance_metrics.l_p_norm import LPNorm
from znnl.distance_metrics.mahalanobis_distance import MahalanobisDistance
from znnl.distance_metrics.order_n_difference import OrderNDifference
from znnl.distance_metrics.cross_entropy_distance import CrossEntropyDistance

__all__ = [
    DistanceMetric.__name__,
    CosineDistance.__name__,
    AngularDistance.__name__,
    LPNorm.__name__,
    OrderNDifference.__name__,
    MahalanobisDistance.__name__,
    HyperSphere.__name__,
    CrossEntropyDistance.__name__
]
