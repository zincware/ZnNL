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
from znnl.distance_metrics.wasserstein_distance import WassersteinDistance
from znnl.loss_functions.simple_loss import SimpleLoss


class WassersteinLoss(SimpleLoss):
    """
    Class for the Wasserstein loss.

    For further information about the used metric see the documentation of the
    WassersteinDistance class.
    """

    def __init__(self):
        """
        Constructor for the Wasserstein loss class.
        """
        super(WassersteinLoss, self).__init__()
        self.metric = WassersteinDistance()
