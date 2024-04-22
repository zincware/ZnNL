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

import logging

logger = logging.getLogger(__name__)

from znnl.distance_metrics.wasserstein_distance import WassersteinDistance
from znnl.loss_functions.simple_loss import SimpleLoss


class WassersteinLoss(SimpleLoss):
    """
    Class for the Wasserstein loss.

    The Wasserstein loss is defined via the Wasserstein distance between two
    distributions. The implementation is based on the WassersteinDistance class,
    which cannot be used for gradient and other auto-diff calculations!

    For further information about the used metric see the documentation of the
    WassersteinDistance class.
    """

    def __init__(self):
        """
        Constructor for the Wasserstein loss class.
        """
        logger.warning("The Wasserstein loss cannot be used for gradient calculations.")

        super(WassersteinLoss, self).__init__()
        self.metric = WassersteinDistance()
