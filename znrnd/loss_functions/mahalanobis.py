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
ZnRND Mahalanobis distance TF loss function.
"""
import znrnd.distance_metrics.mahalanobis_distance as mahalanobis
from znrnd.loss_functions.simple_loss import SimpleLoss


class MahalanobisLoss(SimpleLoss):
    """
    Class for the mean power loss
    """

    def __init__(self):
        """
        Constructor for the Mahalanobis loss class.
        """
        super(MahalanobisLoss, self).__init__()
        self.metric = mahalanobis.MahalanobisDistance()

    def __name__(self) -> str:
        """
        The name of the analsyis module.

        This name should be as detailed as possible.

        Returns
        -------
        name : str
                A string representation of the name.
        """
        return "MahalanobisLoss"
