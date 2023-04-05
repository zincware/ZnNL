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
ZnRND Cosine similarity TF loss function.
"""
from znrnd.distance_metrics.cosine_distance import CosineDistance
from znrnd.loss_functions.simple_loss import SimpleLoss


class CosineDistanceLoss(SimpleLoss):
    """
    Class for the mean power loss
    """

    def __init__(self):
        """
        Constructor for the mean power loss class.
        """
        super(CosineDistanceLoss, self).__init__()
        self.metric = CosineDistance()

    def __name__(self):
        """
        Name of the class.
        """
        return "CosineDistanceLoss"
