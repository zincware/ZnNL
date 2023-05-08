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
from znnl.distance_metrics.exponential_repulsion import ExponentialRepulsion
from znnl.loss_functions.simple_loss import SimpleLoss


class ExponentialRepulsionLoss(SimpleLoss):
    """
    Class for the exponential repulsion loss.

    The exponential repulsion loss is primarily used for the calculation of the
    contraction loss as it returns a repulsive measure of the distance between
    two points.
    """

    def __init__(self, alpha: float = 0.01, temp: float = 0.1):
        """
        Constructor for the exponential repulsion loss class.

        Parameters
        ----------
        alpha : float (default=0.01)
                Factor defining the strength of the repulsion, i.e. the value of the
                repulsion for zero distance.
        temp : float (default=0.1)
                Factor defining the length scale on which the repulsion is taking place.
                This can be interpreted as a temperature parameter softening the
                repulsion.
        """
        super(ExponentialRepulsionLoss, self).__init__()
        self.metric = ExponentialRepulsion(alpha=alpha, temp=temp)
