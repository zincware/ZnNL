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

from typing import Callable, Optional, Tuple

import jax.numpy as np

from znnl.distance_metrics.distance_metric import DistanceMetric
from znnl.distance_metrics.order_n_difference import OrderNDifference
from znnl.loss_functions.contrastive_loss import ContrastiveLoss
from znnl.loss_functions.exponential_repulsion_loss import ExponentialRepulsionLoss
from znnl.loss_functions.mean_power_error import MeanPowerLoss
from znnl.loss_functions.simple_loss import SimpleLoss


class ContrastiveIsolatedPotentialLoss(ContrastiveLoss):
    """
    Class for the Contrastive Isolated Potential Loss (CIP Loss)

    This contrastive loss is a loss function, incorporating contrastive interactions
    between representations of data samples. Compared to other loss functions
    implemented, the contrastive coss is not based on a single znnl.distance_metric.

    It is fully differentiable and can be used to train neural networks.
    The CIP loss is defined through three loss/potential functions:
    - Attractive potential
    - Repulsive potential
    - External potential
    For each of these potentials, users can define their own or use
    the default potential functions.
    The potential functions are combined into a total loss function by summation.
    Note that the potential values should be of similar magnitude otherwise one
    potential can dominate the others.

    The default potential functions are:
    - Attractive potential: L_p norm (p=2)
    - Repulsive potential: Exponential repulsion Loss(default parameters)
    - External potential: ExternalPotential (default parameters)

    The default loss functions can be turned off by setting the corresponding
    boolean to True. All loss functions are always evaluated, but the loss is only added
    to the total loss if the corresponding boolean is False.
    """

    def __init__(
        self,
        attractive_pot_fn: Optional[SimpleLoss] = None,
        repulsive_pot_fn: Optional[SimpleLoss] = None,
        external_pot_fn: Optional[Callable] = None,
        turn_off_attractive_potential: bool = False,
        turn_off_repulsive_potential: bool = False,
        turn_off_external_potential: bool = False,
    ):
        """
        Constructor for the CIP Loss class.

        Parameters
        ----------
        attractive_pot_fn : SimpleLoss
                The attractive potential function.
        repulsive_pot_fn : SimpleLoss
                The repulsive potential function.
        external_pot_fn : Callable
                The external potential function.
        turn_off_attractive_potential : bool (default: False)
                Turn off the attractive potential.
        turn_off_repulsive_potential : bool (default: False)
                Turn off the repulsive potential.
        turn_off_external_potential : bool (default: False)
                Turn off the external potential.
        """
        super().__init__()

        self.attractive_pot_fn = attractive_pot_fn
        self.repulsive_pot_fn = repulsive_pot_fn
        self.external_pot_fn = external_pot_fn

        self.turn_off_attractive_potential = turn_off_attractive_potential
        self.turn_off_repulsive_potential = turn_off_repulsive_potential
        self.turn_off_external_potential = turn_off_external_potential

        self._set_default_potentials()

    def _set_default_potentials(self):
        """
        Set the default potentials for the CIP Loss.

        The default potentials are:
        - Attractive potential: L_p norm(p=2)
        - Repulsive potential: Exponential repulsion
        - External potential: ExternalPotential
        """
        if self.attractive_pot_fn is None:
            self.attractive_pot_fn = MeanPowerLoss(order=2)
        if self.repulsive_pot_fn is None:
            self.repulsive_pot_fn = ExponentialRepulsionLoss()
        if self.external_pot_fn is None:
            self.external_pot_fn = ExternalPotential()

    def compute_losses(self, inputs: np.ndarray, targets: np.ndarray) -> Tuple[float]:
        """
        Compute the CIP losses.

        This method returns the attractive, repulsive and external losses separately.

        Parameters
        ----------
        inputs : np.ndarray
                Input to contrastive loss.
        targets : np.ndarray
                Targets of the dataset.

        Returns
        -------
        losses : Tuple[float]
                Tuple of the attractive, repulsive and external losses.
        """
        sim_mask, diff_mask, idx_map = self.create_label_map_symmetric(targets=targets)

        # Get all interacting pairs of data
        data_left = np.take(inputs, idx_map[0], axis=0)
        data_right = np.take(inputs, idx_map[1], axis=0)

        if self.turn_off_attractive_potential:
            attractive_loss = 0
        else:
            attractive_loss = self.attractive_pot_fn(data_left, data_right, sim_mask)

        if self.turn_off_repulsive_potential:
            repulsive_loss = 0
        else:
            repulsive_loss = self.repulsive_pot_fn(data_left, data_right, diff_mask)

        if self.turn_off_external_potential:
            external_loss = 0
        else:
            external_loss = self.external_pot_fn(inputs)

        return attractive_loss, repulsive_loss, external_loss


class ExternalPotential:
    """
    Class for an external potential function.

    An external potential function is a function that maps a place in space to a
    potential value.
    """

    def __init__(
        self,
        distance_metric: Optional[DistanceMetric] = None,
        scale: Optional[float] = 1.0,
        center: Optional[np.array] = None,
    ):
        """
        Constructor for the simple loss parent class.

        Parameters
        ----------
        distance_metric : Optional[DistanceMetric]
                The distance metric used to compute the external potential.
        scale : Optional[float]
                The factor scaling the width of the external potential relative to the
                input space.
                Values larger than 1.0 will result a wider potential.
                Values smaller than 1.0 will result in a narrower potential.
        center : Optional[np.array]
                The center of the external potential.
                Has to be of the same shape as points in the input space.
        """
        super().__init__()
        if distance_metric is None:
            distance_metric = OrderNDifference(order=4)
        self.metric = distance_metric
        self.scale = scale
        self.center = center

    def __call__(self, point_1: np.array) -> float:
        """
        Call the external potential function.

        First setting the center point which defines the minimum of the external
        potential. Then computing the external potential for the given points by using
        the distance metric.

        Parameters
        ----------
        point_1 : np.array
                The points for which the external potential is computed.

        Returns
        -------
        loss : float
                Total loss of all points based on the similarity measurement
        """
        if self.center:
            point_2 = np.repeat(self.center[None, ...], point_1.shape[0], axis=0)
        else:
            point_2 = np.zeros_like(point_1)

        return np.mean(self.metric(point_1 / self.scale, point_2 / self.scale), axis=0)
