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
from abc import ABC
from typing import Callable, Optional, Tuple

import jax.numpy as np

from znnl.distance_metrics.distance_metric import DistanceMetric
from znnl.distance_metrics.order_n_difference import OrderNDifference
from znnl.loss_functions.exponential_repulsion_loss import ExponentialRepulsionLoss
from znnl.loss_functions.mean_power_error import MeanPowerLoss
from znnl.loss_functions.simple_loss import SimpleLoss


class ContrastiveLoss:
    """
    Class for the Contrastive Loss.

    The contrastive loss is a loss function, which is not based on a distance metric.

    It is fully differentiable and can be used to train neural networks.
    The contrastive coss is defined through three loss/potential functions:
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
        Constructor for the Contrastive Loss class.

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

        self.attractive_pot_fn = attractive_pot_fn
        self.repulsive_pot_fn = repulsive_pot_fn
        self.external_pot_fn = external_pot_fn

        self.turn_off_potentials = np.array(
            [
                turn_off_attractive_potential,
                turn_off_repulsive_potential,
                turn_off_external_potential,
            ]
        )

        self._set_default_potentials()

    def _set_default_potentials(self):
        """
        Set the default potentials for the Contrastive Loss.

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

    @staticmethod
    def create_label_map(targets: np.ndarray):
        """
        Compute the indices of pairs of similar and different labels.

        Find pairs of similar label and pairs of different labels in the one-hot encoded
        targets of a dataset.

        Parameters
        ----------
        targets : np.ndarray
                The one-hot encoded targets of the dataset.

        Returns
        -------
        pair_sim : Tuple[np.array, np.array]
                Tuple of arrays containing the indices of the pairs of similar labels.
        pair_diff : Tuple[np.array, np.array]
                Tuple of arrays containing the indices of the pairs of different labels.
        """
        # Gram matrix of vectors
        matrix: np.ndarray = np.einsum("ij, kj -> ik", targets, targets)

        # Get the indices of the upper triangle of the gram matrix
        triangle_idx: Tuple[np.array, np.array] = np.triu_indices(len(matrix), 1)

        # Create masks for similar and different pairs using the triangle indices
        mask_sim: np.array = np.bool_(matrix[triangle_idx])
        mask_diff: np.array = np.logical_not(mask_sim)

        # Compute the indices of pairs of similar and different labels
        pair_sim = (triangle_idx[0][mask_sim], triangle_idx[1][mask_sim])
        pair_diff = (triangle_idx[0][mask_diff], triangle_idx[1][mask_diff])

        return pair_sim, pair_diff

    def compute_losses(self, inputs: np.ndarray, targets: np.ndarray):
        """
        Compute the contrastive losses.

        This method returns the attractive, repulsive and external losses separately.

        Parameters
        ----------
        inputs : np.ndarray
                Input to contrastive loss.
        targets : np.ndarray
                Targets of the dataset.

        Returns
        -------
        losses : Tuple[float, float, float]
                Tuple of the attractive, repulsive and external losses.
        """

        att_map, rep_map = self.create_label_map(targets=targets)

        attractive_loss = self.attractive_pot_fn(
            np.take(inputs, att_map[0], axis=0),
            np.take(inputs, att_map[1], axis=0),
        )
        repulsive_loss = self.repulsive_pot_fn(
            np.take(inputs, rep_map[0], axis=0),
            np.take(inputs, rep_map[1], axis=0),
        )
        external_loss = self.external_pot_fn(inputs)

        return attractive_loss, repulsive_loss, external_loss

    def __call__(self, inputs: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute the contrastive loss.

        Parameters
        ----------
        inputs : np.ndarray
                Input to contrastive loss.
        targets : np.ndarray
                Targets of the dataset.

        Returns
        -------
        loss : float
                total loss of all points based on the similarity measurement
        """

        attractive_loss, repulsive_loss, external_loss = self.compute_losses(
            inputs, targets
        )

        return_list = np.array([attractive_loss, repulsive_loss, external_loss])
        take_idx = np.where(self.turn_off_potentials == False)[0]
        return np.sum(np.take(return_list, take_idx))


class ExternalPotential(ABC):
    """
    Class for the external potential function.

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
