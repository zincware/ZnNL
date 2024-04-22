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
from typing import Tuple, Union

import jax.numpy as np


class ContrastiveLoss(ABC):
    """
    Abstract base class for the Contrastive Loss.

    The contrastive loss is a loss function, incorporating contrastive interactions
    between representations of data samples. It is used to learn similarity between
    data samples.
    It is fully differentiable and can be used to train neural networks.
    """

    def __init__(self):
        """
        Constructor for the Contrastive Loss class.
        """
        super().__init__()

    @staticmethod
    def create_label_map_symmetric(
        targets: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Compute the indices of pairs of similar and different labels assuming symmetry
        of the interactions.

        Find pairs of similar label and pairs of different labels in the one-hot encoded
        targets of a dataset.

        Parameters
        ----------
        targets : np.ndarray
                The one-hot encoded targets of the dataset.

        Returns
        -------
        label_map: Tuple[mask_sim, mask_diff, map_idx]

            mask_sim : np.array
                    Mask for data points with similar labels.
                    The mask is a binary array with 1 for similar labels and 0 for different
                    labels.
                    It has the shape (n_pairs, )
            mask_diff : np.array
                    Mask for data points with different labels.
                    The mask is a binary array with 1 for different labels and 0 for similar
                    labels.
                    It has the shape (n_pairs, )
            map_idx : Tuple[np.array, np.array]
                    Tuple of arrays containing the indices of the pairs of similar and
                    different labels.
                    It has the shape (2, n_pairs)
        """
        # Gram matrix of vectors
        matrix: np.ndarray = np.einsum("ij, kj -> ik", targets, targets)

        # Get the indices of the upper triangle of the gram matrix
        idx_map: Tuple[np.array, np.array] = np.triu_indices(len(matrix), 1)

        # Create masks for similar and different pairs using the triangle indices
        sim_mask: np.array = np.array(matrix[idx_map], dtype=int)
        diff_mask: np.array = np.array((sim_mask - 1) * -1, dtype=int)

        return sim_mask, diff_mask, idx_map

    @staticmethod
    def create_label_map(targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the indices of pairs of similar and different labels for interactions
        without symmetry.

        Find pairs of similar label and pairs of different labels in the one-hot encoded
        targets of a dataset.

        Parameters
        ----------
        targets : np.ndarray
                The one-hot encoded targets of the dataset.

        Returns
        -------
        label_map: Tuple[pos_mask, neg_mask]

            pos_mask : np.ndarray
                    Mask for data points with similar labels.
                    The mask is a binary array with 1 for similar labels and 0 for different
                    labels.
                    It has the shape (n, n) with n being the number of data points.
            neg_mask : np.ndarray
                    Mask for data points with different labels.
                    The mask is a binary array with 1 for different labels and 0 for similar
                    labels.
                    It has the shape (n, n) with n being the number of data points.
        """
        # Gram matrix of vectors
        pos_mask: np.ndarray = np.einsum("ij, kj -> ik", targets, targets)

        # Create negative mask
        neg_mask = 1 - pos_mask

        # Remove the diagonal of the gram matrix
        pos_mask = pos_mask - np.eye(len(pos_mask))

        return pos_mask, neg_mask

    def compute_losses(
        self, inputs: np.ndarray, targets: np.ndarray
    ) -> Union[float, Tuple[float]]:
        """
        Compute the contrastive losses.

        This method returns all losses computed.
        Depending on the implementation, it can return multiple losses.

        Parameters
        ----------
        inputs : np.ndarray
                Input to contrastive loss.
        targets : np.ndarray
                Targets of the dataset.

        Returns
        -------
        losses : Tuple[float]
                Tuple of all losses computed.
        """
        raise NotImplementedError("Implemented in child classes.")

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

        losses = self.compute_losses(inputs, targets)

        if isinstance(losses, tuple):
            return np.array([losses]).sum()
        else:
            return losses
