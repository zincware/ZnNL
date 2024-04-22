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

import jax.numpy as np

from znnl.loss_functions.contrastive_loss import ContrastiveLoss


class ContrastiveInfoNCELoss(ContrastiveLoss):
    """
    Class for the Contrastive InfoNCE Loss.

    The InfoNCE loss is a type of contrastive loss function based on information
`    theory and Noise-Contrastive Estimation (NCE). It is used to train neural networks
    by maximizing the mutual information between representations of data samples.

    The InfoNCE loss is an implementation of a contrastive loss function using a
    repulsive and attractive potential in one term.
    The loss is scaled by a temperature parameter.
    """

    def __init__(
        self,
        temperature: float = 1.0,
    ):
        """
        Constructor for the Contrastive Loss class.

        Parameters
        ----------
        temperature : float
                The temperature parameter for the InfoNCE loss.
        """

        super().__init__()
        self.temperature = temperature

    def compute_losses(self, inputs: np.ndarray, targets: np.ndarray):
        """
        Compute the contrastive InfoNCE loss.

        The InfoNCE loss is a type of contrastive loss function.
        It is used to train neural networks by maximizing or minimized the mutual
        information between the representations of samples.

        Parameters
        ----------
        inputs : np.ndarray
                Input to contrastive loss.
        targets : np.ndarray
                Targets of the dataset.

        Returns
        -------
        losses : float
                The InfoNCE loss.
        """
        # Create the map of which pairs are similar and which are different
        pos_mask, neg_mask = self.create_label_map(targets=targets)

        # Compute the exponential dot product
        exp_dot_product, dot_product = self.compute_exponential_dot_product(
            inputs, inputs
        )

        # Compute the sum of negative samples
        sum_neg_samples = np.sum(exp_dot_product * neg_mask, axis=1, keepdims=True)

        # Add the sum of negative samples to the exponential dot product of positive
        # samples to get the denominator of each positive sample
        denominator = sum_neg_samples + exp_dot_product

        # Log the denominator
        log_denominator = np.log(denominator) * pos_mask

        # Calculate the log of each positive sample
        log_prob = (-dot_product + log_denominator) * pos_mask

        # Average all positive samples (take care of not dividing by zero)
        norm = np.sum(pos_mask, axis=1)
        log_prob = np.sum(log_prob, axis=1) * np.where(norm == 0, 0, 1 / norm)
        # log_prob = np.nan_to_num(log_prob, nan=0.0)

        # Average over all samples
        loss = np.mean(log_prob)

        return loss

    def compute_exponential_dot_product(
        self, inputs_1: np.ndarray, inputs_2: np.ndarray
    ):
        """
        Compute the exponential dot product.

        Compute the exponential dot product of the inputs.

        Parameters
        ----------
        inputs : np.ndarray
                The input data of shape (k, n_features), where
                k = (batch_size^2 - batch_size)/2

        Returns
        -------
        dot_product : np.ndarray
                The exponential dot product of shape
        """
        # Compute the dot product of the inputs
        dot_product: np.ndarray = (
            np.einsum("ij, kj -> ik", inputs_1, inputs_2) / self.temperature
        )

        # For stability taken from
        # https://github.com/tufts-ml/SupContrast/blob/master/losses.py
        _max = np.max(dot_product, axis=1, keepdims=True)
        dot_product = dot_product - _max

        # Compute the exponential of the dot product using the temperature
        exp_dot_product: np.ndarray = np.exp(dot_product)

        return exp_dot_product, dot_product
