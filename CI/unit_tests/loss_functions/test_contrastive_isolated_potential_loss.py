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
import numpy as onp

from znnl.distance_metrics import LPNorm
from znnl.loss_functions import (
    ContrastiveIsolatedPotentialLoss,
    ExponentialRepulsionLoss,
    ExternalPotential,
)


class TestContrastiveIsolatedPotentialLoss:
    """
    Class for the testing of the contrastive isolated potential loss.
    """

    def test_contrastive_loss(self):
        """
        Test the contrastive loss call method
        """
        # General case
        predictions = np.array([[1, 1, 2], [1, 1, 1], [0, 0, 0], [2, 1, 1]])
        targets = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]])  # one-hot
        contrastive_loss = ContrastiveIsolatedPotentialLoss()
        loss = contrastive_loss(predictions, targets)
        onp.testing.assert_almost_equal(loss, 3.36333, decimal=4)

        # Only attractive potential
        predictions = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]])
        targets = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]])  # one-hot
        contrastive_loss = ContrastiveIsolatedPotentialLoss(
            turn_off_external_potential=True
        )
        loss = contrastive_loss(predictions, targets)
        onp.testing.assert_almost_equal(loss, 0.0, decimal=4)

        # Only repulsive potential
        predictions = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        targets = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # one-hot
        contrastive_loss = ContrastiveIsolatedPotentialLoss(
            turn_off_external_potential=True,
            repulsive_pot_fn=ExponentialRepulsionLoss(alpha=1, temp=1),
        )
        loss = contrastive_loss(predictions, targets)
        onp.testing.assert_almost_equal(loss, 1.0, decimal=4)

        # Only external potential
        predictions = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        targets = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        contrastive_loss = ContrastiveIsolatedPotentialLoss(
            turn_off_attractive_potential=True,
            turn_off_repulsive_potential=True,
            external_pot_fn=ExternalPotential(distance_metric=LPNorm(order=2)),
        )
        loss = contrastive_loss(predictions, targets)
        onp.testing.assert_almost_equal(loss, np.sqrt(3), decimal=4)
