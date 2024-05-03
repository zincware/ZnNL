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

from znnl.loss_functions import ContrastiveInfoNCELoss


class TestContrastiveInfoNCELoss:
    """
    Class for the testing of the contrastive InfoNCE loss.
    """

    @classmethod
    def setup_class(cls):
        """
        Prepare the test class
        """
        cls.predictions = np.array([[1, 0, 1], [1, 1, 0], [0, 0, 1]])
        cls.targets = np.array([[1, 0], [1, 0], [0, 1]])  # one-hot

    def test_compute_losses(self):
        """
        Test the contrastive loss call method
        """
        contrastive_loss = ContrastiveInfoNCELoss()

        # Test particular case
        loss = contrastive_loss(self.predictions, self.targets)
        expected_loss = (
            1
            / 3
            * (-1 + np.log(np.exp(1) + np.exp(1)) - 1 + np.log(np.exp(1) + np.exp(0)))
        )
        onp.testing.assert_almost_equal(loss, expected_loss)

        # Test case of no positive pairs
        targets = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
        loss = contrastive_loss(self.predictions, targets)
        expected_loss = 0.0
        onp.testing.assert_almost_equal(loss, expected_loss)

        # Test case of no negative pairs
        targets = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]])
        loss = contrastive_loss(self.predictions, targets)
        expected_loss = 0.0
        onp.testing.assert_almost_equal(loss, expected_loss)

    def test_temperature(self):
        """
        Test the contrastive loss call method with temperature
        """
        contrastive_loss = ContrastiveInfoNCELoss(temperature=2.0)

        # Test particular case
        loss = contrastive_loss(self.predictions, self.targets)
        expected_loss = (
            1
            / 3
            * (
                -1 / 2
                + np.log(np.exp(1 / 2) + np.exp(1 / 2))
                - 1 / 2
                + np.log(np.exp(1 / 2) + np.exp(0))
            )
        )
        onp.testing.assert_almost_equal(loss, expected_loss)
