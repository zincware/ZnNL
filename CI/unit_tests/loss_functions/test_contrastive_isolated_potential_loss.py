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

from znnl.loss_functions import ContrastiveIsolatedPotentialLoss


class TestContrastiveIsolatedPotentialLoss:
    """
    Class for the testing of the contrastive isolated potential loss.
    """

    @classmethod
    def setup_class(cls):
        """
        Prepare the test class
        """
        cls.predictions = np.array([[1, 1, 2], [1, 1, 1], [0, 0, 0], [2, 1, 1]])
        cls.targets = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]])  # one-hot

    def test_contrastive_loss(self):
        """
        Test the contrastive loss call method
        """
        contrastive_loss = ContrastiveIsolatedPotentialLoss()
        loss = contrastive_loss(self.predictions, self.targets)
        onp.testing.assert_almost_equal(loss, 3.36333, decimal=4)
