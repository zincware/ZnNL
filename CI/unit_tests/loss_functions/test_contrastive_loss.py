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

from znnl.loss_functions import ContrastiveLoss


class TestContrastiveLoss:
    """
    Class for the testing of the contrastive loss functions.
    """

    @classmethod
    def setup_class(cls):
        """
        Prepare the test class
        """
        cls.predictions = np.array([[1, 1, 2], [1, 1, 1], [0, 0, 0], [2, 1, 1]])
        cls.targets = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]])  # one-hot

    def test_create_label_map(self):
        """
        Test the creation of the label map method.
        """
        contrastive_loss = ContrastiveLoss()
        mask_sim, mask_diff, map_idx = contrastive_loss.create_label_map(self.targets)

        # Assert shapes
        assert mask_sim.shape == (6,)
        assert mask_diff.shape == (6,)
        assert map_idx[0].shape == (6,)
        assert map_idx[1].shape == (6,)

        # Assert values
        print(map_idx)
        onp.testing.assert_array_equal(mask_sim, np.array([0, 0, 1, 0, 0, 0]))
        onp.testing.assert_array_equal(mask_diff, np.array([1, 1, 0, 1, 1, 1]))
        onp.testing.assert_array_equal(map_idx[0], np.array([0, 0, 0, 1, 1, 2]))
        onp.testing.assert_array_equal(map_idx[1], np.array([1, 2, 3, 2, 3, 3]))

    def test_contrastive_loss(self):
        """
        Test the contrastive loss call method
        """
        contrastive_loss = ContrastiveLoss()
        loss = contrastive_loss(self.predictions, self.targets)
        onp.testing.assert_almost_equal(loss, 3.9300, decimal=4)
