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
    Class for the testing of the contrastive loss function.
    """

    @classmethod
    def setup_class(cls):
        """
        Prepare the test class
        """
        cls.targets = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]])  # one-hot

    def test_create_label_map_symmetric(self):
        """
        Test the creation of the label map method with symmetric interactions.
        """
        contrastive_loss = ContrastiveLoss()
        sim_mask, diff_mask, idx_map = contrastive_loss.create_label_map_symmetric(
            self.targets
        )

        # Assert shapes
        assert sim_mask.shape == (6,)
        assert diff_mask.shape == (6,)
        assert idx_map[0].shape == (6,)
        assert idx_map[1].shape == (6,)

        # Assert values
        onp.testing.assert_array_equal(sim_mask, np.array([0, 0, 1, 0, 0, 0]))
        onp.testing.assert_array_equal(diff_mask, np.array([1, 1, 0, 1, 1, 1]))
        onp.testing.assert_array_equal(idx_map[0], np.array([0, 0, 0, 1, 1, 2]))
        onp.testing.assert_array_equal(idx_map[1], np.array([1, 2, 3, 2, 3, 3]))

    def test_create_label_map(self):
        """
        Test the creation of the label map method.
        """
        contrastive_loss = ContrastiveLoss()
        pos_mask, neg_mask = contrastive_loss.create_label_map(self.targets)

        # Assert shapes
        assert pos_mask.shape == (4, 4)
        assert neg_mask.shape == (4, 4)

        # Assert values
        onp.testing.assert_array_equal(
            pos_mask, np.array([[0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]])
        )
        onp.testing.assert_array_equal(
            neg_mask, np.array([[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 1], [0, 1, 1, 0]])
        )
