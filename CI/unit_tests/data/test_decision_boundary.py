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
Unit test for the decision boundary.
"""

import jax.numpy as np
import numpy as onp
from pytest import approx

from znnl.data.decision_boundary import (
    DecisionBoundaryGenerator,
    circle,
    linear_boundary,
)


class TestDecisionBoundary:
    """
    Unit test for the decision boundary.
    """

    def test_linear_boundary(self):
        """
        Test the linear boundary.
        """
        target_ratio = 0.0
        for _ in range(10):
            input_data = onp.random.uniform(0, 1, size=(10000, 2))
            target_ratio += linear_boundary(input_data, 1.0, 0.0).mean()

        assert target_ratio / 10 == approx(0.5, rel=0.01)

    def test_circle(self):
        """
        Test the circle boundary.
        """
        target_ratio = 0.0
        for _ in range(10):
            input_data = onp.random.uniform(0, 1, size=(10000, 2))
            target_ratio += circle(input_data, 0.25).mean()

        # P(x in class 1) = 1 - (pi / 16)
        assert target_ratio / 10 == approx(1 - (np.pi / 16), abs=0.01)

    def test_one_hot_decision_boundary_generator(self):
        """
        Test the actual generator.
        """
        generator = DecisionBoundaryGenerator(
            n_samples=10000, discriminator="circle", one_hot=True
        )

        # Check the dataset shapes
        assert generator.train_ds["inputs"].shape == (10000, 2)
        assert generator.train_ds["targets"].shape == (10000, 2)
        assert generator.test_ds["inputs"].shape == (10000, 2)
        assert generator.test_ds["targets"].shape == (10000, 2)

    def test_serial_decision_boundary_generator(self):
        """
        Test the actual generator.
        """
        generator = DecisionBoundaryGenerator(
            n_samples=10000, discriminator="circle", one_hot=False
        )

        # Check the dataset shapes
        assert generator.train_ds["inputs"].shape == (10000, 2)
        assert generator.train_ds["targets"].shape == (10000, 1)
        assert generator.test_ds["inputs"].shape == (10000, 2)
        assert generator.test_ds["targets"].shape == (10000, 1)
