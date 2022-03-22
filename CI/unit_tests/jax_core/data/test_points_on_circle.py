"""
ZnRND: A Zincwarecode package.

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
Test the data generator module.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import jax.numpy as np
import pytest
from znrnd.jax_core.data.points_on_a_circle import PointsOnCircle


class TestPointsOnCircle:
    """
    Test the points on a circle module.
    """

    @classmethod
    def setup_class(cls):
        """
        Prepare the test suite.
        """
        cls.data_generator = PointsOnCircle()

    def test_no_noise(self):
        """
        Test that circle is generated with no noise.
        """
        self.data_generator.build_pool(100, noise=False)

        np.linalg.norm(self.data_generator.data_pool, axis=1).mean() == 1.0

    def test_noise(self):
        """
        Test circle generation with noise.
        """
        self.data_generator.build_pool(100, noise=True)

        # Assert close to 1.0
        np.linalg.norm(self.data_generator.data_pool, axis=1).mean() == pytest.approx(
            1.0, 0.00001
        )
        # Assert that it is not exactly 1.0
        np.linalg.norm(self.data_generator.data_pool, axis=1).mean() != 1.0
        # Check standard deviation
        np.linalg.norm(self.data_generator.data_pool, axis=1).std() == pytest.approx(
            1e-3, 0.0001
        )

    def test_concentric(self):
        """
        Test that concentric circles are constructed.
        """
        self.data_generator.radius = np.array([1.0, 3.0])

        self.data_generator.build_pool(100, noise=False)

        np.linalg.norm(self.data_generator.data_pool[0:99], axis=1).mean() == 1.0
        np.linalg.norm(self.data_generator.data_pool[100:199], axis=1).mean() == 3.0
        np.linalg.norm(self.data_generator.data_pool, axis=1).mean() == 2.0
