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
Test the points on lattice module.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np

from znnl.data.points_on_a_lattice import PointsOnLattice


class TestPointsOnLattice:
    """
    Test the points on a lattice data generator.
    """

    @classmethod
    def setup_class(cls):
        """
        Prepare the class for test.
        """
        cls.data_generator = PointsOnLattice()

    def test_lattice_build_default(self):
        """
        Test that the lattice is constructed correctly using the default boundary.

        Test for particular points.
        Test for correct spacing.
            Calculate the difference for shifting the lattice by one in each dimension.
        Test for the size of the lattice.
        """
        self.data_generator.build_pool()

        # Test particular points
        assert True in [all([0, 0] == item) for item in self.data_generator.data_pool]
        assert True in [all([-5, -5] == item) for item in self.data_generator.data_pool]
        assert True in [all([5, -4] == item) for item in self.data_generator.data_pool]
        assert True in [all([-1, -1] == item) for item in self.data_generator.data_pool]
        assert True in [all([0, -1] == item) for item in self.data_generator.data_pool]

        # Test dimension 1
        diff = (
            np.roll(self.data_generator.data_pool[:, 0], -1)
            - self.data_generator.data_pool[:, 0]
        ).reshape(11, 11)
        # Test lattice size
        np.testing.assert_array_equal(diff[:, -1], np.array([-10] * 11))
        # Test spacing
        np.testing.assert_array_equal(diff[:, :-1], np.ones_like(diff[:, :-1]))

        # Test dimension 2
        diff = (
            np.roll(self.data_generator.data_pool[:, 1], -11)
            - self.data_generator.data_pool[:, 1]
        ).reshape(11, 11)
        # Test lattice size
        np.testing.assert_array_equal(diff[-1, :], np.array([-10] * 11))
        # Test spacing
        np.testing.assert_array_equal(diff[:-1, :], np.ones_like(diff[:-1, :]))

    def test_lattice_build_non_default(self):
        """
        Test that the lattice is constructed correctly using a non-default boundary.
        """
        self.data_generator.build_pool(
            x_points=21, y_points=21, x_boundary=2.0, y_boundary=2.0
        )

        assert True in [
            np.allclose([0.0, 0.0], item) for item in self.data_generator.data_pool
        ]
        assert True in [
            np.allclose([-1.2, -0.6], item) for item in self.data_generator.data_pool
        ]
        assert True in [
            np.allclose([0.6, -0.8], item) for item in self.data_generator.data_pool
        ]
        assert True in [
            np.allclose([-2.0, -2.0], item) for item in self.data_generator.data_pool
        ]
        assert True in [
            np.allclose([0.0, -1.2], item) for item in self.data_generator.data_pool
        ]

        # Test dimension 1
        diff = (
            np.roll(self.data_generator.data_pool[:, 0], -1)
            - self.data_generator.data_pool[:, 0]
        ).reshape(21, 21)
        # Test lattice size
        np.testing.assert_array_equal(diff[:, -1], np.array([-4] * 21))
        # Test spacing
        np.testing.assert_array_almost_equal(
            diff[:, :-1], 0.2 * np.ones_like(diff[:, :-1])
        )

        # Test dimension 2
        diff = (
            np.roll(self.data_generator.data_pool[:, 1], -21)
            - self.data_generator.data_pool[:, 1]
        ).reshape(21, 21)
        # Test lattice size
        np.testing.assert_array_equal(diff[-1, :], np.array([-4] * 21))
        # Test spacing
        np.testing.assert_array_almost_equal(
            diff[:-1, :], 0.2 * np.ones_like(diff[:-1, :])
        )
