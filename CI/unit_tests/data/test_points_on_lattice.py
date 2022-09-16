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

from znrnd.data.points_on_a_lattice import PointsOnLattice
import numpy as np


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
        Test that the lattice is constructed correctly.
        """
        self.data_generator.build_pool()

        assert True in [
            np.allclose([0.0, 0.0], item) for item in self.data_generator.data_pool
        ]
        assert True in [
            np.allclose([-0.6, -0.6], item) for item in self.data_generator.data_pool
        ]
        assert True in [
            np.allclose([0.6, -0.8], item) for item in self.data_generator.data_pool
        ]
        assert True in [
            np.allclose([-1.0, -1.0], item) for item in self.data_generator.data_pool
        ]
        assert True in [
            np.allclose([0.0, -1.0], item) for item in self.data_generator.data_pool
        ]

    def test_lattice_build_non_default(self):
        """
        Test that the lattice is constructed correctly.
        """
        self.data_generator.build_pool(x_points=20, y_points=20, boundary=2.0)

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
