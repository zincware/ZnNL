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

from znrnd.jax_core.data.points_on_a_lattice import PointsOnLattice


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

    def test_lattice_build(self):
        """
        Test that the lattice is constructed correctly.
        """
        self.data_generator.build_pool()

        assert True in [all([0, 0] == item) for item in self.data_generator.data_pool]
        assert True in [all([-5, -5] == item) for item in self.data_generator.data_pool]
        assert True in [all([5, -4] == item) for item in self.data_generator.data_pool]
        assert True in [all([-1, -1] == item) for item in self.data_generator.data_pool]
        assert True in [all([0, -1] == item) for item in self.data_generator.data_pool]
