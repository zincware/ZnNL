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
Test the confined particles class.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import jax.numpy as np
import pytest

from znrnd.jax_core.data.confined_particles import ConfinedParticles


class TestConfinedParticles:
    """
    Class for the confined particles test suite.
    """

    @classmethod
    def setup_class(cls):
        """
        Prepare the test suite.
        """
        cls.data_generator = ConfinedParticles()

    def test_fill_box(self):
        """
        Test that particles are added correctly.

        Notes
        -----
        For a uniform distribution:

        mean: 0.5 * (min + max)
        variance: 1/12 * (max - min) ** 2
        """
        self.data_generator.build_pool(500)

        # 0.5 * (0 + 2) = 1.0
        np.mean(self.data_generator.data_pool, axis=0).mean() == pytest.approx(
            1.0, 0.01
        )

        # 1/12 * (2 - 0)**2 = 1/3
        np.std(self.data_generator.data_pool, axis=0).mean() == pytest.approx(
            0.333, 0.001
        )
