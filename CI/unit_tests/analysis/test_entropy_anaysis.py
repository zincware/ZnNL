"""
ZnRND: A zincwarecode package.

License
-------
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the zincwarecode Project.

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
import pytest
from jax.lib import xla_bridge

from znnl.analysis import EntropyAnalysis


class TestEntropyAnalysis:
    """
    Test suite for the entropy analysis module.
    """

    @classmethod
    def setup_class(cls):
        """
        Prepare the test.

        Returns
        -------

        """
        # Print the device being used.
        print(xla_bridge.get_backend().platform)

    def test_von_neumann_entropy(self):
        """
        Test the computation of the von-Neumann entropy.

        Returns
        -------

        """
        matrix = np.eye(2) * 0.5

        calculator = EntropyAnalysis(matrix=matrix)

        entropy = calculator.compute_von_neumann_entropy(effective=False)

        assert np.real(entropy) == pytest.approx(0.69, 0.01)

        entropy = calculator.compute_von_neumann_entropy(effective=True)

        assert np.real(entropy) == pytest.approx(1.0, 0.001)
