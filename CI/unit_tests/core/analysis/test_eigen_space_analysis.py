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
Test the eigen space analysis tools

Notes
-----
"""
from znrnd.core.analysis.eigensystem import EigenSpaceAnalysis

import jax.numpy as np
import jax
import jax.random as random

import numpy as onp


class TestEigenspaceAnalysis:
    """
    Test suite for the Eigenspace analysis module.
    """
    @classmethod
    def setup_class(cls):
        """
        Prepare the test.

        Returns
        -------

        """
        matrix = random.uniform(key=random.PRNGKey(1), shape=(500, 500))

        cls.calculator = EigenSpaceAnalysis(matrix=matrix)

    def test_eigenvalue_density(self):
        """
        Test the eigenvalue density construction.

        We test this for a Wigner distribution and ensure that the semi-circle rule is
        obeyed.

        Returns
        -------

        """
        self.calculator.compute_eigenvalue_density(nbins=500)

    def test_eigenvalue_spacing_distribution(self):
        """
        Check that the Wigner surmise is satisfied.

        Returns
        -------

        """
        self.calculator.compute_eigenvalue_spacing_density(nbins=5000)
