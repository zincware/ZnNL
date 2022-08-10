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
None of these tests enforce any of the surmises. You can look at the generated plots
to confirm that they are true but we need to add some return on the methods and check
that the Wigner semi-circle rule and Wigner surmise are met for this data.
"""
import jax.random as random
from jax.lib import xla_bridge

from znrnd.analysis.eigensystem import EigenSpaceAnalysis


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

        # Print the device being used.
        print(xla_bridge.get_backend().platform)

        cls.calculator = EigenSpaceAnalysis(matrix=matrix)

    def test_eigenvalue_density(self):
        """
        Test the eigenvalue density construction.

        We test this for a Wigner distribution and ensure that the semi-circle rule is
        obeyed.

        Returns
        -------

        """
        self.calculator.compute_eigenvalue_density(n_bins=500)

    def test_eigenvalue_spacing_distribution(self):
        """
        Check that the Wigner surmise is satisfied.

        Returns
        -------

        """
        self.calculator.compute_eigenvalue_spacing_density(n_bins=5000)
