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
Unit tests for the matrix utils module.
"""
import jax.numpy as np

from znrnd.core.utils.matrix_utils import compute_eigensystem

from numpy.testing import assert_array_equal


class TestComputeEigensystem:
    """
    Test the compute eigensystem function.
    """

    def test_unscaled(self):
        """
        Test the computation of unscaled eigensystems.
        """
        matrix = np.eye(2)

        values, vectors = compute_eigensystem(matrix, reduce=False)

        assert_array_equal(np.real(values), [1, 1])

    def test_scaled(self):
        """
        Test the computation of scaled eigensystems.

        Notes
        -----
        TODO: Check if this means that the scaled are eigensystems are invalid. In this
              case, we violate the fundamental definition of the identity matrix.
        """
        matrix = np.eye(2)

        values, vectors = compute_eigensystem(matrix, reduce=True)

        assert_array_equal(np.real(values), [0.5, 0.5])
