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
import numpy as onp
from numpy.testing import assert_array_equal

from znrnd.core.utils.matrix_utils import (
    compute_eigensystem,
    normalize_covariance_matrix,
)


class TestMatrixUtils:
    """
    Test the matrix utils class.
    """

    def test_unscaled_eigenvalues(self):
        """
        Test the computation of unscaled eigen systems.
        """
        matrix = np.eye(2)

        values, vectors = compute_eigensystem(matrix, normalize=False)

        assert_array_equal(np.real(values), [1, 1])

    def test_scaled_eigenvalues(self):
        """
        Test the computation of scaled eigen systems.

        Notes
        -----
        TODO: Check if this means that the scaled are eigen systems are invalid. In this
              case, we violate the fundamental definition of the identity matrix.
        """
        matrix = np.eye(2)

        values, vectors = compute_eigensystem(matrix, normalize=True)

        assert_array_equal(np.real(values), [0.5, 0.5])

    def test_normalizing_covariance_matrix(self):
        """
        Test that the covariance matrix is correctly normalized.

        Returns
        -------
        We fix the diagonals and test whether it performs the correct operation. You
        should note that this is not a correctly normalized covariance matrix rather
        one that can be tested well to properly scale under the normalization procedure.
        """
        # 4x4 covariance matrix
        covariance_matrix = onp.random.uniform(low=0, high=3, size=(4, 4))

        # Fix diagonals
        for i in range(4):
            covariance_matrix[i, i] = i + 3

        normalized_matrix = normalize_covariance_matrix(covariance_matrix)

        # Assert diagonals are 1
        assert_array_equal(
            np.diagonal(normalized_matrix), np.array([1.0, 1.0, 1.0, 1.0])
        )

        # Test 1st row
        row = 0
        row_mul = row + 3
        multiplier = np.sqrt(
            np.array([3 * row_mul, 4 * row_mul, 5 * row_mul, 6 * row_mul])
        )
        truth_array = covariance_matrix[row] / multiplier
        assert_array_equal(normalized_matrix[row], truth_array)

        # Test 2nd row
        row = 1
        row_mul = row + 3
        multiplier = np.sqrt(
            np.array([3 * row_mul, 4 * row_mul, 5 * row_mul, 6 * row_mul])
        )
        truth_array = covariance_matrix[row] / multiplier
        assert_array_equal(normalized_matrix[row], truth_array)

        # Test 3rd row
        row = 2
        row_mul = row + 3
        multiplier = np.sqrt(
            np.array([3 * row_mul, 4 * row_mul, 5 * row_mul, 6 * row_mul])
        )
        truth_array = covariance_matrix[row] / multiplier
        assert_array_equal(normalized_matrix[row], truth_array)

        # Test 4th row
        row = 3
        row_mul = row + 3
        multiplier = np.sqrt(
            np.array([3 * row_mul, 4 * row_mul, 5 * row_mul, 6 * row_mul])
        )
        truth_array = covariance_matrix[row] / multiplier
        assert_array_equal(normalized_matrix[row], truth_array)
