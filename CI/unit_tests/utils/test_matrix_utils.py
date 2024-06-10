"""
ZnNL: A Zincwarecode package.

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
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import jax.numpy as np
import numpy as onp
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_raises

from znnl.utils.matrix_utils import (
    compute_eigensystem,
    compute_magnitude_density,
    flatten_rank_4_tensor,
    normalize_gram_matrix,
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

        assert_array_almost_equal(np.real(values), [1.0, 1.0])

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

        normalized_matrix = normalize_gram_matrix(covariance_matrix)

        # Assert diagonals are 1
        assert_array_almost_equal(
            np.diagonal(normalized_matrix), np.array([1.0, 1.0, 1.0, 1.0])
        )

        # Test 1st row
        row = 0
        row_mul = row + 3
        multiplier = np.sqrt(
            np.array([3 * row_mul, 4 * row_mul, 5 * row_mul, 6 * row_mul])
        )
        truth_array = covariance_matrix[row] / multiplier
        assert_array_almost_equal(normalized_matrix[row], truth_array)

        # Test 2nd row
        row = 1
        row_mul = row + 3
        multiplier = np.sqrt(
            np.array([3 * row_mul, 4 * row_mul, 5 * row_mul, 6 * row_mul])
        )
        truth_array = covariance_matrix[row] / multiplier
        assert_array_almost_equal(normalized_matrix[row], truth_array)

        # Test 3rd row
        row = 2
        row_mul = row + 3
        multiplier = np.sqrt(
            np.array([3 * row_mul, 4 * row_mul, 5 * row_mul, 6 * row_mul])
        )
        truth_array = covariance_matrix[row] / multiplier
        assert_array_almost_equal(normalized_matrix[row], truth_array)

        # Test 4th row
        row = 3
        row_mul = row + 3
        multiplier = np.sqrt(
            np.array([3 * row_mul, 4 * row_mul, 5 * row_mul, 6 * row_mul])
        )
        truth_array = covariance_matrix[row] / multiplier
        assert_array_almost_equal(normalized_matrix[row], truth_array)

    def test_compute_magnitude_density(self):
        """
        Test that the magnitude density is correctly computed.

            * Compute a gram matrix
            * Compute magnitude density
            * Compare to norm of vectors
        """
        # Create a random array
        array = onp.random.random((7, 10))
        # Compute a scalar product matrix of the array
        matrix = np.einsum("ij, kj -> ik", array, array)
        # Compute the density of array amplitudes
        array_norm = onp.linalg.norm(array, ord=2, axis=1)
        array_norm_density = array_norm / array_norm.sum()

        # Evaluate the magnitude density with the function that is to be tested
        mag_density = compute_magnitude_density(matrix)

        assert_array_almost_equal(array_norm_density, mag_density)

    def test_flatten_rank_4_tensor(self):
        """
        Test the flattening of a rank 4 tensor.
        """
        # Check for assertion errors
        tensor = np.arange(24).reshape((2, 3, 2, 2))
        assert_raises(ValueError, flatten_rank_4_tensor, tensor)
        tensor = np.arange(24).reshape((2, 2, 3, 2))
        assert_raises(ValueError, flatten_rank_4_tensor, tensor)

        # Check the flattening
        tensor = np.arange(4 * 4).reshape(2, 2, 2, 2)
        assertion_matrix = np.array(
            [[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]]
        )
        assert_array_equal(flatten_rank_4_tensor(tensor), assertion_matrix)
