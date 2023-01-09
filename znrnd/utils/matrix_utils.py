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
Module for helper functions related to matrices
"""
import jax.numpy as np


def compute_eigensystem(matrix: np.ndarray, normalize: bool = False, clip: bool = True):
    """
    Compute the eigenspace of a matrix.

    Parameters
    ----------
    matrix : np.ndarray
            Matrix for which the space should be computed.
    normalize : bool (default=False)
            If true, the eigenvalues are divided by the sum of the eigenvalues of the
            given matrix. This is equivalent to dividing by the size of the dataset.
    clip : bool (default=True)
            Clip the eigenvalues to a very small number to avoid negatives. This should
            only be used if you are sure that negative numbers only arise due to some
            numeric reason and that they should not exist.

    Returns
    -------
    eigenvectors : np.ndarray
            Eigenvectors of the matrix
    eigenvalues : np.ndarray
            Eigenvalues of the matrix

    Notes
    -----
    TODO: Extend sorting such that when returned, the ith eigenvector belongs with the
          ith eigenvalue. Currently the sorting destroys this.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)

    if clip:
        # TODO Add some logging and an argument in the methods that call this function.
        eigenvalues = np.clip(eigenvalues, 1e-14, None)

    if normalize:
        eigenvalues /= eigenvalues.sum()

    return eigenvalues[::-1].sort(), eigenvectors


def normalize_covariance_matrix(covariance_matrix: np.ndarray):
    """
    Method for normalizing a covariance matrix.

    Returns
    -------
    normalized_covariance_matrix : np.ndarray
            A normalized covariance matrix, i.e, the matrix given if all of its inputs
            had been normalized.
    """
    order = np.shape(covariance_matrix)[0]

    diagonals = np.diagonal(covariance_matrix)

    repeated_diagonals = np.repeat(diagonals[None, :], order, axis=0)

    normalizing_matrix = np.sqrt(repeated_diagonals * repeated_diagonals.T)

    return covariance_matrix / normalizing_matrix


def calculate_l_pq_norm(matrix: np.ndarray, p: int = 2, q: int = 2):
    """
    Calculate the L_pq norm of a matrix.

    The norm calculates (sum_j (sum_i abs(a_ij)^p)^(p/q) )^(1/q)
    For the defaults (p = 2, q = 2) the function calculates the Frobenius
    (or Hilbert-Schmidt) norm.

    Parameters
    ----------
    matrix: np.ndarray
            Matrix to calculate the L_pq norm of
    p: int
            Inner power of the norm.
    q: int
            Outer power of the norm.

    Returns
    -------
    L_qp norm of the matrix.
    """
    inner_sum = np.sum(np.power(matrix, q), axis=-1)
    outer_sum = np.sum(np.power(inner_sum, q / p), axis=-1)
    return np.power(outer_sum, 1 / q)
