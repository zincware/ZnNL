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


def compute_eigensystem(matrix: np.ndarray, reduce: bool = True):
    """
    Compute the eigenspace of a matrix.

    Parameters
    ----------
    matrix : np.ndarray
            Matrix for which the space should be computed.
    reduce : bool (default=True)
            If true, the eigenvalues are divided by the size of the zeroth axis of the
            given matrix. This is equivalent to dividing by the size of the dataset.

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
    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    if reduce:
        eigenvalues /= matrix.shape[0]

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
