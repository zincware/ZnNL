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


def normalize_gram_matrix(gram_matrix: np.ndarray):
    """
    Method for normalizing a covariance matrix.

    Returns
    -------
    normalized_covariance_matrix : np.ndarray
            A normalized covariance matrix, i.e, the matrix given if all of its inputs
            had been normalized.
    """
    order = np.shape(gram_matrix)[0]

    diagonals = np.diagonal(gram_matrix)

    repeated_diagonals = np.repeat(diagonals[None, :], order, axis=0)

    normalizing_matrix = np.sqrt(repeated_diagonals * repeated_diagonals.T)

    # Check for zeros in the normalizing matrix
    normalizing_matrix = np.where(
        normalizing_matrix == 0, 0, 1 / normalizing_matrix
    )  # Avoid division by zero

    return gram_matrix * normalizing_matrix


def compute_magnitude_density(gram_matrix: np.ndarray) -> np.ndarray:
    """
    Method for computing the normalized magnitude density of each component of a gram
    matrix.

    Parameters
    ----------
    gram_matrix : np.ndarray
            Covariance matrix to calculate the magnitude distribution of.

    Returns
    -------
    magnitude_density: np.ndarray
            Magnitude density of the individual entries.
    """
    magnitudes = np.sqrt(np.diagonal(gram_matrix))
    density = magnitudes / magnitudes.sum()
    return density


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
    p: int (default=2)
            Inner power of the norm.
    q: int (default=2)
            Outer power of the norm.

    Returns
    -------
    calculate_l_pq_norm: np.ndarray
            L_qp norm of the matrix.
    """
    inner_sum = np.sum(np.power(matrix, q), axis=-1)
    outer_sum = np.sum(np.power(inner_sum, q / p), axis=-1)
    return np.power(outer_sum, 1 / q)


def flatten_rank_4_tensor(tensor: np.ndarray) -> np.ndarray:
    """
    Flatten a rank 4 tensor to a rank 2 tensor using a specific reshaping.

    The tensor is assumed to be of shape (n, n, m, m). The reshaping is done by
    concatenating first with the third and then with the fourth dimension, resulting
    in a tensor of shape (n * m, n * m).

    Parameters
    ----------
    tensor : np.ndarray (shape=(n, n, m, m))
            Tensor to flatten.

    Returns
    -------
    flattened_tensor : np.ndarray (shape=(n * m, n * m))
            Flattened tensor.
    """

    if not tensor.shape[0] == tensor.shape[1]:
        raise ValueError("The first two dimensions of the tensor must be equal.")
    if not tensor.shape[2] == tensor.shape[3]:
        raise ValueError("The third and fourth dimensions of the tensor must be equal.")

    _tensor = np.moveaxis(tensor, [1, 2], [2, 1])
    return _tensor.reshape(
        _tensor.shape[0] * _tensor.shape[1], _tensor.shape[0] * _tensor.shape[1]
    )


def unflatten_rank_2_tensor(tensor: np.ndarray, n: int, m: int) -> np.ndarray:
    """
    Unflatten a rank 2 tensor to a rank 4 tensor using a specific reshaping.

    The tensor is assumed to be of shape (n * m, n * m). The reshaping is done by
    concatenating first with the third and then with the fourth dimension, resulting
    in a tensor of shape (n, n, m, m).

    Parameters
    ----------
    tensor : np.ndarray (shape=(n * m, n * m))
            Tensor to unflatten.
    n : int
            First dimension of the unflattened tensor.
    m : int
            Second dimension of the unflattened tensor.

    Returns
    -------
    unflattened_tensor : np.ndarray (shape=(n, n, m, m))
            Unflattened tensor.
    """

    if not n * m == tensor.shape[0]:
        raise ValueError(
            "The shape of the tensor does not match the given dimensions. "
            f"Expected {n * m} but got {tensor.shape[0]}."
        )
    if not n * m == tensor.shape[1]:
        raise ValueError(
            "The shape of the tensor does not match the given dimensions. "
            f"Expected {n * m} but got {tensor.shape[1]}."
        )

    _tensor = tensor.reshape(n, m, n, m)
    return np.moveaxis(_tensor, [2, 1], [1, 2])


def calculate_trace(matrix: np.ndarray, normalize: bool = False) -> np.ndarray:
    """
    Calculate the trace of a matrix, including optional normalization.

    Parameters
    ----------
    matrix : np.ndarray
            Matrix to calculate the trace of.
    normalize : bool (default=True)
            If true, the trace is normalized by the size of the matrix.

    Returns
    -------
    trace : np.ndarray
            Trace of the matrix.
    """
    normalization_factor = np.shape(matrix)[0] if normalize else 1
    return np.trace(matrix) / normalization_factor
