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
Module for the computation of the matrix covariance entropy.
"""

import numpy as np

from znnl.analysis.entropy import EntropyAnalysis
from znnl.utils.matrix_utils import normalize_gram_matrix


def compute_covariance_entropy(matrix: np.ndarray):
    """
    Function to compute the covariance entropy of a matrix.

    Parameters
    ----------
    Matrix of which to compute the covariance entropy.

    Returns
    -------
    Covariance entropy of the matrix."""

    cov_matrix = normalize_gram_matrix(matrix)
    calculator = EntropyAnalysis(matrix=cov_matrix)
    entropy = calculator.compute_von_neumann_entropy(
        effective=False, normalize_eig=True
    )

    return entropy
