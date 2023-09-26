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
Module for the computation of the matrix magnitude entropy.
"""

import numpy as np

from znnl.analysis.entropy import EntropyAnalysis
from znnl.utils.matrix_utils import compute_magnitude_density


def compute_magnitude_entropy(matrix: np.ndarray):
    """
    Function to compute the magnitude entropy of a matrix.

    Parameters
    ----------
    Matrix of which to compute the magnitude entropy.

    Returns
    -------
    Magnitude entropy of the matrix."""

    magnitude_dist = compute_magnitude_density(gram_matrix=matrix)
    entropy = EntropyAnalysis.compute_shannon_entropy(magnitude_dist)

    return entropy
