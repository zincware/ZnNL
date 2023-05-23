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
Module for the computation of the tensornetwork entropy.
"""

import jax.numpy as np


def compute_tensornetwork_entropy(ntk: np.ndarray, targets: np.ndarray):
    """
    Function that calculates the tensornetwork entropy.

    Inputs
    ------
    Takes the ntk and the array of targets as an input.

    Returns
    -------
    Returns the tensornetwork entropy.
    """
    sortlist = np.argsort(targets)
    unique_targets, target_counts = np.unique(targets[sortlist], return_counts=True)
    n_targets = len(unique_targets)
    sorted_ntk = ntk[sortlist, sortlist, ...]

    tensornetwork_matrix = np.empty(shape=(n_targets, n_targets))
    for i in range(n_targets):
        for j in range(n_targets):
            start1, end1 = _get_start_and_end_value(i, target_counts)
            start2, end2 = _get_start_and_end_value(j, target_counts)
            tensornetwork_matrix[i, j] = np.average(
                sorted_ntk[start1:end1, start2:end2, ...]
            )

    eigenvalues = np.linalg.eigvals(tensornetwork_matrix)
    entropy = np.sum(eigenvalues * np.log(eigenvalues))

    return entropy


def _get_start_and_end_value(i, target_counts):
    """
    Function that takes the index of the desired sub_ntk block and the list of
    the number of data_points per target as an input and returns the start and
    end value of the whole ntk.
    """
    if i - 1 < 0:
        start = 0
    else:
        start = target_counts[i - 1]
    end = target_counts[i]

    return start, end
