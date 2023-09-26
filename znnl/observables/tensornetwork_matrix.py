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
Module for the computation of the tensornetwork matrix.
"""

import jax.numpy as jnp
import numpy as np
from jax import vmap


def compute_tensornetwork_matrix(ntk: np.ndarray, targets: np.ndarray):
    """
    Function that calculates the tensornetwork matrix.

    Inputs
    ------
    Takes the ntk and the array of targets as an input.

    Returns
    -------
    Returns the tensornetwork matrix.
    """
    unique_targets, target_indices, target_counts = np.unique(
        targets, return_counts=True, return_inverse=True
    )
    sortlist = np.argsort(target_indices)
    # sorting of the ntk
    sorted_ntk = ntk[sortlist, ...]
    sorted_ntk = sorted_ntk[:, sortlist, ...]
    if len(sorted_ntk.shape) == 4:
        sorted_ntk = np.mean(sorted_ntk, axis=(2, 3))

    """
    this is a previous way of computating the matrix which is slower because of the
    for loops.
    -----------------
    n_targets = len(target_counts)
    tensornetwork_matrix = np.empty(shape=(n_targets, n_targets))
    for i in range(n_targets):
        for j in range(n_targets):
            start1, end1 = _get_start_and_end_value(i, target_counts)
            start2, end2 = _get_start_and_end_value(j, target_counts)
            tensornetwork_matrix[i, j] = np.average(
                sorted_ntk[start1:end1, start2:end2, ...]
            )
    """

    endlist = np.cumsum(target_counts)
    startlist = np.append(np.array([0]), endlist[:-1])

    def mapfunc(start1, end1, start2, end2):
        """
        Takes average over part of the ntk from start1 to end1 in dimension 1
        and the same for dimension2.
        Looks way more complicated than it should be because of vmap fun with
        dynamic slicing.

        Basicall this doesnt take a slice of the ntk but sets everything outside
        of the desired value to 0 and then does a average over the remaining values.
        """
        slice1bool = jnp.logical_or(
            jnp.arange(sorted_ntk.shape[0]) < start1,
            jnp.arange(sorted_ntk.shape[0]) >= end1,
        )
        slice2bool = jnp.logical_or(
            jnp.arange(sorted_ntk.shape[0]) < start2,
            jnp.arange(sorted_ntk.shape[0]) >= end2,
        )
        sliceboolmatrix = jnp.logical_or(slice1bool[:, None], slice2bool)

        partialntk = jnp.where(sliceboolmatrix, 0, sorted_ntk)

        return jnp.sum(partialntk) / (end1 - start1) / (end2 - start2)

    inner_map = vmap(mapfunc, in_axes=(None, None, 0, 0))
    outer_map = vmap(inner_map, in_axes=(0, 0, None, None))
    tensornetwork_matrix = outer_map(startlist, endlist, startlist, endlist)

    return tensornetwork_matrix


def _get_start_and_end_value(i, target_counts):
    """
    Function that takes the index of the desired sub_ntk block and the list of
    the number of data_points per target as an input and returns the start and
    end value of the whole ntk.
    """
    start = np.sum(target_counts[:i])
    end = np.sum(target_counts[: i + 1])

    return start, end
