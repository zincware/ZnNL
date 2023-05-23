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
Module for the computation of the Fisher trace.
"""
import jax
import jax.numpy as np


def compute_fisher_trace(loss_derivative: np.ndarray, ntk: np.ndarray) -> float:
    """
    Compute the Fisher matrix trace from the NTK.

    Parameters
    ----------
    loss_derivative : np.ndarray (n_data_points, network_output)
            Loss derivative to use in the computation.
    ntk : np.ndarray
            NTK of the network in one state.

    Returns
    -------
    fisher_trace : float
            Trace of the Fisher matrix corresponding to the NTK.
    """
    try:
        assert len(ntk.shape) == 4
    except AssertionError:
        raise TypeError(
            "The ntk needs to be rank 4 for the fisher trace calculation."
            "Maybe you have set the model to trace over the output dimensions?"
        )

    def _inner_fn(a, b, c):
        """
        Function to be mapped over.
        """
        return a * b * c

    map_1 = jax.vmap(_inner_fn, in_axes=(None, 0, 0))
    map_2 = jax.vmap(map_1, in_axes=(0, None, 0))
    map_3 = jax.vmap(map_2, in_axes=(0, 0, 0))

    dataset_size = loss_derivative.shape[0]
    indices = np.arange(dataset_size)
    fisher_trace = np.sum(
        map_3(loss_derivative, loss_derivative, ntk[indices, indices, :, :])
    )

    return fisher_trace / dataset_size
