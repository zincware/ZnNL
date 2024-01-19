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

import neural_tangents as nt
from typing import Callable, Union, Sequence
from znnl.models.jax_model import JaxModel
import jax
import jax.numpy as np


class loss_ntk_calculation:
    def __init__(
        self,
        metric_fn: Callable,
        model: JaxModel,
    ):
        """Constructor for the loss ntk calculation class."""

        # Set the attributes
        self.metric_fn = metric_fn
        self.ntk_batch_size = model.ntk_batch_size
        self.store_on_device = model.store_on_device
        self.trace_axes = model.trace_axes

        # Set the loss ntk function
        _function_for_loss_ntk = lambda x, y: self._function_for_loss_ntk_helper(
            x, y, metric_fn, model._ntk_apply_fn
        )

        # Prepare NTK calculation
        empirical_ntk = nt.batch(
            nt.empirical_ntk_fn(f=_function_for_loss_ntk, trace_axes=self.trace_axes),
            batch_size=self.ntk_batch_size,
            store_on_device=self.store_on_device,
        )
        self.empirical_ntk_jit = jax.jit(empirical_ntk)

    def _function_for_loss_ntk_helper(params, dataset, metric_fn, apply_fn) -> float:
        """
        Helper function to create a subloss apply function of structure
        (params, dataset) -> loss.
        """
        return metric_fn(apply_fn(params, dataset["inputs"]), dataset["targets"])

    def compute_loss_ntk(
        self, x_i: np.ndarray, x_j: np.ndarray, model: JaxModel, infinite: bool = False
    ):
        """
        Compute the loss NTK matrix for the model.

        Parameters
        ----------
        x_i : np.ndarray
                Dataset for which to compute the loss NTK matrix.
        x_j : np.ndarray (optional)
                Dataset for which to compute the loss NTK matrix.
        infinite : bool (default = False)
                If true, compute the infinite width limit as well.

        Returns
        -------
        NTK : dict
                The NTK matrix for both the empirical and infinite width computation.
        """

        if x_j is None:
            x_j = x_i
        empirical_ntk = self.empirical_ntk_jit(
            x_i,
            x_j,
            {
                "params": model.model_state.params,
                "batch_stats": model.model_state.batch_stats,
            },
        )

        if infinite:
            try:
                infinite_ntk = self.kernel_fn(x_i, x_j, "ntk")
            except AttributeError:
                raise NotImplementedError("Infinite NTK not available for this model.")
        else:
            infinite_ntk = None

        return {"empirical": empirical_ntk, "infinite": infinite_ntk}
