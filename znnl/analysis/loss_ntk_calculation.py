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
from typing import Callable
from znnl.models.jax_model import JaxModel
import jax
import jax.numpy as np


class loss_ntk_calculation:
    def __init__(
        self,
        metric_fn: Callable,
        model: JaxModel,
        dataset: dict,
    ):
        """Constructor for the loss ntk calculation class."""

        # Set the attributes
        self.metric_fn = metric_fn
        self.ntk_batch_size = model.ntk_batch_size
        self.store_on_device = model.store_on_device
        self.trace_axes = model.trace_axes
        self.input_shape = dataset["inputs"].shape
        self.input_dimension = int(np.prod(np.array(self.input_shape[1:])))
        self.target_shape = dataset["targets"].shape
        self.metric_fn = metric_fn
        self.apply_fn = model._ntk_apply_fn

        # Prepare NTK calculation
        empirical_ntk = nt.batch(
            nt.empirical_ntk_fn(
                f=self._function_for_loss_ntk,
                trace_axes=self.trace_axes,
            ),
            batch_size=self.ntk_batch_size,
            store_on_device=self.store_on_device,
        )
        self.empirical_ntk_jit = jax.jit(empirical_ntk)

    def _function_for_loss_ntk(self, params, datapoint) -> float:
        """
        Helper function to create a subloss apply function.
        The datapoint here has to be shaped so that its an array of length
        input dimension + output dimension.
        This is done so that the inputs and targets can be understood
        by the neural tangents empirical_ntk_fn function.
        """
        _input = datapoint[: self.input_dimension]
        _target = datapoint[self.input_dimension :]
        return self.metric_fn(
            self.apply_fn(params, _input),
            _target,
        )

    def compute_loss_ntk(
        self, x_i: np.ndarray, x_j: np.ndarray, model: JaxModel, infinite: bool = False
    ):
        """
        Compute the loss NTK matrix for the model.
        The dataset gets reshaped to (n_data, input_dimension + output_dimension)
        so that the neural tangents empirical_ntk_fn function can take each input
        target pair as its input.

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

        x_i = np.concatenate(
            (
                x_i["inputs"].reshape(x_i["inputs"].shape[0], -1),
                x_i["targets"].reshape(x_i["targets"].shape[0], -1),
            ),
            axis=1,
        )

        if x_j is None:
            x_j = x_i
        else:
            x_j = np.concatenate(
                (
                    x_j["inputs"].reshape(x_j["inputs"].shape[0], -1),
                    x_j["targets"].reshape(x_j["targets"].shape[0], -1),
                ),
                axis=1,
            )

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
