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

from typing import Callable

import jax
import jax.numpy as np
import neural_tangents as nt

from znnl.models.jax_model import JaxModel


class LossNTKCalculation:
    def __init__(
        self,
        metric_fn: Callable,
        model: JaxModel,
        dataset: dict,
    ):
        """
        Constructor for the loss ntk calculation class.

        Parameters
        ----------

        metric_fn : Callable
                The metric function to be used for the loss calculation.
                !This has to be the metric, not the Loss!
                If you put in the Loss here you won't get an error but an
                incorrect result.

        model : JaxModel
                The model for which to calculate the loss NTK.

        dataset : dict
                The dataset for which to calculate the loss NTK.
                The dictionary should contain the keys "inputs" and "targets".
        """

        # Set the attributes
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
                trace_axes=(),
                vmap_axes=0,
            ),
            batch_size=self.ntk_batch_size,
            store_on_device=self.store_on_device,
        )
        self.empirical_ntk_jit = jax.jit(empirical_ntk)

    @staticmethod
    def _reshape_dataset(dataset):
        """
        Helper function to reshape the dataset for the Loss NTK calculation.

        Parameters
        ----------
        dataset : dict
                The dataset to be reshaped.
                Should contain the keys "inputs" and "targets".

        Returns
        -------
        reshaped_dataset : np.ndarray
                The reshaped dataset.
        """
        return np.concatenate(
            (
                dataset["inputs"].reshape(dataset["inputs"].shape[0], -1),
                dataset["targets"].reshape(dataset["targets"].shape[0], -1),
            ),
            axis=1,
        )

    @staticmethod
    def _unshape_data(
        datapoint: np.ndarray,
        input_dimension: int,
        input_shape: tuple,
        target_shape: tuple,
        batch_length: int,
    ):
        """
        Helper function to unshape the data for the subloss calculation.

        Parameters
        ----------
        datapoint : np.ndarray
                The datapoint to be unshaped.
        input_dimension : int
                The total dimension of the input, i.e. the product of its shape.
        input_shape : tuple
                The shape of the original input.
        target_shape : tuple
                The shape of the original target.

        Returns
        -------
        input: np.ndarray
                The unshaped input.
        target: np.ndarray
                The unshaped target.
        """
        return datapoint[:, :input_dimension].reshape(
            batch_length, *input_shape[1:]
        ), datapoint[:, input_dimension:].reshape(batch_length, *target_shape[1:])

    def _function_for_loss_ntk(self, params, datapoint) -> float:
        """
        Helper function to create a subloss apply function.
        The datapoint here has to be shaped so that its an array of length
        input dimension + output dimension. This is done so that the inputs
        and targets can be understood by the neural tangents empirical_ntk_fn
        function. It gets unpacked by the _unshape_data function in here.

        Parameters
        ----------
        params : dict
                The parameters of the model.
        datapoint : np.ndarray
                The datapoint for which to calculate the subloss. Shaped as
                described in the description of this function.

        Returns
        -------
        subloss : float
                The subloss for the given datapoint.
        """
        batch_length = datapoint.shape[0]
        _input, _target = self._unshape_data(
            datapoint,
            self.input_dimension,
            self.input_shape,
            self.target_shape,
            batch_length,
        )
        return self.metric_fn(
            self.apply_fn(params, _input),
            _target,
        )

    def compute_loss_ntk(
        self,
        x_i: np.ndarray,
        model: JaxModel,
        x_j: np.ndarray = None,
        infinite: bool = False,
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
        Loss NTK : dict
                The Loss NTK matrix for both the empirical and
                infinite width computation.
        """

        x_i = self._reshape_dataset(x_i)

        if x_j is None:
            x_j = x_i
        else:
            x_j = self._reshape_dataset(x_j)

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
