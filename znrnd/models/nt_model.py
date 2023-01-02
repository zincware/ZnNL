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
Module for the neural tangents infinite width network models.
"""
import logging
from typing import Callable, Sequence, Union

import jax
import jax.numpy as np
import neural_tangents as nt
from neural_tangents.stax import serial

from znrnd.accuracy_functions.accuracy_function import AccuracyFunction
from znrnd.models.jax_model import JaxModel
from znrnd.optimizers.trace_optimizer import TraceOptimizer
from znrnd.utils import normalize_covariance_matrix

logger = logging.getLogger(__name__)


class NTModel(JaxModel):
    """
    Class for a neural tangents model.
    """

    def __init__(
        self,
        loss_fn: Callable,
        optimizer: Union[Callable, TraceOptimizer],
        input_shape: tuple,
        training_threshold: float = 0.01,
        nt_module: serial = None,
        accuracy_fn: AccuracyFunction = None,
        batch_size: int = 10,
        trace_axes: Union[int, Sequence[int]] = (-1,),
        seed: int = None,
    ):
        """
        Constructor for a Flax model.

        Parameters
        ----------
        loss_fn : Callable
                A function to use in the loss computation.
        optimizer : Callable
                optimizer to use in the training. OpTax is used by default and
                cross-compatibility is not assured.
        input_shape : tuple
                Shape of the NN input.
        training_threshold : float
                The loss value at which point you consider the model trained.
        nt_module : serial
                NT model used.
        accuracy_fn : AccuracyFunction
                Accuracy function to use for accuracy computation.
        batch_size : int, default 10
                Batch size to use in the NTK computation.
        trace_axes : Union[int, Sequence[int]]
                Tracing over axes of the NTK.
                The default value is trace_axes(-1,), which reduces the NTK to a tensor
                of rank 2.
                For a full NTK set trace_axes=().
        seed : int, default None
                Random seed for the RNG. Uses a random int if not specified.
        """
        self.init_fn = nt_module[0]
        self.apply_fn = jax.jit(nt_module[1])
        self.kernel_fn = nt.batch(nt_module[2], batch_size=batch_size)
        self.empirical_ntk = nt.batch(
            nt.empirical_ntk_fn(f=self.apply_fn, trace_axes=trace_axes),
            batch_size=batch_size,
        )
        self.empirical_ntk_jit = jax.jit(self.empirical_ntk)

        # Save input parameters, call self.init_model
        super().__init__(
            loss_fn,
            optimizer,
            input_shape,
            training_threshold,
            accuracy_fn,
            seed,
        )

    def _init_params(self, kernel_init: Callable = None, bias_init: Callable = None):
        """Initialize a state for the model parameters.

        Parameters
        ----------
        kernel_init : Callable
                Define the kernel initialization.
        bias_init : Callable
                Define the bias initialization.

        Returns
        -------
        Initial state for the model parameters.
        """
        if kernel_init is not None:
            raise NotImplementedError(
                "Currently, there is no option customize the weight initialization. "
            )
        if bias_init is not None:
            raise NotImplementedError(
                "Currently, there is no option customize the bias initialization. "
            )

        _, params = self.init_fn(self.rng(), self.input_shape)

        return params

    def apply(self, params: dict, inputs: np.ndarray):
        """Apply the model to a feature vector.

        Parameters
        ----------
        params: dict
                Contains the model parameters to use for the model computation.
        inputs : np.ndarray
                Feature vector on which to apply the model.

        Returns
        -------
        Output of the model.
        """
        return self.apply_fn(params, inputs)

    def compute_ntk(
        self,
        x_i: np.ndarray,
        x_j: np.ndarray = None,
        normalize: bool = True,
        infinite: bool = False,
    ):
        """
        Compute the NTK matrix for the model.

        Parameters
        ----------
        x_i : np.ndarray
                Dataset for which to compute the NTK matrix.
        x_j : np.ndarray (optional)
                Dataset for which to compute the NTK matrix.
        normalize : bool (default = True)
                If true, divide each row by its max value.
        infinite : bool (default = False)
                If true, compute the infinite width limit as well.

        Returns
        -------
        NTK : dict
                The NTK matrix for both the empirical and infinite width computation.
        """
        if x_j is None:
            x_j = x_i
        empirical_ntk = self.empirical_ntk_jit(x_i, x_j, self.model_state.params)

        if infinite:
            infinite_ntk = self.kernel_fn(x_i, x_j, "ntk")
        else:
            infinite_ntk = None

        if normalize:
            empirical_ntk = normalize_covariance_matrix(empirical_ntk)
            if infinite:
                infinite_ntk = normalize_covariance_matrix(infinite_ntk)

        return {"empirical": empirical_ntk, "infinite": infinite_ntk}
