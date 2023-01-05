"""
ZnRND: A Zincwarecode package.

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
Module for the use of a Flax model with ZnRND.
"""
import logging
from typing import Callable, List, Sequence, Union

import jax
import jax.numpy as np
import neural_tangents as nt
from flax import linen as nn

from znrnd.accuracy_functions.accuracy_function import AccuracyFunction
from znrnd.models.jax_model import JaxModel
from znrnd.utils import normalize_covariance_matrix

logger = logging.getLogger(__name__)


class FundamentalModel(nn.Module):
    """
    Model to be called in the ZnRND FlaxModel
    """

    layer_stack: List[nn.Module]

    @nn.compact
    def __call__(self, feature_vector: np.ndarray):
        """
        Call method of the model.

        Parameters
        ----------
        feature_vector : np.ndarray
                Inputs to the model.

        Returns
        -------
        output from the model.
        """
        for item in self.layer_stack:
            feature_vector = item(feature_vector)

        return feature_vector


class FlaxModel(JaxModel):
    """
    Class for the Flax model in ZnRND.
    """

    def __init__(
        self,
        loss_fn: Callable,
        optimizer: Callable,
        input_shape: tuple,
        training_threshold: float = 0.01,
        batch_size: int = 10,
        layer_stack: List[nn.Module] = None,
        flax_module: nn.Module = None,
        trace_axes: Union[int, Sequence[int]] = (-1,),
        accuracy_fn: AccuracyFunction = None,
        seed: int = None,
    ):
        """
        Construct a Flax model.

        Parameters
        ----------
        layer_stack : List[nn.Module]
                A list of flax modules to be used in the call method.
        loss_fn : Callable
                A function to use in the loss computation.
        optimizer : Callable
                optimizer to use in the training. OpTax is used by default and
                cross-compatibility is not assured.
        input_shape : tuple
                Shape of the NN input.
        training_threshold : float
                The loss value at which point you consider the model trained.
        batch_size : int
                Size of batch to use in the NTk calculation.
        flax_module : nn.Module
                Flax module to use instead of building one from scratch here.
        accuracy_fn : Callable
                Ann accuracy function to use in the model analysis.
        seed : int, default None
                Random seed for the RNG. Uses a random int if not specified.
        """
        logger.info(
            "Flax models have occasionally experienced memory allocation issues on "
            "GPU. This is an ongoing bug that we are striving to fix soon."
        )
        if layer_stack is not None:
            self.model = FundamentalModel(layer_stack)
        if flax_module is not None:
            self.model = flax_module
        if layer_stack is None and flax_module is None:
            raise TypeError("Provide either a Flax nn.Module or a layer stack.")

        self.apply_fn = jax.jit(self.model.apply)

        self.empirical_ntk = nt.batch(
            nt.empirical_ntk_fn(
                f=self._ntk_apply_fn,
                trace_axes=trace_axes,
                vmap_axes=0,
                implementation=nt.NtkImplementation.AUTO,
            ),
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

    def _ntk_apply_fn(self, params, x: np.ndarray):
        """
        Return an NTK capable apply function.

        Parameters
        ----------
        params : dict
                Network parameters to use in the calculation.
        x : np.ndarray
                Data on which to apply the network

        Returns
        -------
        Acts on the data with the model architecture and parameter set.
        """
        # train=False
        return self.model.apply({"params": params}, x, mutable=["batch_stats"])[0]

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
        if kernel_init:
            self.model.kernel_init = kernel_init
        if bias_init:
            self.model.bias_init = bias_init

        params = self.model.init(self.rng(), np.ones(list(self.input_shape)))["params"]

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
        return self.apply_fn({"params": params}, inputs)

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
            raise NotImplementedError("Infinite NTK is not available for Flax models.")
        else:
            infinite_ntk = None

        if normalize:
            empirical_ntk = normalize_covariance_matrix(empirical_ntk)
            if infinite:
                infinite_ntk = None

        return {"empirical": empirical_ntk, "infinite": infinite_ntk}
