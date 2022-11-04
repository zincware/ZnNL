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
from typing import Callable, List

import jax
import jax.numpy as np
from flax import linen as nn

from znrnd.accuracy_functions.accuracy_function import AccuracyFunction
from znrnd.models.model import Model

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


class FlaxModel(Model):
    """
    Class for the Flax model in ZnRND.
    """

    def __init__(
        self,
        loss_fn: Callable,
        optimizer: Callable,
        input_shape: tuple,
        training_threshold: float,
        layer_stack: List[nn.Module] = None,
        flax_module: nn.Module = None,
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
        flax_module : nn.Module
                Flax module to use instead of building one from scratch here.
        compute_accuracy : bool, default False
                If true, an accuracy computation will be performed. Only valid for
                classification tasks.
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
        raise NotImplementedError("Not yet available.")
