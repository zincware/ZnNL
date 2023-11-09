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

import logging
from typing import Callable, List, Sequence, Union

import jax
import jax.numpy as np
from flax import linen as nn

from znnl.models.jax_model import JaxModel

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
        optimizer: Callable,
        input_shape: tuple,
        batch_size: int = 10,
        layer_stack: List[nn.Module] = None,
        flax_module: nn.Module = None,
        trace_axes: Union[int, Sequence[int]] = (-1,),
        store_on_device: bool = True,
        seed: int = None,
    ):
        """
        Construct a Flax model.

        Parameters
        ----------
        layer_stack : List[nn.Module]
                A list of flax modules to be used in the call method.
        optimizer : Callable
                optimizer to use in the training. OpTax is used by default and
                cross-compatibility is not assured.
        input_shape : tuple
                Shape of the NN input.
        batch_size : int
                Size of batch to use in the NTk calculation.
        flax_module : nn.Module
                Flax module to use instead of building one from scratch here.
        trace_axes : Union[int, Sequence[int]]
                Tracing over axes of the NTK.
                The default value is trace_axes(-1,), which reduces the NTK to a tensor
                of rank 2.
                For a full NTK set trace_axes=().
        store_on_device : bool, default True
                Whether to store the NTK on the device or not.
                This should be set False for large NTKs that do not fit in GPU memory.
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
            optimizer=optimizer,
            input_shape=input_shape,
            seed=seed,
            trace_axes=trace_axes,
            ntk_batch_size=batch_size,
            store_on_device=store_on_device,
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

    def _ntk_apply_fn(self, params, inputs: np.ndarray):
        """
        Return an NTK capable apply function.

        Parameters
        ----------
        params: dict
                Contains the model parameters to use for the model computation.
                It is a dictionary of structure
                {'params': params, 'batch_stats': batch_stats}
        inputs : np.ndarray
                Feature vector on which to apply the model.

        TODO(Konsti): Make the apply function work with the batch_stats.

        Returns
        -------
        Acts on the data with the model architecture and parameter set.
        """
        return self.model.apply(
            {"params": params["params"]}, inputs, mutable=["batch_stats"]
        )[0]

    def train_apply_fn(self, params: dict, inputs: np.ndarray):
        """
        Apply function used for training the model.

        This is the function that is used to apply the model to the data in the training
        loop. It is defined for each child class indivudally and is used to create the
        train state.

        Parameters
        ----------
        params: dict
                Contains the model parameters to use for the model computation.
        inputs : np.ndarray
                Feature vector on which to apply the model.

        TODO(Konsti): Make the apply function work with the batch_stats.

        Returns
        -------
        Output of the model.
        """
        return self.apply_fn({"params": params["params"]}, inputs)

    def apply(self, params: dict, inputs: np.ndarray):
        """
        Apply the model to a feature vector.

        Parameters
        ----------
        params: dict
                Contains the model parameters to use for the model computation.
                It is a dictionary of structure
                {'params': params, 'batch_stats': batch_stats}
        inputs : np.ndarray
                Feature vector on which to apply the model.

        TODO(Konsti): Make the apply function work with the batch_stats.

        Returns
        -------
        Output of the model.
        """
        return self.apply_fn({"params": params["params"]}, inputs)
