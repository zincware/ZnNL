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
from typing import Callable, Union

import jax
import jax.numpy as np
from neural_tangents.stax import serial

from znnl.models.jax_model import JaxModel
from znnl.optimizers.trace_optimizer import TraceOptimizer

logger = logging.getLogger(__name__)


class NTModel(JaxModel):
    """
    Class for a neural tangents model.
    """

    def __init__(
        self,
        optimizer: Union[Callable, TraceOptimizer],
        input_shape: tuple,
        nt_module: serial = None,
        seed: int = None,
    ):
        """
        Constructor for a Flax model.

        Parameters
        ----------
        optimizer : Callable
                optimizer to use in the training. OpTax is used by default and
                cross-compatibility is not assured.
        input_shape : tuple
                Shape of the NN input.
        nt_module : serial
                NT model used.
        seed : int, default None
                Random seed for the RNG. Uses a random int if not specified.
        """
        self.init_fn = nt_module[0]
        self.apply_fn = jax.jit(nt_module[1])

        # Save input parameters, call self.init_model
        super().__init__(
            optimizer=optimizer,
            input_shape=input_shape,
            seed=seed,
        )

    def _init_params(self, kernel_init: Callable = None, bias_init: Callable = None):
        """
        Initialize a state for the model parameters.

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
        """
        Apply the model to a feature vector.

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
        return self.apply_fn(params["params"], inputs)

    def train_apply_fn(self, params: dict, inputs: np.ndarray):
        """
        Apply function used for training the model.

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
        return self.apply_fn(params["params"], inputs)

    def ntk_apply_fn(self, params: dict, inputs: np.ndarray):
        """
        NTK Apply function for the neural_tangents module.

        Parameters
        ----------
        params: dict
                Contains the model parameters to use for the model computation.
        inputs : np.ndarray
                Feature vector on which to apply the model.

        Returns
        -------
        The apply function used in the NTK computation.
        """
        return self.apply_fn(params["params"], inputs)
