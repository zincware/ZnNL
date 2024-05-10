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

from typing import Any, Callable, Optional, Sequence, Union

import jax
import jax.numpy as np
import jax.random
import optax
from flax.training.train_state import TrainState
from transformers import FlaxPreTrainedModel

from znnl.ntk_computation.jax_ntk import JAXNTKComputation
from znnl.optimizers.trace_optimizer import TraceOptimizer
from znnl.utils.prng import PRNGKey


class TrainState(TrainState):
    """
    Train state for a Jax model.

    Parameters
    ----------
    apply_fn : Callable
            Function to apply the model.
    params : dict
            Parameters of the model.
    tx : Callable
            Optimizer to use.
    batch_stats : Any
            Batch statistics of the model.
            This only set if batch_stats=True in a JaxModel.
    """

    batch_stats: Any = None
    use_batch_stats: bool = False


class JaxModel:
    """
    Parent class for Jax-based models.
    """

    def __init__(
        self,
        optimizer: Union[Callable, TraceOptimizer],
        input_shape: Optional[tuple] = None,
        seed: Optional[int] = None,
        pre_built_model: Union[None, FlaxPreTrainedModel] = None,
    ):
        """
        Construct a znrnd model.
        Parameters
        ----------
        optimizer : Callable
                optimizer to use in the training. OpTax is used by default and
                cross-compatibility is not assured.
        input_shape : Optional[tuple]
                Shape of the NN input. Required if no pre-built model is passed.
        seed : int, default None
                Random seed for the RNG. Uses a random int if not specified.
        pre_built_model : Union[None, FlaxPreTrainedModel] (default = None)
                Pre-built model to use instead of building one from scratch here.
                So far, this is only implemented for Hugging Face flax models.
        """

        self.optimizer = optimizer
        self.input_shape = input_shape

        # Initialized in self.init_model
        self.rng = None

        # Input shape is required if no full model is passed.
        if pre_built_model is None and input_shape is None:
            raise ValueError(
                "Input shape must be specified if no pre-built model is passed."
                "Model has not been constructed."
            )

        # initialize the model state
        if pre_built_model is None:
            self.init_model(seed=seed)
        else:
            self.model_state = self._create_train_state(params=pre_built_model.params)

        self.apply_jit = jax.jit(self.apply)

    def init_model(
        self,
        seed: int = None,
        kernel_init: Callable = None,
        bias_init: Callable = None,
    ):
        """
        Initialize a model.

        Parameters
        ----------
        seed : int, default None
                Random seed for the RNG. Uses a random int if not specified.
        kernel_init : Callable
                Define the kernel initialization.
        bias_init : Callable
                Define the bias initialization.
        """
        self.rng = PRNGKey(seed)
        params = self._init_params(kernel_init, bias_init)
        self.model_state = self._create_train_state(params)

    def _create_train_state(self, params: dict) -> TrainState:
        """
        Create a training state of the model.

        Returns
        -------
        initial state of model to then be trained.
        Notes
        -----
        TODO: Make the TrainState class passable by the user as it can track custom
              model properties.
        """
        # Set dummy optimizer for case of trace optimizer.
        if isinstance(self.optimizer, TraceOptimizer):
            optimizer = optax.sgd(1.0)
        else:
            optimizer = self.optimizer

        # Create the train state taking into account the batch statistics.
        if "batch_stats" in params:
            train_state = TrainState.create(
                apply_fn=self.train_apply_fn,
                params=params["params"],
                batch_stats=params["batch_stats"],
                use_batch_stats=True,
                tx=optimizer,
            )
        else:
            train_state = TrainState.create(
                apply_fn=self.train_apply_fn,
                params=params,
                tx=optimizer,
            )
        return train_state

    def ntk_apply_fn(self, params: dict, inputs: np.ndarray):
        """
        Apply function used in the NTK computation.

        Parameters
        ----------
        params: dict
                Contains the model parameters to use for the model computation.
                It is a dictionary of structure
                {'params': params, 'batch_stats': batch_stats}
        inputs : np.ndarray
                Feature vector on which to apply the model.

        Returns
        -------
        The apply function used in the NTK computation.
        """
        raise NotImplementedError("Implemented in child class")

    def train_apply_fn(self, params: dict, inputs: np.ndarray):
        """
        Apply function used for training the model.

        It is defined in the child class and used to create the train state.
        this method is used to apply the model to the data in the training loop.

        Parameters
        ----------
        params: dict
                Contains the model parameters to use for the model computation.
                It is a dictionary of structure
                {'params': params, 'batch_stats': batch_stats}
        inputs : np.ndarray
                Feature vector on which to apply the model.
        """
        raise NotImplementedError("Implemented in child class")


    def __call__(self, feature_vector: np.ndarray):
        """
        Call the network.

        Parameters
        ----------
        feature_vector : np.ndarray
                Feature vector on which to apply operation.

        Returns
        -------
        output of the model.
        """
        return self.apply_jit(
            {
                "params": self.model_state.params,
                "batch_stats": self.model_state.batch_stats,
            },
            feature_vector,
        )
