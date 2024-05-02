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

from functools import partial
from typing import Any, Callable, Optional, Sequence, Union

import jax
import jax.numpy as np
import jax.random
import neural_tangents as nt
import optax
from flax.training.train_state import TrainState
from transformers import FlaxPreTrainedModel

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
        ntk_batch_size: int = 10,
        trace_axes: Union[int, Sequence[int]] = (),
        store_on_device: bool = True,
        pre_built_model: Union[None, FlaxPreTrainedModel] = None,
        ntk_implementation: Union[None, nt.NtkImplementation] = None,
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
        ntk_batch_size : Optional[int], default 10
                Batch size to use in the NTK computation.
        trace_axes : Union[int, Sequence[int]]
                Tracing over axes of the NTK.
                The default value is trace_axes=(), providing the full NTK of rank 4.
                For a traced NTK set trace_axes=(-1,), which reduces the NTK to a
                tensor of rank 2.
        store_on_device : bool, default True
                Whether to store the NTK on the device or not.
                This should be set False for large NTKs that do not fit in GPU memory.
        pre_built_model : Union[None, FlaxPreTrainedModel] (default = None)
                Pre-built model to use instead of building one from scratch here.
                So far, this is only implemented for Hugging Face flax models.
        ntk_implementation : Union[None, nt.NtkImplementation] (default = None)
                Implementation of the NTK computation.
                The implementation depends on the trace_axes and the model
                architecture. The default does automatically take into account the
                trace_axes. For trace_axes=() the default is NTK_VECTOR_PRODUCTS,
                for all other cases including trace_axes=(-1,) the default is
                JACOBIAN_CONTRACTION. For more specific use cases, the user can
                set the implementation manually.
                Information about the implementation and specific requirements can be
                found in the neural_tangents documentation.
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

        # Prepare NTK calculation
        if not ntk_implementation:
            if trace_axes == ():
                ntk_implementation = nt.NtkImplementation.NTK_VECTOR_PRODUCTS
            else:
                ntk_implementation = nt.NtkImplementation.JACOBIAN_CONTRACTION
        self.empirical_ntk = nt.batch(
            nt.empirical_ntk_fn(
                f=self._ntk_apply_fn,
                trace_axes=trace_axes,
                implementation=ntk_implementation,
            ),
            batch_size=ntk_batch_size,
            store_on_device=store_on_device,
        )
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

    def _ntk_apply_fn(self, params: dict, inputs: np.ndarray):
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

    def compute_ntk(
        self,
        x_i: np.ndarray,
        x_j: np.ndarray = None,
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
        infinite : bool (default = False)
                If true, compute the infinite width limit as well.

        Returns
        -------
        NTK : dict
                The NTK matrix for both the empirical and infinite width computation.
        """
        if x_j is None:
            x_j = x_i
        empirical_ntk = self.empirical_ntk(
            x_i,
            x_j,
            {
                "params": self.model_state.params,
                "batch_stats": self.model_state.batch_stats,
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
