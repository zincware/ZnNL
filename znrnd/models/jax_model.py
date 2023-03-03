"""
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincware Project.

Description: Parent class for the Jax-based models.
"""
from typing import TYPE_CHECKING, Callable, List, Sequence, Tuple, Union

import jax
import jax.numpy as np
import jax.random
import neural_tangents as nt
import optax
from flax.training.train_state import TrainState

from znrnd.optimizers.trace_optimizer import TraceOptimizer
from znrnd.utils.prng import PRNGKey


class JaxModel:
    """
    Parent class for Jax-based models.
    """

    def __init__(
        self,
        optimizer: Union[Callable, TraceOptimizer],
        input_shape: tuple,
        seed: int = None,
        ntk_batch_size: int = 10,
        trace_axes: Union[int, Sequence[int]] = (-1,),
    ):
        """
        Construct a znrnd model.

        Parameters
        ----------
        optimizer : Callable
                optimizer to use in the training. OpTax is used by default and
                cross-compatibility is not assured.
        input_shape : tuple
                Shape of the NN input.
        seed : int, default None
                Random seed for the RNG. Uses a random int if not specified.
        ntk_batch_size : int, default 10
                Batch size to use in the NTK computation.
        trace_axes : Union[int, Sequence[int]]
                Tracing over axes of the NTK.
                The default value is trace_axes(-1,), which reduces the NTK to a tensor
                of rank 2.
                For a full NTK set trace_axes=().
        """
        self.optimizer = optimizer
        self.input_shape = input_shape

        # Initialized in self.init_model
        self.rng = None

        # initialize the model state
        self.model_state = None
        self.init_model(seed)

        # Prepare NTK calculation
        self.empirical_ntk = nt.batch(
            nt.empirical_ntk_fn(f=self._ntk_apply_fn, trace_axes=trace_axes),
            batch_size=ntk_batch_size,
        )
        self.empirical_ntk_jit = jax.jit(self.empirical_ntk)

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
        self.model_state = self._create_train_state(kernel_init, bias_init)

    def _create_train_state(
        self, kernel_init: Callable = None, bias_init: Callable = None
    ) -> TrainState:
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
        params = self._init_params(kernel_init, bias_init)

        # Set dummy optimizer for case of trace optimizer.
        if isinstance(self.optimizer, TraceOptimizer):
            optimizer = optax.sgd(1.0)
        else:
            optimizer = self.optimizer

        return TrainState.create(apply_fn=self.apply_fn, params=params, tx=optimizer)

    def _ntk_apply_fn(self, params: dict, inputs: np.ndarray):
        """
        Apply function used in the NTK computation.

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
        empirical_ntk = self.empirical_ntk_jit(x_i, x_j, self.model_state.params)

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
        return self.apply(self.model_state.params, feature_vector)
