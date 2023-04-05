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
from dataclasses import dataclass
from typing import Callable

import jax.numpy as np
import optax
from flax.training.train_state import TrainState


@dataclass
class TraceOptimizer:
    """
    Class implementation of the trace optimizer

    Attributes
    ----------
    scale_factor : float
            Scale factor to apply to the optimizer.
    rescale_interval : int
            Number of epochs to wait before re-scaling the learning rate.
    """

    scale_factor: float
    rescale_interval: float = 1

    @optax.inject_hyperparams
    def optimizer(self, learning_rate):
        return optax.sgd(learning_rate)

    def apply_optimizer(
        self,
        model_state: TrainState,
        data_set: np.ndarray,
        ntk_fn: Callable,
        epoch: int,
    ):
        """
        Apply the optimizer to a model state.

        Parameters
        ----------
        model_state : TrainState
                Current state of the model
        data_set : jnp.ndarray
                Data-set to use in the computation.
        ntk_fn : Callable
                Function to use for the NTK computation
        epoch : int
                Current epoch

        Returns
        -------
        new_state : TrainState
                New state of the model
        """
        eps = 1e-8
        # Check if the update should be performed.
        if epoch % self.rescale_interval == 0:
            # Compute the ntk trace.
            ntk = ntk_fn(data_set)["empirical"]
            trace = np.trace(ntk)

            # Create the new optimizer.
            new_optimizer = self.optimizer(self.scale_factor / (trace + eps))

            # Create the new state
            new_state = TrainState.create(
                apply_fn=model_state.apply_fn,
                params=model_state.params,
                tx=new_optimizer,
            )
        else:
            # If no update is needed, return the old state.
            new_state = model_state

        return new_state
