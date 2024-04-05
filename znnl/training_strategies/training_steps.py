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
from functools import partial
from typing import Callable

import jax.random
from flax.training.train_state import TrainState
from jax import jit

logger = logging.getLogger(__name__)


class TrainStep:
    """
    Train a single step.

    This class is used to train a model for a single step.
    Performing a step is done by calling the class.
    It serves the purpose of allowing for models with and without batch statistics.
    """

    def __init__(self, state) -> None:
        """
        Constructor of the TrainStep class.

        Utilizes the train state to determine whether batch statistics are used.

        Parameters
        ----------
        state : TrainState
                Current state of the neural network.
        """
        self.use_batch_stats = state.use_batch_stats
        if self.use_batch_stats:
            self.train_step = self.batch_stat_step
        else:
            self.train_step = self.vanilla_step

    @staticmethod
    @partial(
        jit,
        static_argnums=(
            2,
            3,
        ),
    )
    def vanilla_step(
        state: TrainState, batch: dict, loss_fn: Callable, compute_metrics_fn: Callable
    ):
        """
        Train a single step.

        Parameters
        ----------
        state : TrainState
                Current state of the neural network.
        batch : dict
                Batch of data to train on.
        loss_fn : Callable
                Loss function to use for the training.
        compute_metrics_fn : Callable
                Metric to evaluate the training on.

        Returns
        -------
        state : dict
                Updated state of the neural network.
        metrics : dict
                Metrics for the current model.
        """

        def loss_fn_helper(params):
            """
            helper loss computation
            """
            inner_predictions = state.apply_fn({"params": params}, batch["inputs"])
            loss = loss_fn(inner_predictions, batch["targets"])
            return loss, inner_predictions

        grad_fn = jax.value_and_grad(loss_fn_helper, has_aux=True)
        (_, predictions), grads = grad_fn(state.params)

        state = state.apply_gradients(grads=grads)  # in place state update.
        metrics = compute_metrics_fn(predictions=predictions, targets=batch["targets"])

        return state, metrics

    @staticmethod
    @partial(
        jit,
        static_argnums=(
            2,
            3,
        ),
    )
    def batch_stat_step(
        state: TrainState, batch: dict, loss_fn: Callable, compute_metrics_fn: Callable
    ):
        """
        Train a single step.

        Parameters
        ----------
        state : TrainState
                Current state of the neural network.
        batch : dict
                Batch of data to train on.

        Returns
        -------
        state : dict
                Updated state of the neural network.
        metrics : dict
                Metrics for the current model.
        """

        def loss_fn_helper(params):
            """
            helper loss computation
            """
            inner_predictions, batch_stats = state.apply_fn(
                params={"params": params, "batch_stats": state.batch_stats},
                inputs=batch["inputs"],
            )
            loss = loss_fn(inner_predictions, batch["targets"])
            return loss, (inner_predictions, batch_stats)

        grad_fn = jax.value_and_grad(loss_fn_helper, has_aux=True)

        (_, (predictions, updates)), grads = grad_fn(state.params)

        state = state.apply_gradients(grads=grads, batch_stats=updates["batch_stats"])

        metrics = compute_metrics_fn(predictions=predictions, targets=batch["targets"])

        return state, metrics

    def __call__(
        self,
        state: TrainState,
        batch: dict,
        loss_fn: Callable,
        compute_metrics_fn: Callable,
    ):
        """
        Train a single step.

        This method calls the train_step method.

        Parameters
        ----------
        state : TrainState
                Current state of the neural network.
        batch : dict
                Batch of data to train on.


        Returns
        -------
        state : dict
                Updated state of the neural network.
        metrics : dict
                Metrics for the current model.
        """
        return self.train_step(state, batch, loss_fn, compute_metrics_fn)
