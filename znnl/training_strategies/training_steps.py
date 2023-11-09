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
from jax import jit, lax

logger = logging.getLogger(__name__)

# Implement the batch stats train step here


def train_step(
    state: TrainState, batch: dict, loss_fn: Callable, compute_metrics_fn: Callable
):
    """
    Train a single step.

    This method is a wrapper around the vanilla_step and batch_stat_step methods.
    It decides which method to call based on the use_batch_stats flag in the TrainState.

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
    if state.use_batch_stats:
        state, metrics = batch_stat_step(state, batch, loss_fn, compute_metrics_fn)
    else:
        state, metrics = vanilla_step(state, batch, loss_fn, compute_metrics_fn)
    return state, metrics


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

    (_, (predictions, batch_stats)), grads = grad_fn(state.params)

    state = state.apply_gradients(grads=grads)  # in place state update.
    state = state.replace(batch_stats=batch_stats["batch_stats"])  # update batch stats

    metrics = compute_metrics_fn(predictions=predictions, targets=batch["targets"])

    return state, metrics
