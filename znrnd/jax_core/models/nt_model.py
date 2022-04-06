"""
ZnRND: A zincwarecode package.

License
-------
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the zincwarecode Project.

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
Module for the neural tangents infinite width network models.
"""
import logging
from typing import Callable, List

import jax
import jax.numpy as np
import numpy as onp
from tqdm import trange
from neural_tangents.stax import serial

from znrnd.jax_core.models.model import Model

logger = logging.getLogger(__name__)


class TrainState:
    """
    Implementation of the Flax TrainState class for the neural tangent models.
    """
    params: list
    optimizer: callable



class NTModel(Model):
    """
    Class for a neural tangents model.
    """
    rng = jax.random.PRNGKey(onp.random.randint(0, 500))

    def __init__(
            self,
            loss_fn: Callable,
            optimizer: Callable,
            input_shape: tuple,
            training_threshold: float,
            nt_module: serial = None
    ):
        """
        Constructor for a Flax model.

        Parameters
        ----------
        loss_fn : Callable
                A function to use in the loss computation.
        optimizer : Callable
                optimizer to use in the training. OpTax is used by default and
                cross-compatibility is not assured.
        input_shape : tuple
                Shape of the NN input.
        training_threshold : float
                The loss value at which point you consider the model trained.
        """
        self.init_fn = nt_module[0]
        self.apply_fn = nt_module[1]
        self.kernel_fn = nt_module[2]
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.input_shape = input_shape
        self.training_threshold = training_threshold

        # initialize the model state
        init_rng = jax.random.PRNGKey(onp.random.randint(0, 500))
        self.params = self.init_fn(init_rng, input_shape=input_shape)

    def _train_step(self, state: dict, batch: dict):
        """
        Train a single step.

        Parameters
        ----------
        state : dict
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

        def loss_fn(params):
            """
            helper loss computation
            """
            predictions = self.model.apply({"params": params}, batch["inputs"])
            loss = self.loss_fn(predictions, batch["targets"])

            return loss, predictions

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

        (_, predictions), grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        metrics = self._compute_metrics(
            predictions=predictions, targets=batch["targets"]
        )

        return state, metrics

