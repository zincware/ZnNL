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
from typing import Callable

import jax
import jax.numpy as np
import neural_tangents as nt
import numpy as onp
import optax
from flax.training import train_state
from neural_tangents.stax import serial
from tqdm import trange

from znrnd.core.models.model import Model
from znrnd.core.utils.matrix_utils import normalize_covariance_matrix

logger = logging.getLogger(__name__)


class TrainState:
    """
    Train state for NTK models
    """

    def __init__(self, apply_fn, params, tx: optax.GradientTransformation):
        """
        Constructor of the train state.

        Parameters
        ----------
        apply_fn
        params
        tx
        """
        self.apply_fn = apply_fn
        self.params = params
        self.optimizer = tx
        self.opt_state = tx.init()

    def apply_gradients(self, gradients):
        """
        Apply the gradients to parameters and update the state.

        Parameters
        ----------
        gradients

        Returns
        -------

        """

        updates, self.opt_state = self.optimizer.update(
            gradients, self.opt_state, self.params
        )
        self.params = optax.apply_updates(self.params, updates)


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
        nt_module: serial = None,
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

        Notes
        -----
        Kernel operations are batched to a fixed size of 10, reduce if needed.
        This was selected to safely compute NTK on MNIST.
        """
        self.init_fn = nt_module[0]
        self.apply_fn = nt_module[1]
        self.kernel_fn = nt.batch(nt_module[2], batch_size=5)
        self.empirical_ntk = nt.batch(nt.empirical_ntk_fn(self.apply_fn), batch_size=5)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.input_shape = input_shape
        self.training_threshold = training_threshold

        # initialize the model state
        init_rng = jax.random.PRNGKey(onp.random.randint(0, 500))
        self.model_state = self._create_train_state(init_rng)

    def compute_ntk(
        self, x_i: np.ndarray, x_j: np.ndarray = None, normalize: bool = True
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

        Returns
        -------
        NTK : dict
                The NTK matrix for both the empirical and infinite width computation.
        """
        if x_j is None:
            x_j = x_i

        empirical_ntk = self.empirical_ntk(x_i, x_j, self.model_state.params)
        infinite_ntk = self.kernel_fn(x_i, x_j, "ntk")

        if normalize:
            empirical_ntk = normalize_covariance_matrix(empirical_ntk)
            infinite_ntk = normalize_covariance_matrix(infinite_ntk)

        return {"empirical": empirical_ntk, "infinite": infinite_ntk}

    def _create_train_state(self, init_rng: int):
        """
        Create a training state of the model.

        Parameters
        ----------
        init_rng : int
                Initial rng for train state that is immediately deleted.

        Returns
        -------
        initial state of model to then be trained.

        Notes
        -----
        TODO: Make the TrainState class passable by the user as it can track custom
              model properties.
        """
        _, params = self.init_fn(init_rng, self.input_shape)

        return train_state.TrainState.create(
            apply_fn=self.apply_fn, params=params, tx=self.optimizer
        )

    def _compute_metrics(self, predictions: np.ndarray, targets: np.ndarray):
        """
        Compute the current metrics of the training.

        Parameters
        ----------
        predictions : np.ndarray
                Predictions made by the network.
        targets : np.ndarray
                Targets from the training data.

        Returns
        -------
        metrics : dict
                A dict of current training metrics, e.g. {"loss": ..., "accuracy": ...}
        """
        loss = self.loss_fn(predictions, targets)
        accuracy = np.mean(np.argmax(predictions, -1) == targets)

        metrics = {"loss": loss, "accuracy": accuracy}

        return metrics

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
            predictions = self.apply_fn(params, batch["inputs"])
            loss = self.loss_fn(predictions, batch["targets"])
            return loss, predictions

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

        (_, predictions), grads = grad_fn(state.params)

        state = state.apply_gradients(grads=grads)
        metrics = self._compute_metrics(
            predictions=predictions, targets=batch["targets"]
        )

        return state, metrics

    def _evaluate_step(self, params: dict, batch: dict):
        """
        Evaluate the model on test data.

        Parameters
        ----------
        state : dict
                Current state of the neural network.
        batch : dict
                Batch of data to test on.

        Returns
        -------
        metrics : dict
                Metrics dict computed on test data.
        """
        predictions = self.apply_fn(params, batch["inputs"])

        return self._compute_metrics(predictions, batch["targets"])

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
            predictions = self.apply_fn(params, batch["inputs"])
            loss = self.loss_fn(predictions, batch["targets"])

            return loss, predictions

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, predictions), grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        metrics = self._compute_metrics(
            predictions=predictions, targets=batch["targets"]
        )

        return state, metrics

    def _evaluate_step(self, params: dict, batch: dict):
        """
        Evaluate the model on test data.

        Parameters
        ----------
        state : dict
                Current state of the neural network.
        batch : dict
                Batch of data to test on.

        Returns
        -------
        metrics : dict
                Metrics dict computed on test data.
        """
        predictions = self.apply_fn(params, batch["inputs"])

        return self._compute_metrics(predictions, batch["targets"])

    def _train_epoch(self, state: dict, train_ds: dict, batch_size: int):
        """
        Train for a single epoch.

        Performs the following steps:

        * Shuffles the data
        * Runs an optimization step on each batch
        * Computes the metrics for the batch
        * Return an updated optimizer, state, and metrics dictionary.

        Parameters
        ----------
        state : dict
                Current state of the model.
        train_ds : dict
                Dataset on which to train.
        batch_size : int
                Size of each batch.

        Returns
        -------
        state : dict
                State of the model after the epoch.
        metrics : dict
                Dict of metrics for current state.
        """
        # Some housekeeping variables.
        train_ds_size = len(train_ds["inputs"])
        steps_per_epoch = train_ds_size // batch_size

        if train_ds_size == 1:
            state, metrics = self._train_step(state, train_ds)
            batch_metrics = [metrics]

        else:
            # Prepare the shuffle.
            permutations = jax.random.permutation(self.rng, train_ds_size)
            permutations = permutations[: steps_per_epoch * batch_size]
            permutations = permutations.reshape((steps_per_epoch, batch_size))

            # Step over items in batch.
            batch_metrics = []
            for permutation in permutations:
                batch = {k: v[permutation, ...] for k, v in train_ds.items()}
                state, metrics = self._train_step(state, batch)
                batch_metrics.append(metrics)

        # Get the metrics off device for printing.
        batch_metrics_np = jax.device_get(batch_metrics)
        epoch_metrics_np = {
            k: onp.mean([metrics[k] for metrics in batch_metrics_np])
            for k in batch_metrics_np[0]
        }

        return state, epoch_metrics_np

    def _evaluate_model(self, params: dict, test_ds: dict):
        """
        Evaluate the model.

        Parameters
        ----------
        params : dict
                Current state of the model.
        test_ds : dict
                Dataset on which to evaluate.
        Returns
        -------
        loss : float
                Loss of the model.
        """
        metrics = self._evaluate_step(params, test_ds)
        metrics = jax.device_get(metrics)
        summary = jax.tree_map(lambda x: x.item(), metrics)

        return summary

    def train_model(
        self,
        train_ds: dict,
        test_ds: dict,
        epochs: int = 50,
        batch_size: int = 1,
    ):
        """
        Train the model.

        See the parent class for a full doc-string.
        """
        if self.model_state is None:
            init_rng = jax.random.PRNGKey(onp.random.randint(0, 500))
            state = self._create_train_state(init_rng)
            self.model_state = state
        else:
            state = self.model_state

        loading_bar = trange(1, epochs + 1, ncols=100, unit="batch")
        for i in loading_bar:
            loading_bar.set_description(f"Epoch: {i}")

            state, train_metrics = self._train_epoch(
                state, train_ds, batch_size=batch_size
            )
            test_loss = self._evaluate_model(state.params, test_ds)

            loading_bar.set_postfix(
                test_loss=test_loss["loss"], accuracy=test_loss["accuracy"]
            )

        # Update the final model state.
        self.model_state = state

        return test_loss

    def train_model_recursively(
        self, train_ds: dict, test_ds: dict, epochs: int = 100, batch_size: int = 1
    ):
        """
        Check parent class for full doc string.
        """
        if self.model_state is None:
            init_rng = jax.random.PRNGKey(onp.random.randint(0, 500))
            state = self._create_train_state(init_rng)
            self.model_state = state
        else:
            state = self.model_state

        condition = False
        counter = 0
        while not condition:
            loading_bar = trange(1, epochs + 1, ncols=100, unit="batch")
            for i in loading_bar:
                loading_bar.set_description(f"Epoch: {i}")

                state, train_metrics = self._train_epoch(
                    state, train_ds, batch_size=batch_size
                )
                test_loss = self._evaluate_model(state.params, test_ds)

                loading_bar.set_postfix(test_loss=test_loss)

            # Update the final model state.
            self.model_state = state

            # Perform checks and update parameters
            counter += 1
            epochs = int(1.1 * epochs)
            if test_loss <= self.training_threshold:
                condition = True

            # Re-initialize the network if it is simply not converging.
            if counter % 10 == 0:
                logger.info("Model training stagnating, re-initializing model.")
                init_rng = jax.random.PRNGKey(onp.random.randint(0, 500))
                state = self._create_train_state(init_rng)
                self.model_state = state

    def __call__(self, feature_vector: np.ndarray):
        """
        See parent class for full doc string.
        """
        state = self.model_state

        return self.apply_fn(state.params, feature_vector)
