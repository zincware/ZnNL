"""
ZnRND: A Zincwarecode package.

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
Module for the use of a Flax model with ZnRND.
"""
import logging
from typing import Callable, List, Tuple

import jax
import jax.numpy as np
import numpy as onp
from flax import linen as nn
from flax.training import train_state
from tqdm import trange

from znrnd.accuracy_functions.accuracy_function import AccuracyFunction
from znrnd.models.model import Model
from znrnd.utils.prng import PRNGKey

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


class FlaxModel(Model):
    """
    Class for the Flax model in ZnRND.
    """

    model: nn.Module
    model_state: train_state.TrainState = None

    def __init__(
        self,
        loss_fn: Callable,
        optimizer: Callable,
        input_shape: tuple,
        training_threshold: float,
        layer_stack: List[nn.Module] = None,
        flax_module: nn.Module = None,
        accuracy_fn: AccuracyFunction = None,
        seed: int = None,
    ):
        """
        Constructor for a Flax model.

        Parameters
        ----------
        layer_stack : List[nn.Module]
                A list of flax modules to be used in the call method.
        loss_fn : Callable
                A function to use in the loss computation.
        optimizer : Callable
                optimizer to use in the training. OpTax is used by default and
                cross-compatibility is not assured.
        input_shape : tuple
                Shape of the NN input.
        training_threshold : float
                The loss value at which point you consider the model trained.
        flax_module : nn.Module
                Flax module to use instead of building one from scratch here.
        compute_accuracy : bool, default False
                If true, an accuracy computation will be performed. Only valid for
                classification tasks.
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

        # Initialized in self.init_model
        self.rng = None
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.input_shape = input_shape
        self.training_threshold = training_threshold
        self.model_state = None

        self.apply_fn = jax.jit(self.model.apply)

        # initialize the model state
        self.init_model(seed)

        self.accuracy_fn = accuracy_fn

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

    def compute_ntk(
        self,
        x_i: np.ndarray,
        x_j: np.ndarray = None,
        normalize: bool = True,
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
        normalize : bool (default = True)
                If true, divide each row by its max value.
        infinite : bool (default = False)
                If true, compute the infinite width limit as well.

        Returns
        -------
        NTK : dict
                The NTK matrix for both the empirical and infinite width computation.
        """
        raise NotImplementedError("Not yet available.")

    def _compute_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
    ):
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
        if self.accuracy_fn is not None:
            accuracy = self.accuracy_fn(predictions, targets)
            metrics = {"loss": loss, "accuracy": accuracy}

        else:
            metrics = {"loss": loss}

        return metrics

    def _create_train_state(
        self,
        kernel_init: Callable = None,
        bias_init: Callable = None,
    ):
        """
        Create a training state of the model.

        Parameters
        ----------
        kernel_init : Callable
                Define the kernel initialization.
        bias_init : Callable
                Define the bias initialization.

        Returns
        -------
        initial state of model to then be trained.
        """
        if kernel_init:
            self.model.kernel_init = kernel_init
        if bias_init:
            self.model.bias_init = bias_init

        params = self.model.init(self.rng(), np.ones(list(self.input_shape)))["params"]

        return train_state.TrainState.create(
            apply_fn=self.apply_fn, params=params, tx=self.optimizer
        )

    def _train_step(self, state: train_state.TrainState, batch: dict):
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
            inner_predictions = self.apply_fn({"params": params}, batch["inputs"])
            loss = self.loss_fn(inner_predictions, batch["targets"])

            return loss, inner_predictions

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
        params : dict
                Current parameters of the neural network.
        batch : dict
                Batch of data to test on.

        Returns
        -------
        metrics : dict
                Metrics dict computed on test data.
        """
        predictions = self.apply_fn({"params": params}, batch["inputs"])

        return self._compute_metrics(predictions, batch["targets"])

    def _train_epoch(
        self, state: train_state.TrainState, train_ds: dict, batch_size: int
    ) -> Tuple[train_state.TrainState, dict]:
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
        state : train_state.TrainState
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
            permutations = jax.random.permutation(self.rng(), train_ds_size)
            permutations = np.array_split(permutations, steps_per_epoch)

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

    def _evaluate_model(self, params: dict, test_ds: dict) -> dict:
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
        loss : dict
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
        disable_loading_bar: bool = False,
    ):
        """
        Train the model.

        See the parent class for a full doc-string.
        """
        if self.model_state is None:
            self.init_model()

        state = self.model_state

        loading_bar = trange(
            1, epochs + 1, ncols=100, unit="batch", disable=disable_loading_bar
        )
        test_losses = []
        test_accuracy = []
        train_losses = []
        train_accuracy = []
        for i in loading_bar:
            loading_bar.set_description(f"Epoch: {i}")

            state, train_metrics = self._train_epoch(
                state, train_ds, batch_size=batch_size
            )
            metrics = self._evaluate_model(state.params, test_ds)

            loading_bar.set_postfix(test_loss=metrics["loss"])
            if self.accuracy_fn is not None:
                loading_bar.set_postfix(accuracy=metrics["accuracy"])
                test_accuracy.append(metrics["accuracy"])
                train_accuracy.append(train_metrics["accuracy"])

            test_losses.append(metrics["loss"])
            train_losses.append(train_metrics["loss"])

        # Update the final model state.
        self.model_state = state

        return {
            "test_losses": test_losses,
            "test_accuracy": test_accuracy,
            "train_losses": train_losses,
            "train_accuracy": train_accuracy,
        }

    def train_model_recursively(
        self,
        train_ds: dict,
        test_ds: dict,
        epochs: int = 100,
        batch_size: int = 50,
        disable_loading_bar: bool = False,
    ):
        """
        Check parent class for full doc string.
        """
        if len(train_ds["inputs"]) < batch_size:
            batch_size = len(train_ds["inputs"])

        if self.model_state is None:
            self.init_model()
        state = self.model_state

        condition = False
        counter = 0
        while not condition:
            loading_bar = trange(
                1, epochs + 1, ncols=100, unit="batch", disable=disable_loading_bar
            )
            for i in loading_bar:
                loading_bar.set_description(f"Epoch: {i}")

                state, train_metrics = self._train_epoch(
                    state, train_ds, batch_size=batch_size
                )
                metrics = self._evaluate_model(state.params, test_ds)

                loading_bar.set_postfix(test_loss=metrics["loss"])

            # Update the final model state.
            self.model_state = state

            # Perform checks and update parameters
            counter += 1
            epochs = int(1.1 * epochs)
            if metrics["loss"] <= self.training_threshold:
                condition = True

            # Re-initialize the network if it is simply not converging.
            if counter % 10 == 0:
                logger.info("Model training stagnating, re-initializing model.")
                self.init_model()

    def __call__(self, feature_vector: np.ndarray):
        """
        See parent class for full doc string.
        """
        state = self.model_state

        return self.apply_fn({"params": state.params}, feature_vector)
