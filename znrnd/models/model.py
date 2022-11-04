"""
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincware Project.

Description: Parent class for the models.
"""
import logging
from typing import Callable

import jax.numpy as np
import jax.random
import numpy as onp
from flax.training.train_state import TrainState
from tqdm import trange

from znrnd.accuracy_functions.accuracy_function import AccuracyFunction
from znrnd.utils.prng import PRNGKey

logger = logging.getLogger(__name__)


class Model:
    """
    Parent class for ZnRND Models.

    Attributes
    ----------
    model : Callable
            A callable class or function that takes a feature
            vector and returns something from it. Typically this is a
            neural network layer stack.
    """

    def __init__(
        self,
        loss_fn: Callable,
        optimizer: Callable,
        input_shape: tuple,
        training_threshold: float,
        accuracy_fn: AccuracyFunction = None,
        seed: int = None,
    ):
        """Construct a znrnd model.

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
        seed : int, default None
                Random seed for the RNG. Uses a random int if not specified.
        """
        self.loss_fn = loss_fn
        self.accuracy_fn = accuracy_fn
        self.optimizer = optimizer
        self.input_shape = input_shape
        self.training_threshold = training_threshold

        # Initialized in self.init_model
        self.rng = None

        # initialize the model state
        self.model_state = None
        self.init_model(seed)

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
    ):
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

        return TrainState.create(
            apply_fn=self.apply_fn, params=params, tx=self.optimizer
        )

    def _train_step(self, state: TrainState, batch: dict):
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

        def loss_fn(params):
            """
            helper loss computation
            """
            inner_predictions = self.apply(params, batch["inputs"])
            loss = self.loss_fn(inner_predictions, batch["targets"])
            return loss, inner_predictions

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

        (_, predictions), grads = grad_fn(state.params)

        state = state.apply_gradients(grads=grads)  # in place state update.
        metrics = self._compute_metrics(
            predictions=predictions, targets=batch["targets"]
        )

        return state, metrics

    def _train_epoch(
        self, state: TrainState, train_ds: dict, batch_size: int
    ) -> tuple[TrainState, dict]:
        """
        Train for a single epoch.

        Performs the following steps:

        * Shuffles the data
        * Runs an optimization step on each batch
        * Computes the metrics for the batch
        * Return an updated optimizer, state, and metrics dictionary.

        Parameters
        ----------
        state : TrainState
                Current state of the model.
        train_ds : dict
                Dataset on which to train.
        batch_size : int
                Size of each batch.

        Returns
        -------
        state : TrainState
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

    def validate_model(self, dataset: dict, loss_fn: Callable):
        """
        Validate the model on some external data.

        Parameters
        ----------
        loss_fn : Callable
                Loss function to use in the computation.
        dataset : dict
                Dataset on which to validate the model.
                {"inputs": np.ndarray, "targets": np.ndarray}.

        Returns
        -------
        metrics : dict
                Metrics computed in the validation. {"loss": [], "accuracy": []}.
                Note, for ease of large scale experiments we always return both keywords
                whether they are computed or not.
        """
        predictions = self.apply(self.model_state.params, dataset["inputs"])

        loss = loss_fn(predictions, dataset["targets"])

        if self.accuracy_fn is not None:
            accuracy = self.accuracy_fn(predictions, dataset["targets"])
        else:
            accuracy = None

        return {"loss": loss, "accuracy": accuracy}

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

    def train_model(
        self,
        train_ds: dict,
        test_ds: dict,
        epochs: int = 50,
        batch_size: int = 1,
        disable_loading_bar: bool = False,
    ):
        """
        Train the model on data.

        Parameters
        ----------
        train_ds : dict
                Train dataset with inputs and targets.
        test_ds : dict
                Test dataset with inputs and targets.
        epochs : int
                Number of epochs to train over.
        batch_size : int
                Size of the batch to use in training.
        disable_loading_bar : bool
                Disable the output visualization of the loading par.
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
        epochs_latest_data: int = 0,
        len_latest_data: int = 1,
        epochs_all_data: int = 50,
        batch_size: int = 50,
        disable_loading_bar: bool = False,
    ):
        """
        Train a model recursively until a threshold is reached or the models fails
        to converge.

        Parameters
        ----------
        train_ds : dict
                Train dataset with inputs and targets.
        test_ds : dict
                Test dataset with inputs and targets.
        epochs_latest_data: int
                Number of epochs to train the latest added data per recursion.
        len_latest_data : int
                Defining the length of the latest added data.
                The latest added data defines as the last n subjects in the target set,
                where n == len_latest_data.
        epochs_all_data: int
                Number of epochs to train all data per recursion.
        batch_size : int
                Size of the batch to use in training.
        disable_loading_bar : bool
                Disable the output visualization of the loading par.
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
                1,
                epochs_latest_data + epochs_all_data + 1,
                ncols=100,
                unit="batch",
                disable=disable_loading_bar,
            )
            for i in loading_bar:
                loading_bar.set_description(f"Epoch: {i}")

                if i < epochs_latest_data + 1:
                    train_data = {
                        "inputs": train_ds["inputs"][-len_latest_data:],
                        "targets": train_ds["targets"][-len_latest_data:],
                    }
                else:
                    train_data = train_ds

                state, train_metrics = self._train_epoch(
                    state, train_data, batch_size=batch_size
                )
                metrics = self._evaluate_model(state.params, test_ds)

                loading_bar.set_postfix(test_loss=metrics["loss"])

            # Update the final model state.
            self.model_state = state

            # Perform checks and update parameters
            counter += 1
            epochs_latest_data = int(1.1 * epochs_latest_data)
            epochs_all_data = int(1.1 * epochs_all_data)
            if metrics["loss"] <= self.training_threshold:
                condition = True

            # Re-initialize the network if it is simply not converging.
            if counter % 10 == 0:
                logger.info("Model training stagnating, re-initializing model.")
                self.init_model()

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
        self.model(feature_vector)
