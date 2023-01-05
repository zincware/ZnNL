"""
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincware Project.

Description: Parent class for the Jax-based models.
"""
import logging
from typing import Callable, Tuple, Union

import jax
import jax.numpy as np
import jax.random
import neural_tangents as nt
import numpy as onp
import optax
from flax.training.train_state import TrainState
from tqdm import trange

from znrnd.accuracy_functions.accuracy_function import AccuracyFunction
from znrnd.optimizers.trace_optimizer import TraceOptimizer
from znrnd.utils import normalize_covariance_matrix
from znrnd.utils.prng import PRNGKey

logger = logging.getLogger(__name__)


class JaxModel:
    """
    Parent class for Jax-based models.
    """

    def __init__(
        self,
        loss_fn: Callable,
        optimizer: Union[Callable, TraceOptimizer],
        input_shape: tuple,
        training_threshold: float,
        accuracy_fn: AccuracyFunction = None,
        seed: int = None,
        trace_axes: tuple = (-1),
        ntk_batch_size: int = 10,
    ):
        """
        Construct a znrnd model.

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
        ntk_batch_size : int, default 10
                Batch size to use in the NTK computation.
        trace_axes : Union[int, Sequence[int]]
                Tracing over axes of the NTK.
                The default value is trace_axes(-1,), which reduces the NTK to a tensor
                of rank 2.
                For a full NTK set trace_axes=().
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

        # Set dummy optimizer for case of trace optimizer.
        if isinstance(self.optimizer, TraceOptimizer):
            optimizer = optax.sgd(1.0)
        else:
            optimizer = self.optimizer

        return TrainState.create(apply_fn=self.apply_fn, params=params, tx=optimizer)

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
    ) -> Tuple[TrainState, dict]:
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
        predictions = self.apply(params, batch["inputs"])

        return self._compute_metrics(predictions, batch["targets"])

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
        metrics : dict
                Loss of the model.
        """
        metrics = self._evaluate_step(params, test_ds)
        metrics = jax.device_get(metrics)
        summary = jax.tree_map(lambda x: x.item(), metrics)

        return summary

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

            if isinstance(self.optimizer, TraceOptimizer):
                state = self.optimizer.apply_optimizer(
                    model_state=state,
                    data_set=train_ds["inputs"],
                    ntk_fn=self.compute_ntk,
                    epoch=i,
                )

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
        if x_j is None:
            x_j = x_i
        empirical_ntk = self.empirical_ntk_jit(x_i, x_j, self.model_state.params)

        if infinite:
            infinite_ntk = self.kernel_fn(x_i, x_j, "ntk")
        else:
            infinite_ntk = None

        if normalize:
            empirical_ntk = normalize_covariance_matrix(empirical_ntk)
            if infinite:
                infinite_ntk = normalize_covariance_matrix(infinite_ntk)

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
