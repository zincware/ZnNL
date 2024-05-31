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
from typing import Callable, List, Optional, Tuple, Union

import jax
import jax.numpy as np
import jax.random
import numpy as onp
from flax.training.train_state import TrainState
from tqdm import trange

from znnl.accuracy_functions.accuracy_function import AccuracyFunction
from znnl.analysis.jax_ntk import JAXNTKComputation
from znnl.models.jax_model import JaxModel
from znnl.optimizers.trace_optimizer import TraceOptimizer
from znnl.training_recording import JaxRecorder
from znnl.training_strategies.recursive_mode import RecursiveMode
from znnl.training_strategies.training_decorator import train_func
from znnl.training_strategies.training_steps import TrainStep
from znnl.utils.prng import PRNGKey

logger = logging.getLogger(__name__)


class SimpleTraining:
    """
    Class for training a model using a simple training strategy.

    This class is parent to other training strategies and defines the most basic type
    of training strategy.

    In this strategy, all training data is trained for a given number epochs and a batch
    size.
    Each data point is assumed to be equally important for the training. Every data
    point is trained once in each epoch. The order of the batches is randomly generated.

    This training strategy focuses on learning distributions based on i.i.d. data.
    """

    def __init__(
        self,
        model: Union[JaxModel, None],
        loss_fn: Callable,
        accuracy_fn: AccuracyFunction = None,
        seed: int = None,
        recursive_mode: RecursiveMode = None,
        disable_loading_bar: bool = False,
        recorders: List["JaxRecorder"] = None,
    ):
        """
        Construct a simple training strategy for a model.

        Parameters
        ----------
        model : Union[JaxModel, None]
                Model class for a Jax model.
                "None" is only used if the training strategy is passed as an input
                to a bigger framework. The strategy then is applied to the framework
                and the model instantiation is handled by that framework.
        loss_fn : Callable
                A function to use in the loss computation.
        accuracy_fn : AccuracyFunction (default = None)
                Funktion class for computing the accuracy of model and given data.
        seed : int (default = None)
                Random seed for the RNG. Uses a random int if not specified.
        recursive_mode : RecursiveMode
                Defining the recursive mode that can be used in training.
                If the recursive mode is used, the training will be performed until a
                condition is fulfilled.
        disable_loading_bar : bool
                Disable the output visualization of the loading bar.
        recorders : List[JaxRecorder]
                A list of recorders to monitor model training.
        """
        self.model = model
        self.loss_fn = loss_fn
        self.accuracy_fn = accuracy_fn
        self.recursive_mode = recursive_mode

        self.disable_loading_bar = disable_loading_bar
        self.recorders = recorders

        self.rng = PRNGKey(seed)

        self.review_metric = None

        # Initialize the train step
        self._train_step = None
        self._init_train_step()

    def _init_train_step(self):
        """
        Initialize the train step.

        This is necessary to be able to use the train step in the recursive mode.
        """
        if self.model is None:
            logger.info(
                "No model is applied. The train step cannot be initialized yet."
                "To train a model, load a model into the training strategy using the"
                "set_model method."
            )
        else:
            self._train_step = TrainStep(self.model.model_state)

    def set_model(self, model: JaxModel):
        """
        Set the model to train.

        Parameters
        ----------
        model : JaxModel
                Model to train.
        """
        self.model = model
        self._init_train_step()

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
        if self.accuracy_fn is not None:
            accuracy = self.accuracy_fn(predictions, targets)
            metrics = {"loss": loss, "accuracy": accuracy}

        else:
            metrics = {"loss": loss}

        return metrics

    def _evaluate_step(self, params: dict, batch: dict):
        """
        Evaluate the model on test data.

        Parameters
        ----------
        params: dict
                Contains the model parameters to use for the model computation.
                It is a dictionary of structure
                {'params': params, 'batch_stats': batch_stats}
        batch : dict
                Batch of data to test on.

        Returns
        -------
        metrics : dict
                Metrics dict computed on test data.
        """
        predictions = self.model.apply(params, batch["inputs"])

        return self._compute_metrics(predictions, batch["targets"])

    def _evaluate_model(self, params: dict, test_ds: dict) -> dict:
        """
        Evaluate the model.

        Parameters
        ----------
        params: dict
                Contains the model parameters to use for the model computation.
                It is a dictionary of structure
                {'params': params, 'batch_stats': batch_stats}
        test_ds : dict
                Dataset on which to evaluate.
        Returns
        -------
        metrics : dict
                Loss of the model.
        """
        metrics = self._evaluate_step(params, test_ds)
        metrics = jax.device_get(metrics)
        summary = jax.tree_util.tree_map(lambda x: x.item(), metrics)

        return summary

    def _train_epoch(
        self, state: TrainState, train_ds: dict, batch_size: int
    ) -> Tuple[TrainState, dict]:
        """
        Train for a single epoch.

        This is the default method for training a full epoch.
        Can be implemented in different ways in the child classes.

        Performs the following steps:

        * Shuffles the data
        * Batches the data assigning equal probability
        * Runs an optimization step on each batch weighting
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
        metrics : Tuple[TrainState, dict]
                Tuple of train state and metrics for current state.
        """
        # Some housekeeping variables.
        train_ds_size = len(train_ds["inputs"])
        steps_per_epoch = train_ds_size // batch_size

        if train_ds_size == 1:
            state, metrics = self._train_step(
                state=state,
                batch=train_ds,
                loss_fn=self.loss_fn,
                compute_metrics_fn=self._compute_metrics,
            )
            batch_metrics = [metrics]

        else:
            # Prepare the shuffle.
            permutations = jax.random.permutation(self.rng(), train_ds_size)
            permutations = np.array_split(permutations, steps_per_epoch)

            # Step over items in batch.
            batch_metrics = []
            for permutation in permutations:
                batch = {k: v[permutation, ...] for k, v in train_ds.items()}
                state, metrics = self._train_step(
                    state=state,
                    batch=batch,
                    loss_fn=self.loss_fn,
                    compute_metrics_fn=self._compute_metrics,
                )
                batch_metrics.append(metrics)

        # Get the metrics off device for printing.
        batch_metrics_np = jax.device_get(batch_metrics)
        epoch_metrics_np = {
            k: onp.mean([metrics[k] for metrics in batch_metrics_np])
            for k in batch_metrics_np[0]
        }

        return state, epoch_metrics_np

    def update_training_kwargs(self, **kwargs):
        """
        Check model and keyword arguments before executing the training.

        In detail:
            * Raise an error if no model is applied.
            * Set default value for epochs (default = 50)
            * set default value for the batch size (default = train data length)
            * Adapt batch size if there is too little data for one batch

        Parameters
        ----------
        kwargs : dict
                See more details in the docstring of the train_model method below.

        Returns
        -------
        Updated kwargs of the train_fn.
        """
        if self.model is None:
            raise KeyError(
                f"self.model = {self.model}. "
                "If the training strategy should operate on a model, a model"
                "must be given."
                "Pass the model in the construction."
            )

        if not kwargs["epochs"]:
            kwargs["epochs"] = 50
        if not kwargs["batch_size"]:
            kwargs["batch_size"] = len(kwargs["train_ds"]["targets"])

        if len(kwargs["train_ds"]["inputs"]) < kwargs["batch_size"]:
            kwargs["batch_size"] = len(kwargs["train_ds"]["inputs"])
            logger.info(
                "The size of the train data is smaller than the batch size: Setting"
                " the batch size equal to the train data size of"
                f" {kwargs['batch_size']}."
            )

        return kwargs

    @train_func
    def train_model(
        self,
        train_ds: dict,
        test_ds: dict,
        epochs: Optional[Union[int, List[int]]] = None,
        batch_size: Optional[Union[int, List[int]]] = None,
        **kwargs,
    ) -> dict:
        """
        Train the model on data.

        Parameters
        ----------
        train_ds : dict
                Train dataset with inputs and targets.
        test_ds : dict
                Test dataset with inputs and targets.
        epochs : Optional[Union[int, List[int]]] (default = 50)
                Number of epochs to train over.
        batch_size : Optional[Union[int, List[int]]]
                Size of the batch to use in training.
        **kwargs
                No additional kwargs in this class.

        Returns
        -------
        in_training_metrics : dict
            Whilst the recorders can return all useful metrics, the model still returns
            the loss and accuracy that is measured during the training. These can differ
            as the loss and accuracy during training can be done batch-wise in between
            model updates whereas the recorder will store the results on a single set
            of parameters.
        """
        state = self.model.model_state

        if isinstance(self.model.optimizer, TraceOptimizer):
            ntk_computation = JAXNTKComputation(
                self.model.ntk_apply_fn, trace_axes=(-1,), batch_size=batch_size
            )

        loading_bar = trange(
            0, epochs, ncols=100, unit="batch", disable=self.disable_loading_bar
        )

        train_losses = []
        train_accuracy = []
        for i in loading_bar:
            # Update the recorder properties
            if self.recorders is not None:
                for item in self.recorders:
                    item.record(epoch=i, model=self.model)

            loading_bar.set_description(f"Epoch: {i}")

            if isinstance(self.model.optimizer, TraceOptimizer):
                state = self.model.optimizer.apply_optimizer(
                    model_state=state,
                    data_set=train_ds["inputs"],
                    ntk_fn=ntk_computation.compute_ntk,
                    epoch=i,
                )

            state, train_metrics = self._train_epoch(
                state, train_ds, batch_size=batch_size
            )
            self.review_metric = self._evaluate_model(
                {"params": state.params, "batch_stats": state.batch_stats}, test_ds
            )
            train_losses.append(train_metrics["loss"])

            # Update the loading bar
            loading_bar.set_postfix(test_loss=self.review_metric["loss"])
            try:
                loading_bar.set_postfix(accuracy=self.review_metric["accuracy"])
                train_accuracy.append(train_metrics["accuracy"])
            except KeyError:
                pass

            # Update the class model state.
            self.model.model_state = state

        return {
            "train_losses": train_losses,
            "train_accuracy": train_accuracy,
        }
