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
import numpy as onp
from flax.training.train_state import TrainState
from tqdm import trange

from znnl.accuracy_functions.accuracy_function import AccuracyFunction
from znnl.distance_metrics import DistanceMetric
from znnl.models.jax_model import JaxModel
from znnl.optimizers.trace_optimizer import TraceOptimizer
from znnl.training_recording import JaxRecorder
from znnl.training_strategies.recursive_mode import RecursiveMode
from znnl.training_strategies.simple_training import SimpleTraining
from znnl.training_strategies.training_decorator import train_func

logger = logging.getLogger(__name__)


class LossAwareReservoir(SimpleTraining):
    """
    Class for a biased training strategy using a loss aware reservoir.

    Instead of training on the whole data set, this strategy trains on a reservoir of
    data. The reservoir consists has a maximum size N and is dynamically updated.
    The update is performed by evaluating the loss for each point inside the training
    data. The N points with the biggest loss are selected into the reservoir.

    In addition, the user can select a number of latest_points.
    Background:
    In continual learning, new data is trained when it is available.
    In order to not forget about previously seen data, some old data can be added
    to each batch. The selection of old data is tackled by the loss aware reservoir.
    The latest_points define how many points at the end of the data set have not been
    trained, i.e. are not part of the previously seen data.
    Already seen points, can be put into the reservoir, based on their loss.
    A batch then consists of [latest_points, reservoir_batch].
    From another point of view, it is regular training of the reservoir, adding the
    latest_points to each batch.

    The latest_points are mostly used in recursive training procedures.
    Applying to RND: latest_points = 1, as RND adds one point after each training.

    Aside from the latest points, this class manipulates the probability of training a
    point by scaling it with the loss.

    This training strategy focuses on training of data with strong initial differences
    in the loss. This strategy aims to equalize strong loss differences without
    overfitting points with small initial loss.
    """

    def __init__(
        self,
        model: Union[JaxModel, None],
        loss_fn: Callable,
        accuracy_fn: AccuracyFunction = None,
        seed: int = None,
        reservoir_size: int = 500,
        reservoir_metric: Optional[DistanceMetric] = None,
        latest_points: int = 1,
        recursive_mode: RecursiveMode = None,
        disable_loading_bar: bool = False,
        recorders: List["JaxRecorder"] = None,
    ):
        """
        Constructor of the loss aware reservoir training strategy.

        Parameters
        ----------
        model : Union[JaxModel, None]
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
        reservoir_size : int (default = 100)
                Size of the reservoir, corresponding to the maximum number of points the
                model is trained on.
                This property constraints the memory used while training.
        reservoir_metric : Optional[DistanceMetric]
                The metric i.e. point-wise loss function used to select points into the
                reservoir.
                As default the distance metric underlying the loss function chosen for
                training the model is use.
        latest_points : int (default = 1)
                Number of points defined to be added to the train data set lastly.
                These points will not be selected into the reservoir and are trained in
                every epoch. Selecting latest_points a training batch consist of
                [latest_points, reservoir_batch].
        recursive_mode : RecursiveMode
                Defining the recursive mode that can be used in training.
                If the recursive mode is used, the training will be performed until a
                condition is fulfilled.
                The loss value at which point you consider the model trained.
        disable_loading_bar : bool
                Disable the output visualization of the loading bar.
        recorders : List[JaxRecorder]
                A list of recorders to monitor model training.
        """
        super().__init__(
            model=model,
            loss_fn=loss_fn,
            accuracy_fn=accuracy_fn,
            seed=seed,
            recursive_mode=recursive_mode,
            disable_loading_bar=disable_loading_bar,
            recorders=recorders,
        )

        self.train_data_size = None
        # Define loss aware reservoir of training data
        self.reservoir = None
        self.reservoir_size = reservoir_size
        self.latest_points = latest_points
        # Define distance metric used to select data into the reservoir
        if reservoir_metric:
            self.reservoir_metric = reservoir_metric
        else:
            self.reservoir_metric = loss_fn.metric

    def _update_reservoir(self, train_ds: dict) -> List[int]:
        """
        Updates the reservoir in the following steps:

        * Exclude latest_points from the train_data
        * Check whether the the reservoir will be empty or it can cover all data
        * Compute distance of representations of the remaining training set
        * Sort the training set according to the distance

        The reservoir is a list of indices based on the selection.
        The points are selected by comparing their losses. The n points with the biggest
        loss are put into the reservoir, with n being the reservoir length.

        Parameters
        ----------
        train_ds : dict
                Train data set

        Returns
        -------
        Loss aware reservoir : List[int]
        """
        # Exclude the latest_points from train_ds
        if self.latest_points == 0:
            old_data = train_ds
        else:
            old_data = {k: v[: -self.latest_points, ...] for k, v in train_ds.items()}

        # If the reservoir no old data is available, return an empty array
        if old_data["inputs"].shape[0] == 0:
            data_idx = np.array([])
        # Return the old train data indices if the reservoir can cover them all
        elif self.reservoir_size >= self.train_data_size - self.latest_points:
            data_idx = np.arange(self.train_data_size - self.latest_points)
        # If the reservoir is smaller than the train, data select data via the loss
        else:
            distances = self._compute_distance(old_data)
            data_idx = np.argsort(distances)[::-1][: self.reservoir_size]

        return data_idx

    def _append_latest_points(self, data_idx: List[int], freq: int = 1):
        """
        Append indices of the latest data points to a list of indices.

        The latest data points are located at the end of the training data.
        Therefore, they are accessed via the size of the train data set and the number
        of the latest points.
        They are placed at the beginning of the returned data.

        The latest_data is not shuffled before appending, as the default is
        latest_points=1, which will lead to an unnecessary overhead in most cases.

        Parameters
        ----------
        data_idx : List[int]
                List of ints denoting the indices in the train_ds argument in the
                .train_model method.
        freq : int (default = 1)
                Defining how often the latest indices are appended to the data_idx.
                freq > 1 only used for the trace optimizer.

        Returns
        -------
        data : List[int]
                List of data indices with additional entries at the beginning.
        """
        # Get the indices of the latest points
        idx_latest_points = np.arange(
            self.train_data_size - self.latest_points, self.train_data_size
        )
        # Select the latest points multiple times.
        idx_latest_points = np.repeat(idx_latest_points, freq)
        # Append the latest points to the data.
        data = np.concatenate([idx_latest_points, data_idx], axis=0)
        return data

    def _compute_distance(self, dataset: dict) -> np.ndarray:
        """
        Compute the distance i.e. point-wise loss between neural network representations
        using the reservoir metric.

        Parameters
        ----------
        dataset : dict
                Data set on which distances should be computed.

        Returns
        -------
        distances : np.ndarray
                A tensor of distances computed using the attached metric.
        """
        predictions = self.model(dataset["inputs"])
        return self.reservoir_metric(predictions, dataset["targets"])

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

        # Set defaults
        if not kwargs["epochs"]:
            kwargs["epochs"] = 50
        if not kwargs["batch_size"]:
            kwargs["batch_size"] = (
                len(kwargs["train_ds"]["targets"]) - self.latest_points
            )

        # Raise error if less data available than latest points chosen.
        if self.latest_points > len(kwargs["train_ds"]["targets"]):
            raise (
                AttributeError(
                    "len(train_ds) < latest_points. "
                    "More latest points chosen than data available. "
                )
            )

        # Check for adapting the batch_size
        sizes = (
            len(kwargs["train_ds"]["targets"]) - self.latest_points,
            self.reservoir_size,
            kwargs["batch_size"],
        )
        new_batch_size = min(sizes)
        if new_batch_size != kwargs["batch_size"]:
            kwargs["batch_size"] = new_batch_size
            logger.info(
                "The size of the train data is smaller than the batch size. Setting"
                " the batch size equal to the train data size of"
                f" {kwargs['batch_size']}."
            )

        return kwargs

    def _train_epoch(
        self, state: TrainState, train_ds: dict, batch_size: int
    ) -> Tuple[TrainState, dict]:
        """
        Train for a single epoch.

        This is the customized method of training a single epoch using the loss aware
        reservoir training strategy.
        The method performs the following steps:

        * Updating the reservoir
        * Shuffling the data
        * Batching the data
        * Appending the latest points for each batch
        * Running an optimization step on each batch
        * Computing the metrics for the batch
        * Returning an updated optimizer, state, and metrics dictionary.

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
        # Update the reservoir
        self.reservoir = self._update_reservoir(train_ds=train_ds)

        # If reservoir is empty, only train on latest points
        if batch_size == 0:
            latest_data = {
                k: v[-self.latest_points :, ...] for k, v in train_ds.items()
            }
            state, metrics = self._train_step(state, latest_data)
            batch_metrics = [metrics]
        else:
            batches_per_epoch = len(self.reservoir) // batch_size
            # Prepare the shuffle.
            permutations = jax.random.permutation(self.rng(), self.reservoir)
            permutations = np.array_split(permutations, batches_per_epoch)

            # Step over items in batch and append the latest points to each batch
            batch_metrics = []
            for permutation in permutations:
                permutation = self._append_latest_points(permutation)
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

    @train_func
    def train_model(
        self,
        train_ds: dict,
        test_ds: dict,
        epochs: Optional[int] = None,
        batch_size: int = None,
    ) -> dict:
        """
        Train the model on data.

        Parameters
        ----------
        train_ds : dict
                Train dataset with inputs and targets.
        test_ds : dict
                Test dataset with inputs and targets.
        epochs : Optional[int] (default = 50)
                Number of epochs to train over.
        batch_size : int (default = len(train_ds))
                Size of a training batch of the reservoir. The latest_points are not
                taken into account in the batch_size as they are added additionally
                to each batch.

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
        self.train_data_size = len(train_ds["targets"])

        loading_bar = trange(
            1, epochs + 1, ncols=100, unit="batch", disable=self.disable_loading_bar
        )

        train_losses = []
        train_accuracy = []
        for i in loading_bar:
            # Update the recorder properties
            if self.recorders is not None:
                for item in self.recorders:
                    item.update_recorder(epoch=i, model=self.model)

            loading_bar.set_description(f"Epoch: {i}")

            # Update the trace optimizer
            if isinstance(self.model.optimizer, TraceOptimizer):
                # Update the reservoir
                self.reservoir = self._update_reservoir(train_ds)
                # Compute the number of batches
                batches_per_epoch = len(self.reservoir) // batch_size
                # Create the data set used for the trace optimizer
                full_dataset_idx = self._append_latest_points(
                    data_idx=self.reservoir, freq=batches_per_epoch
                )
                state = self.model.optimizer.apply_optimizer(
                    model_state=state,
                    data_set=train_ds["inputs"][full_dataset_idx],
                    ntk_fn=self.model.compute_ntk,
                    epoch=i,
                )

            state, train_metrics = self._train_epoch(
                state=state, train_ds=train_ds, batch_size=batch_size
            )
            self.review_metric = self._evaluate_model(state.params, test_ds)
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
