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
from typing import Callable, List, Optional, Union

import jax.numpy as np
from tqdm import trange

from znrnd.accuracy_functions.accuracy_function import AccuracyFunction
from znrnd.distance_metrics import DistanceMetric
from znrnd.models.jax_model import JaxModel
from znrnd.optimizers.trace_optimizer import TraceOptimizer
from znrnd.training_recording import JaxRecorder
from znrnd.training_strategies.recursive_mode import RecursiveMode
from znrnd.training_strategies.simple_training import (
    SimpleTraining,
    recursive_decorator,
)

logger = logging.getLogger(__name__)


class LossAwareReservoir(SimpleTraining):
    """
    Class for a biased training strategy using a loss aware reservoir.

    Instead of training on the whole data set, this strategy trains on a reservoir of
    data. The reservoir consists has a maximum size N and is dynamically updated.
    The update is performed by evaluating the loss for each point inside the training
    data. The N points with the biggest loss are put into the reservoir.

    This manipulates the probability of training a point by scaling it with the loss.

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

        # Define loss aware reservoir of training data
        self.reservoir = None
        self.reservoir_size = reservoir_size
        # Define distance metric used to select data into the reservoir
        if reservoir_metric:
            self.reservoir_metric = reservoir_metric
        else:
            self.reservoir_metric = loss_fn.metric

    def _update_reservoir(self, train_ds) -> dict:
        """
        Updates the reservoir in the following steps:

        * Compute distance of representations of the training set
        * Sort the training set according to the distance
        * Update the reservoir with the sorted training set

        The reservoir is a dictionary, similar to the train_ds, containing points based
        on a selection.
        The points are selected by comparing their losses. The n points with the biggest
        loss are put into the reservoir, with n being the reservoir length.

        Returns
        -------
        Loss aware reservoir : dict
        """
        distances = self._compute_distance(train_ds)
        max_size = self.reservoir_size

        # Return the whole train data if reservoir can cover them all
        if max_size >= len(train_ds["targets"]):
            return train_ds
        # If the reservoir is smaller than the train data select data via the loss
        sorted_idx = np.argsort(distances)[::-1][:max_size]
        return {k: v[sorted_idx, ...] for k, v in train_ds.items()}

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
            * Raise an error no model is applied.
            * Set default value for epochs (default = 50)
            * set default value for the batch size (default = 1)
            * Adapt batch size if there is too little data for one batch

        Parameters
        ----------
        kwargs : dict
                Keyword arguments of the train_fn.

        Returns
        -------
        Updated kwargs of the train_fn.
        """
        if self.model is None:
            raise KeyError(
                "self.model = None \n"
                "If the training strategy should operate on a model, a model"
                "must be given."
                "Pass the model in the construction."
            )

        if not kwargs["epochs"]:
            kwargs["epochs"] = 50
        if not kwargs["batch_size"]:
            kwargs["batch_size"] = 1

        if self.reservoir_size < kwargs["batch_size"]:
            kwargs["batch_size"] = self.reservoir_size
            if len(kwargs["train_ds"]) < self.reservoir_size:
                kwargs["batch_size"] = len(kwargs["train_ds"])
            logger.info(
                "The size of the train data is smaller than the batch size. Setting"
                " the batch size equal to the train data size of"
                f" {kwargs['batch_size']}."
            )

        return kwargs

    def _check_training_kwargs(
        self, train_ds: dict, epochs: Optional[Union[int, List[int]]], batch_size: int
    ):
        """
        Check if the arguments for the training are properly set.

            * Raise and error if no model is set
            * Reset the batch size if batch_size > len(train_ds)
            * Set default value for epochs

        Parameters
        ----------
        train_ds : dict
                Train dataset with inputs and targets.
        epochs : Optional[Union[int, List[int]]] (default = 50)
                Number of epochs to train over.
        batch_size : int
                Size of the batch to use in training.

        Returns
        -------
        Possible new train parameters
        """
        # Raise error if no model is available
        if self.model is None:
            raise KeyError(
                "self.model = None. "
                "If the training strategy should operate on a model, a model"
                "must be given."
                "Pass the model in the construction."
            )

        if self.reservoir_size < batch_size:
            batch_size = self.reservoir_size
            if len(train_ds) < self.reservoir_size:
                batch_size = len(train_ds)
            logger.info(
                "The size of the train data is smaller than the batch size. "
                f"Setting the batch size equal to the train data size of {batch_size}."
            )
        if not epochs:
            epochs = 50

        return batch_size, epochs

    @recursive_decorator
    def train_model(
        self,
        train_ds: dict,
        test_ds: dict,
        epochs: Optional[int] = None,
        batch_size: int = None,
    ):
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
        batch_size : int (default = 1)
                Size of the batch to use in training.

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

        loading_bar = trange(
            1, epochs + 1, ncols=100, unit="batch", disable=self.disable_loading_bar
        )

        train_losses = []
        train_accuracy = []
        for i in loading_bar:
            self.reservoir = self._update_reservoir(train_ds)
            # Update the recorder properties
            if self.recorders is not None:
                for item in self.recorders:
                    item.update_recorder(epoch=i, model=self.model)

            loading_bar.set_description(f"Epoch: {i}")

            if isinstance(self.model.optimizer, TraceOptimizer):
                state = self.model.optimizer.apply_optimizer(
                    model_state=state,
                    data_set=self.reservoir["inputs"],
                    ntk_fn=self.model.compute_ntk,
                    epoch=i,
                )

            state, train_metrics = self._train_epoch(
                state=state, train_ds=self.reservoir, batch_size=batch_size
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
