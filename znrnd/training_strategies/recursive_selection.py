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

import numpy as onp
from tqdm import trange

from znrnd.accuracy_functions.accuracy_function import AccuracyFunction
from znrnd.model_recording import JaxRecorder
from znrnd.models.jax_model import JaxModel
from znrnd.optimizers.trace_optimizer import TraceOptimizer
from znrnd.training_strategies.simple_training import SimpleTraining

logger = logging.getLogger(__name__)


class RecursiveSelection(SimpleTraining):
    """
    Class for a biased training strategy based on iterative training of individual parts
    of the data.

    Data has non-uniform probability of being trained.
    The data is chosen by slices of indices and is trained for a certain number of
    epochs.
    """

    def __init__(
        self,
        model: JaxModel,
        loss_fn: Callable,
        accuracy_fn: AccuracyFunction = None,
        seed: int = None,
        recursive_use: bool = False,
        recursive_threshold: float = None,
        recorders: List["JaxRecorder"] = None,
    ):
        """
        Construct a biased training strategy for a model.

        Parameters
        ----------
        model : JaxModel
                Model class for a Jax model.
        loss_fn : Callable
                A function to use in the loss computation.
        accuracy_fn : AccuracyFunction (default = None)
                Funktion class for computing the accuracy of model and given data.
        seed : int (default = None)
                Random seed for the RNG. Uses a random int if not specified.
        recursive_use : bool (default = False)
                If False, the training will be performed for a given number of epochs.
                If True, the training will be performed until a condition is fulfilled.
                After a given number of epochs, the training continues for more epochs.
        recursive_threshold : float
                The loss value at which point you consider the model trained.
        recorders : List[JaxRecorder]
                A list of recorders to monitor model training.
        """
        super().__init__(
            model=model,
            loss_fn=loss_fn,
            accuracy_fn=accuracy_fn,
            seed=seed,
            recursive_use=recursive_use,
            recursive_threshold=recursive_threshold,
            recorders=recorders,
        )

    @staticmethod
    def _select_ds(train_ds, data_slice):
        """
        Select a subset of data from a dict.

        Parameters
        ----------
        train_ds : dict
                Train dataset with inputs and targets.
        data_slice : slice
                Slice to select from train_ds

        Returns
        -------
        dict
        Selected subset of data.
        """
        return {k: v[data_slice, ...] for k, v in train_ds.items()}

    def _train_model(
        self,
        train_ds: dict,
        test_ds: dict,
        epochs: list[int] = None,
        train_ds_selection: list[slice] = None,
        batch_size: int = 1,
        disable_loading_bar: bool = False,
        **kwargs,
    ):
        """
        Train the model on data.

        Parameters
        ----------
        train_ds : dict
                Train dataset with inputs and targets.
        test_ds : dict
                Test dataset with inputs and targets.
        epochs : list[int]
                Number of epochs to train over.
                Each epoch defines a training phase.
        train_ds_selection : list[slice]
                Indices or slices selecting training data.
                Each slice or index defines a training phase.
        batch_size : int
                Size of the batch to use in training.
        disable_loading_bar : bool
                Disable the output visualization of the loading par.
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

        if not epochs:
            epochs = [150, 50]
            train_ds_selection = [[-1], slice(1, None, None)]

        state = self.model.model_state

        loading_bar = trange(
            1, onp.sum(epochs) + 1, ncols=100, unit="batch", disable=disable_loading_bar
        )

        train_losses = []
        train_accuracy = []
        training_phase = 0
        epoch_phase_counter = 0
        train_data = self._select_ds(train_ds, train_ds_selection[training_phase])

        for i in loading_bar:
            # Update the recorder properties
            if self.recorders is not None:
                for item in self.recorders:
                    item.update_recorder(epoch=i, model=self.model)

            loading_bar.set_description(f"Phase: {training_phase+1}, Epoch: {i}")

            if epoch_phase_counter >= epochs[training_phase]:
                training_phase += 1
                train_data = self._select_ds(
                    train_ds, train_ds_selection[training_phase]
                )
                epoch_phase_counter = 0

            if isinstance(self.model.optimizer, TraceOptimizer):
                state = self.model.optimizer.apply_optimizer(
                    model_state=state,
                    data_set=train_data["inputs"],
                    ntk_fn=self.model.compute_ntk,
                    epoch=i,
                )

            state, train_metrics = self._train_epoch(
                state=state, train_ds=train_data, batch_size=batch_size
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

            epoch_phase_counter += 1

            # Update the class model state.
            self.model.model_state = state

        return {
            "train_losses": train_losses,
            "train_accuracy": train_accuracy,
        }
