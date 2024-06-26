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
from typing import Callable, List, Union

import jax.numpy as np
from tqdm import trange

from znnl.accuracy_functions.accuracy_function import AccuracyFunction
from znnl.analysis.jax_ntk import JAXNTKComputation
from znnl.models.jax_model import JaxModel
from znnl.optimizers.trace_optimizer import TraceOptimizer
from znnl.training_recording import JaxRecorder
from znnl.training_strategies.recursive_mode import RecursiveMode
from znnl.training_strategies.simple_training import SimpleTraining
from znnl.training_strategies.training_decorator import train_func

logger = logging.getLogger(__name__)


class PartitionedTraining(SimpleTraining):
    """
    Class for the partitioned training strategy.

    In this training strategy, the user can select partitions/subsets of a data set as
    training data. The partitions are sequentially trained using a given number of
    epochs and a batch size. The partitions can be any defined subset of the full
    data set and therefore, also overlap.
    The selected partitions are passed as an argument in the train_model method.

    This training strategy offers the possibility of leaving data out or training data
    multiple times during one training run.

    This strategy aims to equalize of create loss differences of data by training on
    user defined partitions of the data. The user can decide which parts to focus on in
    the training.
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
        Constructor of the partitioned training strategy.

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

    @staticmethod
    def _select_partition(train_ds: dict, data_slice: Union[np.array, slice]):
        """
        Select a partitions

        This method selects a subset from data.
        The training will be performed with the selected subsets instead of all training
        data.

        Parameters
        ----------
        train_ds : dict
                Train dataset with inputs and targets.
        data_slice : Union[np.array, slice]
                Slice to select from train_ds. It can either be a np.array or a slice
                defining the indices.

        Returns
        -------
        partition : dict
        Selected partition of data.
        """
        return {k: v[data_slice, ...] for k, v in train_ds.items()}

    def update_training_kwargs(self, **kwargs):
        """
        Check model and keyword arguments before executing the training.

        In detail:
            * Raise an error if no model is applied.
            * Set default value for epochs (default = [150, 50])
            * set default value for the batch size (default = 1)
            * Set default value for train_ds_selection if necessary
                (default = [slice(-1, None, None), slice(None, None, None)])
            * Check that epochs, batch_size and train_ds_selection are of similar
                length and of type list.

        The combination of default parameters for epochs and train_ds_selection means
        that the last point in train_ds is trained for 150 epochs and the all other
        points are trained for 50 epochs afterwards.

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
            kwargs["epochs"] = [150, 50]
        if not kwargs["batch_size"]:
            kwargs["batch_size"] = [1, 1]
        if not kwargs["train_ds_selection"]:
            kwargs["train_ds_selection"] = [
                slice(-1, None, None),
                slice(None, None, None),
            ]

        # Check type list
        def raise_key_list_error(key):
            raise KeyError(
                f"The kwarg {key} of the function train_model has to be of type list. "
            )

        if type(kwargs["epochs"]) is int:
            raise_key_list_error(kwargs["epochs"])
        if not kwargs["batch_size"]:
            raise_key_list_error(kwargs["batch_size"])
        if not kwargs["train_ds_selection"]:
            raise_key_list_error(kwargs["train_ds_selection"])

        # Check similar length of lists
        if not (
            len(kwargs["epochs"])
            == len(kwargs["batch_size"])
            == len(kwargs["train_ds_selection"])
        ):
            raise KeyError(
                "The args for epochs, batch_size, and train_ds_selection do not "
                "correspond in length: "
                f"len(epochs)={len(kwargs['epochs'])}, "
                f"len(batch_size)={len(kwargs['batch_size'])}, "
                f"len(train_ds_selection)={len(kwargs['train_ds_selection'])}. "
                "Make sure that for all of them have similar length. "
            )

        # Check for adapting the batch_sizes
        for i, selection in enumerate(kwargs["train_ds_selection"]):
            if len(kwargs["train_ds"]["targets"][selection]) < kwargs["batch_size"][i]:
                kwargs["batch_size"][i] = len(kwargs["train_ds"]["targets"][selection])
                logger.info(
                    f"The size of the train data in slice {i} is smaller than the batch"
                    " size. Setting the batch size equal to the train data size of"
                    f" {kwargs['batch_size']}."
                )

        return kwargs

    @train_func
    def train_model(
        self,
        train_ds: dict,
        test_ds: dict,
        epochs: list = None,
        train_ds_selection: list = None,
        batch_size: list = None,
    ) -> dict:
        """
        Train the model on data.

        The training is performed using the PartitionedTraining strategy.
        The partitioning is passes in this method.
        Each partition is defined by as a slice or a list of indices.
        The training order of the partitions in the list is determined from left (first)
        to right (last).
        For each partition, the user selects epochs and batch_size for training.

        Parameters
        ----------
        train_ds : dict
                Train dataset with inputs and targets.
        test_ds : dict
                Test dataset with inputs and targets.
        epochs : list (default = [150, 50])
                Number of epochs to train over.
                Each epoch defines a training phase.
        train_ds_selection : list
                        (default = [slice(-1, None, None), slice(None, None, None)])
                The train is selected by a np.array of indices or slices.
                Each slice or array defines a training phase.
        batch_size : list (default = [1, 1])
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

        if isinstance(self.model.optimizer, TraceOptimizer):
            ntk_computation = JAXNTKComputation(
                self.model.ntk_apply_fn, trace_axes=(-1,), batch_size=batch_size
            )

        loading_bar = trange(
            1,
            np.sum(epochs) + 1,
            ncols=100,
            unit="batch",
            disable=self.disable_loading_bar,
        )

        train_losses = []
        train_accuracy = []
        training_phase = 0
        epoch_phase_counter = 0
        train_data = self._select_partition(
            train_ds, train_ds_selection[training_phase]
        )

        for i in loading_bar:
            # Update the recorder properties
            if self.recorders is not None:
                for item in self.recorders:
                    item.record(epoch=i, model=self.model)

            loading_bar.set_description(f"Phase: {training_phase+1}, Epoch: {i}")

            if epoch_phase_counter >= epochs[training_phase]:
                training_phase += 1
                train_data = self._select_partition(
                    train_ds, train_ds_selection[training_phase]
                )
                epoch_phase_counter = 0

            if isinstance(self.model.optimizer, TraceOptimizer):
                state = self.model.optimizer.apply_optimizer(
                    model_state=state,
                    data_set=train_data["inputs"],
                    ntk_fn=ntk_computation.compute_ntk,
                    epoch=i,
                )

            state, train_metrics = self._train_epoch(
                state=state, train_ds=train_data, batch_size=batch_size[training_phase]
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

            epoch_phase_counter += 1

            # Update the class model state.
            self.model.model_state = state

        return {
            "train_losses": train_losses,
            "train_accuracy": train_accuracy,
        }
