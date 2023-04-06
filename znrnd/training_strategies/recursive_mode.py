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
Module for recording jax training.
"""
import logging
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from znrnd.training_strategies.simple_training import SimpleTraining

import jax.numpy as np

logger = logging.getLogger(__name__)


class RecursiveMode:
    """
    Class of the mode of recursive training.

    Each training strategy comes with a recursive mode, which is defined in this class.
    This class is initialized by the user, but instantiated automatically right before
    the training process inside the training decorator.
    """

    def __init__(
        self,
        update_type: str = "rnd",
        threshold: float = 0.01,
        scale_factor: float = 1.1,
        break_counter: float = 10,
        use_recursive_mode: bool = True,
    ):
        """

        Parameters
        ----------
        update_type : Callable
                Function to update the recursive condition.
        threshold : float
                The loss value which defines the model to be trained.
        scale_factor : float (default = 1.1)
                Factor which is used to scale the number of epochs after one cycle of
                recursive training.
                The update is: epochs = scale_factor * epochs
        break_counter : int
                Maximum number of cycles that can be performed in recursive training,
                before accepting the model state to be stuck a local minimum.
                To get out of the local minimum the model is perturbed with perturb_fn.
        use_recursive_mode : bool (default = True)
                If True, the recursive mode is used.
        """
        self.update_type = update_type
        self.threshold = threshold
        self.scale_factor = scale_factor
        self.break_counter = break_counter
        self.use_recursive_mode = use_recursive_mode

        self.update_fn: Callable = None
        self.perturb_fn = None

        self._training_strategy = None
        self._metric_state: float = None
        self._condition: bool = False

    def instantiate_recursive_mode(self, training_strategy):
        """
        Instantiate all necessary functions for the recursive mode.

        Parameters
        ----------
        training_strategy : SimpleTraining
                Training strategy used to operate recursively on.
        """
        self._training_strategy = training_strategy
        self.set_update_fn()
        self.set_perturb_fn()

        self._condition: bool = False

    def set_perturb_fn(self):
        """
        Set the function for perturbing the model in recursive training.

        Function to perturb the model, when the training is considered to be stuck.
        Calling the function will perturb the model.

        Todo: Enable the user to set a custom perturbation function.
        """
        self.perturb_fn = self._training_strategy.model.init_model

    def set_update_fn(self):
        """
        Set the function for updating recursive condition. This function defines the
        condition that is to be fulfilled.

        There are two types of update functions: "threshold" and "rnd".
        More information to the individual update function can be found the respective
        docstring.
        """
        if self.update_type == "threshold":
            self.update_fn = self._update_fn_threshold
        if self.update_type == "rnd":
            self.update_fn = self._update_fn_rnd

    def _update_fn_threshold(self, dataset):
        """
        Update function defining the condition via a threshold.

        This method checks if the loss of all points in the dataset is below a
        threshold.

        Parameters
        ----------
        dataset : dict
                Data set used to check the condition.

        Returns
        -------
        condition : bool
                Boolean that tells whether the condition is fulfilled.
        """
        self._metric_state = self._training_strategy.evaluate_model(
            self._training_strategy.model.model_state.params, dataset
        )
        condition = self._metric_state <= self.threshold
        return condition

    def _update_fn_rnd(self, dataset) -> bool:
        """
        Update function defining the condition for RND.

        For the case of RND the dataset in this method is split into the last point
        (new point) and all others (old points). The condition is fulfilled, if
        1. The loss of the last point is below the loss of all other points.
        2. The loss of all other points is below a threshold.

        Parameters
        ----------
        dataset : dict
                Data set used to check the condition.

        Returns
        -------
        condition : bool
                Boolean that tells whether the condition is fulfilled.

        """
        ds_1 = {"inputs": dataset["inputs"][-1:], "targets": dataset["targets"][-1:]}
        ds_2 = {"inputs": dataset["inputs"][:-1], "targets": dataset["targets"][:-1]}

        metric_1 = self._training_strategy.evaluate_model(
            self._training_strategy.model.model_state.params, ds_1
        )["loss"]
        metric_2 = self._training_strategy.evaluate_model(
            self._training_strategy.model.model_state.params, ds_2
        )["loss"]

        condition_1 = metric_2 - metric_1 >= 0
        condition_2 = metric_2 <= self.threshold
        return np.logical_and(condition_1, condition_2)

    def perturb_training(self):
        """
        Perturb the training with the perturb_fn as it is stuck in a local minimum.
        Also logging when doing so.
        """
        logger.info("Model training stagnating, perturbing model and starting again.")
        self.perturb_fn()
