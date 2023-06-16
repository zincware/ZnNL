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
from dataclasses import dataclass
from typing import Callable, Optional
from flax.training.early_stopping import EarlyStopping

logger = logging.getLogger(__name__)


@dataclass
class RecursiveMode:
    """
    Class of the mode of recursive training.

    Each training strategy comes with a recursive mode, which is defined in this class.

    The recursive mode trains a model until a condition is satisfied or the training is
    considered to be stuck in a local minimum. In the latter case, the model is
    perturbed and the training is repeated.

    Attributes
    ----------
    use_recursive_mode : bool (default = True)
            If True, the recursive mode is used.
    min_delta : float (default = 1e-2)
            Minimum change in the monitored loss value to qualify as an improvement.
    patience : int (default = 1)
            Number of updates with no improvement after which training will be stopped.
            Defining the train_model method of the training strategy, an update is
            performed after the defined number of epochs.
            It is important to consider the number of epochs when setting the patience.
    threshold : Optional[float]
            The loss value which defines the model to be trained.
    scale_factor : float (default = 1.1)
            Factor which is used to scale the number of epochs after one cycle of
            recursive training.
            The update is: epochs = scale_factor * epochs
    break_counter : int
            Maximum number of cycles that can be performed in recursive training,
            before accepting the model state to be stuck a local minimum.
            To get out of the local minimum the model is perturbed with perturb_fn.
    perturb_fn : Callable
            Function to perturb the model, when the training is considered to be stuck.
            Calling the function will perturb the model.
            The default is set to re-initializing the model. This done inside the
            recursive_decorator function.
    """

    use_recursive_mode: bool = True
    min_delta: float = 1e-2
    patience: int = 1
    threshold: float = None
    scale_factor = float = 1.1
    break_counter: float = 10
    perturb_fn: Callable = None

    early_stop = EarlyStopping(min_delta=min_delta, patience=patience)

    def update_recursive_condition(self, measure) -> bool:
        """
        Check and update the condition for stopping the recursive training.

        Parameters
        ----------
        measure : float
                Measure to compare against the threshold.

        Returns
        -------
        Boolean value
        If True, the training will be stopped.
        If False, the training continues.
        """
        _, self.early_stop = self.early_stop.update(measure)
        early_stopping_condition = self.early_stop.should_stop
        if self.threshold:
            threshold_condition = measure <= self.threshold
            return early_stopping_condition or threshold_condition
        else:
            return early_stopping_condition

    def perturb_training(self):
        """
        Perturb the training with the perturb_fn as it is stuck in a local minimum.
        Also logging when doing so.

        Returns
        -------

        """
        logger.info("Model training stagnating, perturbing model and starting again.")
        self.perturb_fn()
