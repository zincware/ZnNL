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
from dataclasses import dataclass
from typing import Callable

logger = logging.getLogger(__name__)


@dataclass
class RecursiveMode:
    """
    Class of the mode of recursive training.

    Each training strategy comes with a recursive mode, which is defined in this class.

    Attributes
    ----------
    use_recursive_mode : bool (default = True)
            If True, the recursive mode is used.
    threshold : float
            The loss value which defines the model to be trained.
    scale_factor : float (default = 1.1)
            Factor which is used to scale the number of epochs after one cycle of
            recursive training.
            The update is: epochs = scale_factor * epochs
    break_condition : int
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
    threshold: float = 0.01
    scale_factor = float = 1.1
    break_counter: float = 10
    perturb_fn: Callable = None

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
        return measure <= self.threshold

    def perturb_training(self):
        """
        Perturb the training with the perturb_fn as it is stuck in a local minimum.
        Also logging when doing so.

        Returns
        -------

        """
        logger.info("Model training stagnating, perturbing model and starting again.")
        self.perturb_fn()
