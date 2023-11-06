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
Module containing the abstract class for a regularizer.
"""
import logging
from abc import ABC
from typing import Callable, Optional

from znnl.models.jax_model import JaxModel

logger = logging.getLogger(__name__)


class Regularizer(ABC):
    """
    Parent class for a regularizer. All regularizers should inherit from this class.
    """

    def __init__(
        self, reg_factor: float, reg_schedule_fn: Optional[Callable] = None
    ) -> None:
        """
        Constructor of the regularizer class.

        Parameters
        ----------
        reg_factor : float
                Regularization factor.
        reg_schedule_fn : Optional[Callable]
                Function to schedule the regularization factor.
                The function takes the current epoch and the regularization factor
                as input and returns the scheduled regularization factor (float).
                An example function is:

                    def reg_schedule(epoch: int, reg_factor: float) -> float:
                        return reg_factor * 0.99 ** epoch

                where the regularization factor is reduced by 1% each epoch.
                The default is None, which means no scheduling is applied:

                    def reg_schedule(epoch: int, reg_factor: float) -> float:
                        return reg_factor
        """
        self.reg_factor = reg_factor
        self.reg_schedule_fn = reg_schedule_fn

        if self.reg_schedule_fn:
            logger.info(
                "Setting a regularization schedule."
                "The set regularization factor will be overwritten."
            )
            if not callable(self.reg_schedule_fn):
                raise TypeError("Regularization schedule must be a Callable.")

        if self.reg_schedule_fn is None:
            self.reg_schedule_fn = self._schedule_fn_default

    @staticmethod
    def _schedule_fn_default(epoch: int, reg_factor: float) -> float:
        """
        Default function for the regularization factor.

        Parameters
        ----------
        epoch : int
                Current epoch.
        reg_factor : float
                Regularization factor.

        Returns
        -------
        scheduled_reg_factor : float
                Scheduled regularization factor.
        """
        return reg_factor

    def _calculate_regularization(self, params: dict, **kwargs) -> float:
        """
        Calculate the regularization contribution to the loss.
        Individual regularizers should implement this function.
        For more information see the docstring of the child class.

        Parameters
        ----------
        params : dict
                Parameters of the model.
        kwargs : dict
                Additional arguments.
                Individual regularizers can utilize arguments from the set:
                    model : JaxModel
                            Model to regularize.
                    batch : dict
                            Batch of data.
                    epoch : int
                            Current epoch.

        Returns
        -------
        reg_loss : float
                Loss contribution from the regularizer.
        """
        raise NotImplementedError

    def __call__(self, model: JaxModel, params: dict, batch: dict, epoch: int) -> float:
        """
        Call function of the regularizer class.

        Parameters
        ----------
        model : JaxModel
                Model to regularize.
        batch : dict
                Batch of data.
        epoch : int
                Current epoch.

        Returns
        -------
        scaled_reg_loss : float
                Scaled loss contribution from the regularizer.
        """
        self.reg_factor = self.reg_schedule_fn(epoch, self.reg_factor)
        return self.reg_factor * self._calculate_regularization(
            model=model, params=params, batch=batch, epoch=epoch
        )
