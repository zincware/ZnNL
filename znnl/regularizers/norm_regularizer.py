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
Module containing the class for a norm regularizer.
"""
from typing import Callable, Optional

import jax.flatten_util
import jax.numpy as np
import jax.tree_util

from znnl.models.jax_model import JaxModel
from znnl.regularizers.regularizer import Regularizer


class NormRegularizer(Regularizer):
    """
    Class to regularize on the norm of the parameters.

    Regularizing training using the norm of the parameters.
    Any function can be used as norm, as long as it takes the parameters as input
    and returns a scalar.
    The function is applied to each parameter
    """

    def __init__(
        self,
        reg_factor: float = 1e-2,
        reg_schedule_fn: Optional[Callable] = None,
        norm_fn: Optional[Callable] = None,
    ) -> None:
        """
        Constructor of the regularizer class.

        Parameters
        ----------
        reg_factor : float
                Regularization factor.
        reg_schedule_fn : Optional[Callable]
                Function to schedule the regularization factor.
                For more information see the docstring of the parent class.
        norm_fn : Callable
                Function to compute the norm of the parameters.
                If None, the default norm is the mean squared error:
                    norm_fn = lambda x: np.mean(x**2)
                Choosing a different norm, such as the L1 norm, can be done by
                passing the function:
                    norm_fn = lambda x: np.mean(np.abs(x))
        """
        super().__init__(reg_factor, reg_schedule_fn)

        self.norm_fn = norm_fn
        if self.norm_fn is None:
            self.norm_fn = lambda x: np.mean(x**2)

    def _calculate_regularization(self, params: dict, **kwargs) -> float:
        """
        Calculate the regularization contribution to the loss using the norm of the
        parameters.

        Parameters
        ----------
        params : dict
                Parameters of the model.
        kwargs : dict
                Additional arguments.

        Returns
        -------
        reg_loss : float
                Loss contribution from the regularizer.
        """
        param_vector = jax.flatten_util.ravel_pytree(params)[0]
        reg_loss = self.reg_factor * self.norm_fn(param_vector)
        return reg_loss
