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
Module containing the trace regularizer class.
"""
from typing import Callable

import jax.flatten_util
import jax.tree_util

from znnl.regularizers.regularizer import Regularizer


class TraceRegularizer(Regularizer):
    """
    Trace regularizer class.

    Regularizing the loss of gradient based learning proportional to the trace of the
    NTK. As:
        Trace(NTK) = sum_i (d f(x_i)/d theta)^2
    the trace of the NTK is the sum of the squared gradients of the model, the trace
    regularizer is equivalent to regularizing on the sum of the squared gradients of
    the model.
    """

    def __init__(
        self, reg_factor: float = 1e-1, reg_schedule_fn: Callable = None
    ) -> None:
        """
        Constructor of the trace regularizer class.

        Parameters
        ----------
        reg_factor : float
                Regularization factor.
        reg_schedule_fn : Callable

        """
        super().__init__(reg_factor, reg_schedule_fn)

    def _calculate_regularization(
        self, apply_fn: Callable, params: dict, batch: dict, epoch: int
    ) -> float:
        """
        Call function of the trace regularizer class.

        Parameters
        ----------
        apply_fn : Callable
                Function to apply the model to inputs.
        params : dict
                Parameters of the model.
        batch : dict
                Batch of data.

        Returns
        -------
        reg_loss : float
                Loss contribution from the regularizer.
        """
        # Compute squared gradient of shape=(batch_size, n_outputs, params)
        grads = jax.jacrev(apply_fn)(params, batch["inputs"])
        # Square the gradients and take the mean over the batch
        squared_grads = jax.flatten_util.ravel_pytree(grads)[0] ** 2
        reg_loss = self.reg_factor * squared_grads.mean()
        return reg_loss
