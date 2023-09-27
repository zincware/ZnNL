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
from znnl.regularizers.regularizer import Regularizer
from typing import Callable
import jax.flatten_util
import jax.tree_util
import jax.numpy as np


class GradVarianceRegularizer(Regularizer):
    """
    Regularizer class to regularize on the variance of the gradients.

    Regularizing the loss of gradient based learning proportional to the variance of the
    gradients, as:
        Var(grad) = E[(grad - E[grad])^2]
    """

    def __init__(self, reg_factor: float = 1e-1) -> None:
        """
        Constructor of the gradient variance regularizer class.

        Parameters
        ----------
        reg_factor : float
                Regularization factor.
        """
        super().__init__(reg_factor) 
    
    def __call__(self, apply_fn: Callable, params: dict, batch: dict) -> float:
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
        grad_variance = jax.tree_util.tree_map(lambda x: np.var(x, axis=(0, 1)), grads)
        raveled_grad_variance = jax.flatten_util.ravel_pytree(grad_variance)[0]
        reg_loss = self.reg_factor * raveled_grad_variance.mean()
        return reg_loss
