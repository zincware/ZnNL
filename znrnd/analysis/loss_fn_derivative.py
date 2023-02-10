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
Module for the calculating the derivative of a loss function.
"""
import jax.numpy as np
from jax import grad, jit

from znrnd.loss_functions import SimpleLoss


class LossDerivative:
    """
    Class to calculate the derivative of a loss function just in time.

    The gradient calculation is performed with respect to the predictions by calling
    LossDerivative.calculate().

    The loss function is a map from a high dimensional space to a reel number.
    The derivative here, is therefore a gradient calculation of the loss function with
    respect to its input.
    The calculation returns a gradient of size n, when n is the dimension of an input
    of the loss function.
    """

    def __init__(self, loss_fn: SimpleLoss):
        """
        Constructor for the loss function derivative class.

        The gradient calculation of a loss function is performed by calling
        self.calculate.

        Parameters
        ----------
        loss_fn : SimpleLoss
                Loss function to calculate the derivative of.
        """
        self.loss_fn = loss_fn
        self.calculate = jit(grad(self._calculate))

    def _calculate(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Method for calculating the loss of given inputs.

        This method is introduced in order to be jit compiled.
        For the actual gradient computation self.calculate has to be called.

        Parameters
        ----------
        predictions : np.ndarray
                Predictions made by the network.
        targets : np.ndarray
                Targets from the training data.

        Returns
        -------
        loss : float
        """
        return self.loss_fn(predictions, targets)
