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
Module for the analysis of the entropy of a matrix.
"""
import jax.numpy as np
from jax import grad, jit


class LossDerivative:
    """
    Class to calculate the derivative of a loss function just in time.
    """

    def __init__(self, loss_fn):
        """
        Constructor for the loss function derivative class.
        """
        self.loss_fn = loss_fn
        self.calculate = jit(grad(self._calculate))

    def _calculate(self, predictions: np.ndarray, targets: np.ndarray):
        """

        Parameters
        ----------
        predictions : np.ndarray
                Predictions made by the network.
        targets : np.ndarray
                Targets from the training data.

        Returns
        -------
        Gradient of the loss function with respect to the predictions.
        """
        return self.loss_fn(predictions, targets)
