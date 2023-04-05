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
Compute the one hot accuracy between two points.
"""
import jax.numpy as np

from znnl.accuracy_functions.accuracy_function import AccuracyFunction


class LabelAccuracy(AccuracyFunction):
    """
    Compute the one hot accuracy between two points.
    """

    def __init__(self):
        """
        Constructor for the one hot accuracy.
        """
        pass

    def __call__(self, predictions: np.array, targets: np.array) -> float:
        """
        Accuracy function call method.

        Parameters
        ----------
        predictions : np.array
                First set of points to be compared.
        targets : np.array
                Second points to compare.

        Returns
        -------
        accuracy : float
                Accuracy of the points.
        """
        return np.mean(np.argmax(predictions, -1) == np.argmax(targets, -1))
