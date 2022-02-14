"""
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincware Project.

Description: Parent class for the models.
"""
import abc

import numpy as np
import tensorflow as tf


class Model:
    """
    Parent class for PyRND Models.
    """

    @abc.abstractmethod
    def predict(self, point: tf.Tensor) -> np.ndarray:
        """
        Make a prediction on a point.

        Parameters
        ----------
        point : tf.Tensor
                Point on which to perform a prediction.

        Returns
        -------
        prediction : np.ndarray
                Model prediction on the point.
        """
        raise NotImplemented("Implemented in child class.")

    def train_model(
        self,
        x: tf.Tensor,
        y: tf.Tensor,
        re_initialize: bool = False,
    ):
        """
        Train the model on data.

        Returns
        -------

        """
        raise NotImplemented("Implemented in child class.")
