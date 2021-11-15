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
Module to train an MLP to act as a distance metric.
"""
import tensorflow as tf
from typing import Callable
from pyrnd import DataGenerator
from pyrnd import DenseModel


class MLPMetric:
    """
    Class for the MLP distance metric.
    """
    def __init__(
            self,
            data_generator: DataGenerator,
            distance_metric: Callable,
            embedding_operator: DenseModel
            training_points: int = 100,
            validation_points: int = 100
    ):
        """
        Constructor for the MLP metric.

        Parameters
        ----------
        data_generator : DataGenerator
        distance_metric : Callable
        training_points : int
        validation_points : int
        """
        self.data_generator = data_generator
        self.distance_metric = distance_metric
        self.training_point = training_points
        self.validation_point = validation_points

    def build_model(self):
        """
        Build a neural network model.

        Returns
        -------

        """
        pass

    def train_model