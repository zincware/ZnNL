"""
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincware Project.

Description: Test for the dense model module.
"""
import unittest
import pyrnd
import tensorflow as tf
import numpy as np


class TestDenseModels(unittest.TestCase):
    """
    Class to test the dense model module.
    """

    def test_build_model(self):
        """
        Test that the model has been built correctly.

        Returns
        -------
        Assess that all parameters are correctly initialized. At the moment
        this is just the number of layers.

        Notes
        -----
        The number of layers is n_hidden + 1.
        """
        # test default parameters
        model = pyrnd.DenseModel()
        self.assertEqual(len(model.model.layers), 5)
        # Test custom less
        model = pyrnd.DenseModel(layers=2)
        self.assertEqual(len(model.model.layers), 3)
        # Test custom more
        model = pyrnd.DenseModel(layers=9)
        self.assertEqual(len(model.model.layers), 10)

    def test_train_model(self):
        """
        Test the model training.

        Returns
        -------
        Ensure that the model can train under different epoch numbers and that
        the training exit occurs after the required loss is achieved.
        """
        model = pyrnd.DenseModel(layers=2, tolerance=2)

        inputs = np.array([[1, 2], [4, 5], [3, 6], [9, 7], [3, 4], [8, 8]])
        labels = np.log(np.prod(inputs, axis=1)).reshape(6, 1)

        model.train_model(tf.convert_to_tensor(inputs), tf.convert_to_tensor(labels))
