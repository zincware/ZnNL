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
Test the dense model module.
"""
import unittest
import znrnd
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
        model = znrnd.DenseModel()
        self.assertEqual(len(model.model.layers), 5)
        # Test custom less
        model = znrnd.DenseModel(layers=2)
        self.assertEqual(len(model.model.layers), 3)
        # Test custom more
        model = znrnd.DenseModel(layers=9)
        self.assertEqual(len(model.model.layers), 10)
