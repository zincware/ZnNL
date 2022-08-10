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
Test the data generator module.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import jax.numpy as np
from numpy.testing import assert_array_equal

from znrnd.data import DataGenerator


class TestDataGenerator:
    """
    Test suite for the data generator.
    """

    data_generator: DataGenerator = None

    @classmethod
    def setup_class(cls):
        """
        Prepare the test suite.
        """
        cls.data_generator = DataGenerator()
        cls.data_generator.data_pool = np.array([1, 2, 3, 4, 5, 6])

    def test_len_dunder(self):
        """
        Test the __len__ dunder method.
        """
        len(self.data_generator) == 6

    def test_get_dunder(self):
        """
        Test that one can get items from the class.
        """
        for i in range(6):
            self.data_generator[i] == i + 1

    def test_get_random(self):
        """
        Test collection of 3 random points.
        """
        dataset = self.data_generator.get_points(3, method="random")
        # TODO: enforce seed and check value-wise
        len(dataset) == 3

    def test_get_first(self):
        """
        Test collecting the first points.
        """
        dataset = self.data_generator.get_points(3, method="first")
        assert_array_equal(dataset, [1, 2, 3])
        len(dataset) == 3

    def test_get_uniform(self):
        """
        Test collecting 3 points uniformly.
        """
        dataset = self.data_generator.get_points(3, method="uniform")
        assert_array_equal(dataset, [1, 3, 6])
        len(dataset) == 3

    def test_full_data_return(self):
        """
        Test both ways of returning the full dataset.

        Notes
        -----
        Tests asking for too many points and the -1 argument.
        """
        dataset = self.data_generator.get_points(-1)
        assert_array_equal(dataset, [1, 2, 3, 4, 5, 6])
        dataset = self.data_generator.get_points(10)
        assert_array_equal(dataset, [1, 2, 3, 4, 5, 6])
