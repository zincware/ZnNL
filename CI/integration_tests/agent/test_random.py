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
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import jax.numpy as np
import numpy as onp
from numpy.testing import assert_array_equal

from znnl.agents import RandomAgent
from znnl.data import MNISTGenerator


class TestRandomAgent:
    """
    A class to test the random agent.
    """

    def test_selecting_all_data(self):
        """
        Test all data selection.

        This test checks that the build_dataset method returns all the data
        when the number of points to select is equal to the number of points
        in the dataset.
        """
        # Create fake input data for testing
        data_generator = MNISTGenerator(100)
        data_generator.train_ds["inputs"] = np.arange(100)
        data_generator.data_pool = data_generator.train_ds["inputs"]

        point_selector = RandomAgent(data_generator, 0, class_uniform=False)
        selection = point_selector.build_dataset(100)
        print(selection.shape)
        assert len(selection) == 100
        assert np.sum(selection) == np.arange(100).sum()

    def test_selecting_label_uniform(self):
        """
        Test label uniform selection.

        This test checks that the build_dataset method randomly selects the same
        number of points from each class when class_uniform is set to True.
        """
        # Create fake input data for testing
        data_generator = MNISTGenerator(1000)
        data_generator.train_ds["inputs"] = np.arange(1000)

        labels = np.argmax(data_generator.train_ds["targets"], axis=-1)
        print(labels.shape)

        point_selector = RandomAgent(
            data_generator, onp.random.randint(10000), class_uniform=True
        )
        selection = point_selector.build_dataset(100)
        idx = np.array(point_selector.target_indices)

        assert len(selection) == 100

        # Check that the selection is uniform across classes
        assert labels[idx].sum() == np.arange(10).sum() * 10
