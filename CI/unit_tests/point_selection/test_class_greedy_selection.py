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
from numpy.testing import assert_array_equal

from znnl.data import MNISTGenerator
from znnl.point_selection import ClassGreedySelection


class TestClassGreedySelection:
    """
    A class to test the ClassGreedySelection class.
    """

    def test_distance_selection(self):
        """
        Test the select points methods.
        """
        data_generator = MNISTGenerator()
        labels = data_generator.train_ds["targets"][:100]

        point_selector = ClassGreedySelection(labels)

        # Create a distance equivalent to the index of the point
        distances = np.arange(labels.shape[0])

        selected_idx = point_selector.select_points(distances)
        predicted_selection = [l[-1] for l in point_selector.index_list]

        assert_array_equal(predicted_selection, selected_idx)

    def test_selection_for_sparse_points(self):
        """
        Test the select points methods for sparse points.

        If the number of available points is smaller than the number of
        classes, the algorithm should not throw an error.
        """
        data_generator = MNISTGenerator()
        labels = data_generator.train_ds["targets"][:5]

        point_selector = ClassGreedySelection(labels)
        point_selector.select_points(labels)

        # Create a distance equivalent to the index of the point
        distances = np.arange(labels.shape[0])

        # Selecting points should not throw an error
        _ = point_selector.select_points(distances)
