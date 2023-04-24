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

from znnl.distance_metrics.wasserstein_distance import WassersteinDistance


class TestWassersteinDistance:
    """
    Class to test the Wasserstein distance measure module.
    """

    @staticmethod
    def create_sample_set():
        """
        Returns
        -------
        Creates a random normal distributed sample set
        """
        point_1 = np.array(
            [onp.random.normal(0, 10, 100), onp.random.normal(0, 20, 100)]
        ).T
        point_2 = np.array(
            [onp.random.normal(0, 10, 100), onp.random.normal(0, 20, 100)]
        ).T
        return point_1, point_2

    def test_dim_operation(self):
        """
        Test if the metric only acts on axis 1 of a given ndarray.

        The metric is based on the scipy implementation of the Wasserstein
        distance. The scipy implementation only takes two one dimensional
        arrays as input. Therefore, the metric should only act on the second
        axis of a given ndarray, which is to be tested here.
        """
        metric = WassersteinDistance()
        point_1, point_2 = self.create_sample_set()

        metric_results = metric(np.array(point_1), np.array(point_2))
        assert metric_results.shape == (100,)
