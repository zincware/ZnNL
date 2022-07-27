"""
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincware Project.

Description: Test for the similarity measures.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import jax.numpy as np
from numpy.testing import assert_array_almost_equal

from znrnd.core.similarity_measures.cosine_similarity import CosineSim


class TestCosineSimilarity:
    """
    Class to test the similarity measure module.
    """

    def test_cosine_similarity(self):
        """
        Test the cosine similarity measure.

        Returns
        -------
        Assert the correct answer is returned for orthogonal, parallel, and
        somewhere in between.
        """
        metric = CosineSim()

        # Test orthogonal vectors
        point_1 = np.array([[1, 0, 0, 0]])
        point_2 = np.array([[0, 1, 0, 0]])
        assert_array_almost_equal(metric(point_1, point_2), [0])

        # Test parallel vectors
        point_1 = np.array([[1, 0, 0, 0]])
        point_2 = np.array([[1, 0, 0, 0]])
        assert_array_almost_equal(metric(point_1, point_2), [1])

        # Somewhere in between
        point_1 = np.array([[1.0, 0, 0, 0]])
        point_2 = np.array([[0.5, 1.0, 0, 3.0]])
        assert_array_almost_equal(metric(point_1, point_2), [1 - 0.84382623])
