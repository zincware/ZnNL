"""
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincware Project.

Description: Test for the similarity measures.
"""
import unittest
import pyrnd
import tensorflow as tf
import numpy as np


class TestSimilarityMeasures(unittest.TestCase):
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
        metric = pyrnd.similarity_measures.cosine_similarity

        # Test orthogonal vectors
        point_1 = tf.convert_to_tensor([1, 0, 0, 0])
        point_2 = tf.convert_to_tensor([0, 1, 0, 0])
        self.assertEqual(metric(point_1, point_2), 1)

        # Test parallel vectors
        point_1 = tf.convert_to_tensor([1, 0, 0, 0])
        point_2 = tf.convert_to_tensor([1, 0, 0, 0])
        self.assertEqual(metric(point_1, point_2), 0)

        # Somewhere in between
        # Test parallel vectors
        point_1 = tf.convert_to_tensor([1.0, 0, 0, 0])
        point_2 = tf.convert_to_tensor([0.5, 1.0, 0, 3.0])
        self.assertEqual(metric(point_1, point_2), 0.84382623)



