"""
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincware Project.

Description: Module for the similarity measures.

Notes
-----
These similarity measures will return 1 - S(A, B). This is because we need
a quasi-distance for the comparison to occur.
"""
import tensorflow as tf


def cosine_similarity(point_1: tf.Tensor, point_2: tf.Tensor):
    """
    Parameters
    ----------
    point_1 : tf.Tensor
            First point in the comparison.
    point_2 : tf.Tensor
            Second point in the comparison.
    """
    numerator = tf.cast(tf.tensordot(point_1, point_2, 1), tf.float32)
    denominator = tf.sqrt(tf.cast(tf.tensordot(point_1, point_1, 1) *
                          tf.tensordot(point_2, point_2, 1), tf.float32))
    return 1 - tf.divide(numerator, denominator)
