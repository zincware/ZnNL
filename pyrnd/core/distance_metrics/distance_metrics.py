"""
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincware Project.

Description: Module for the distance metrics.
"""
import tensorflow as tf


def euclidean_distance(point_1: tf.Tensor, point_2: tf.Tensor):
    """
    Compute the Euclidean distance metric.

    Parameters
    ----------
    point_1 : tf.Tensor
            First point in the comparison.
    point_2 : tf.Tensor
            Second point in the comparison.
    """
    return tf.norm(point_1 - point_2)
