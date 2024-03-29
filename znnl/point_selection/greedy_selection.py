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

from typing import List, Union

import jax.numpy as np

from znnl.point_selection.point_selection import PointSelection


class GreedySelection(PointSelection):
    """
    Class for the greedy selection routine.
    """

    def __init__(self, selected_points: int = -1, threshold: float = 0.01):
        """
        Constructor for the GreedySelection class.

        Parameters
        ----------
        agent (default = None): RND
                An agent used to select and operate on points.
        selected_points (default = -1) : int
                Number of points to be selected by the algorithm.
        threshold : float
                A value after which points are considered far away.
        """
        super(GreedySelection, self).__init__()
        self.drawn_points = -1  # draw all points
        self.selected_points = selected_points
        self.threshold = threshold

    def select_points(self, distances: np.ndarray) -> Union[List, None]:
        """
        Select points from the pool using the greedy algorithm.

        Returns
        -------
        points : list
                A set of indices of points to be used by the RND class.
        """
        truth_sum = len(np.where(distances > self.threshold))
        if truth_sum > 0:
            return np.argmax(distances)
        else:
            return None
