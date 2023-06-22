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
Greedy selection algorithm for classification tasks in RND. 
"""
from typing import List, Union

import jax.numpy as np

from znnl.point_selection.point_selection import PointSelection


class ClassGreedySelection(PointSelection):
    """
    Class for the greedy selection routine for classification tasks.
    """

    def __init__(self, labels: np.ndarray):
        """
        Constructor for the GreedySelection class for classification tasks.

        Parameters
        ----------
        labels : np.ndarray
                Labels of the data pool in one-hot encoding.
        """
        super(ClassGreedySelection, self).__init__()
        self.labels = labels
        self.drawn_points = -1  # draw all points

        self._create_index_lists()

    def _create_index_lists(self):
        """
        Create an index lists mapping instances of the data pool to the
        according label.

        Returns
        -------
        Creates an index list for each label.
        All index lists are stored in self.index_list.
        """
        index_lists = []
        labels = np.argmax(self.labels, axis=-1)
        for i in range(np.shape(self.labels)[1]):
            index_lists.append(np.where(labels == i)[0])
        self.index_list = index_lists

    def select_points(self, distances: np.ndarray) -> Union[List, None]:
        """
        Select points from the pool using the greedy algorithm.

        Select points of maximum distance of each class.
        If no points are available for a class, no points are selected.

        Returns
        -------
        points : np.array
                A set of indices of points to be used by the RND class.
        """
        max_distances = []
        for l in self.index_list:
            class_distances = np.take(distances, l, axis=0)
            if np.shape(class_distances)[0] == 0:
                pass
            else:
                class_list_idx = np.argmax(class_distances)
                max_distances.append(l[class_list_idx])
        return np.array(max_distances, dtype=int)
