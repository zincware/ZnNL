"""
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincware Project.

Description: Class for the greedy selection routine.
"""
from znrnd.core.point_selection.point_selection import PointSelection
from znrnd.core.rnd.rnd import RND
import tensorflow as tf


class GreedySelection(PointSelection):
    """
    Class for the greedy selection routine.
    """

    def __init__(
        self, agent: RND = None, selected_points: int = -1, threshold: float = 0.01
    ):
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
        self.agent = agent
        self.threshold = threshold

    def select_points(self):
        """
        Select points from the pool using the greedy algorithm.

        Returns
        -------
        points : tf.Tensor
                A set of points to be used by the RND class.
        """
        data = self.agent.generate_points(-1)  # get all points in the pool.
        distances = self.agent.compute_distance(tf.convert_to_tensor(data))
        truth_sum = len(tf.where(distances > self.threshold))
        if truth_sum > 0:
            return [data[tf.math.argmax(distances)]]
        else:
            return None
