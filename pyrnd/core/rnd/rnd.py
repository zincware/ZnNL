"""
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincware Project.

Description: Module for the implementation of random network distillation.
"""
from pyrnd.core.models.model import Model


class RND:
    """
    Class for the implementation of random network distillation.

    Attributes
    ----------
    target : Model
                Model class for the target network
    predictor : Model
            Model class for the predictor.
    target_set : list
            Target set to be built iteratively.
    """

    def __init__(self,
                 target_network: Model,
                 predictor_network: Model,
                 distance_metric: object,
                 data_generator: object,
                 optimizers: list = None):
        """
        Constructor for the RND class.

        Parameters
        ----------
        target_network : Model
                Model class for the target network
        predictor_network : Model
                Model class for the predictor.
        distance_metric : object
                Metric to use in the representation comparison
        data_generator : objector
                Class to generate or select new points from the point cloud
                being studied.
        optimizers : list
                A list of optimizers to use during the training.
        """
        # User defined attributes.
        self.target = target_network
        self.predictor = predictor_network
        self.metric = distance_metric
        self.generator = data_generator
        self.optimizers = optimizers

        # Class defined attributes
        self.target_set: list = []

    def _compute_distance(self):
        """
        Compute the distance between neural network representations.

        Returns
        -------

        """
        pass

    def _generate_points(self):
        """
        Call the data generator and get new data.

        Returns
        -------

        """
        pass

    def _choose_points(self):
        """
        Call compute distances on all generated points and decide what to
        do with them.

        Returns
        -------

        """
        pass

    def _update_target_set(self):
        """
        Add point/s to the target set.

        Returns
        -------

        """
        pass

    def _retrain_network(self):
        """
        Re-train the predictor network.

        Returns
        -------

        """
        pass

    def run_rnd(self):
        """
        Run the random network distillation methods and build the target set.

        Returns
        -------

        """
        # Seed the process
        # Select new point/s
        # Check distances
        # Choose points and update target set
        # Re-train model
        # Select new point.
