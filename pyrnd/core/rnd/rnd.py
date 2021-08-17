"""
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincware Project.

Description: Module for the implementation of random network distillation.
"""
import pyrnd
from pyrnd.core.models.model import Model
from pyrnd.core.point_selection.point_selection import PointSelection
from pyrnd.core.data.data_generator import DataGenerator
from typing import Callable
import tensorflow as tf
import numpy as np


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

    def __init__(
        self,
        data_generator: DataGenerator,
        target_network: Model = None,
        predictor_network: Model = None,
        distance_metric: Callable = None,
        point_selector: PointSelection = None,
        optimizers: list = None,
        tolerance: int = 100,
    ):
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
        point_selector : PointSelection
                Class to select points from the data pool.
        optimizers : list
                A list of optimizers to use during the training.
        tolerance : int
                Number of stationary iterations to go through before ending the
                run.
        """
        # User defined attributes.
        self.target = target_network
        self.predictor = predictor_network
        self.metric = distance_metric
        self.generator = data_generator
        self.point_selector = point_selector
        self.optimizers = optimizers
        self.tolerance = tolerance

        # Class defined attributes
        self.target_set: list = []
        self.historical_length = None
        self.iterations = 0
        self.stationary_iterations = 0

        # Run the class initialization
        self._set_defaults()

    def _set_defaults(self):
        """
        Set the default parameters if necessary.

        Returns
        -------
        Updates the class state for the following:
           * self.point_selector (default = greedy selection)
           * self.metric (default = cosine similarity)
           * Models (default = dense model.)
        """
        # Update the point selector
        if self.point_selector is None:
            self.point_selector = pyrnd.GreedySelection(self)
        # Update the metric
        if self.metric is None:
            self.metric = pyrnd.similarity_measures.cosine_similarity
        # Update the target
        if self.target is None:
            self.target = pyrnd.DenseModel()
        # Update the predictor.
        if self.predictor is None:
            self.predictor = pyrnd.DenseModel()

    def compute_distance(self, points: tf.Tensor):
        """
        Compute the distance between neural network representations.

        Parameters
        ----------
        points : tf.Tensor
                Points on which distances should be computed.

        Returns
        -------
        distances : tf.Tensor
                A tensor of distances computed using the attached metric.
        """
        predictor_predictions = self.predictor.predict(points)  # returns (4, 12)
        target_predictions = self.target.predict(points)  # returns (100, 12)

        return self.metric(target_predictions, predictor_predictions)

    def generate_points(self, n_points: int):
        """
        Call the data generator and get new data.

        Parameters
        ----------
        n_points : int
                Number of points to generate.

        Returns
        -------

        """
        return self.generator.get_points(n_points)

    def _choose_points(self):
        """
        Call compute distances on all generated points and decide what to
        do with them.

        Returns
        -------

        """
        points = self.point_selector.select_points()
        self._update_target_set(points)

    def _update_target_set(self, points: np.ndarray):
        """
        Add point/s to the target set.

        Parameters
        ----------
        points : np.ndarray
                Array of points to be added to the target set.

        Returns
        -------

        """
        if points is None:
            return
        else:
            for item in points:
                self.target_set.append(list(item))

    def _retrain_network(self):
        """
        Re-train the predictor network.

        Returns
        -------

        """
        domain = tf.convert_to_tensor(self.target_set)
        codomain = self.target.predict(domain)

        #self.predictor.train_model(dataset)

    def _seed_process(self):
        """
        Seed an RND process.

        Returns
        -------
        Initializes an RND process.
        """
        seed_point = self.generate_points(1)
        self._update_target_set(np.array(seed_point))
        self._retrain_network()

    def _evaluate_agent(self):
        """
        Determine whether or not it is time to stop the searching.

        Returns
        -------
        Will end the search loop if criteria is met.
        """
        # First iteration handling
        if self.historical_length is None:
            return False
        # Stationary iteration handling
        elif len(self.target_set) == self.historical_length:
            if self.stationary_iterations >= self.tolerance:
                return True  # loop timeout
            else:
                self.stationary_iterations += 1
                return False  # update stationary
        else:
            self.stationary_iterations = 0  # reset stationary
            return False

    def run_rnd(self):
        """
        Run the random network distillation methods and build the target set.

        Returns
        -------

        """
        self._seed_process()
        criteria = False
        while not criteria:
            self._choose_points()
            self._retrain_network()
            criteria = self._evaluate_agent()
            self.iterations += 1
