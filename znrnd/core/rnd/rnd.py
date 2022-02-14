"""
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincware Project.

Description: Module for the implementation of random network distillation.
"""
import time
from typing import Union

import numpy as np
import tensorflow as tf

import znrnd
from znrnd.core.data.data_generator import DataGenerator
from znrnd.core.distance_metrics.distance_metric import DistanceMetric
from znrnd.core.models.model import Model
from znrnd.core.point_selection.point_selection import PointSelection
from znrnd.core.visualization.tsne_visualizer import TSNEVisualizer


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
        distance_metric: DistanceMetric = None,
        point_selector: PointSelection = None,
        visualizer: TSNEVisualizer = None,
        optimizers: list = None,
        target_size: int = None,
        tolerance: int = 100,
        seed_point: list = None,
    ):
        """
        Constructor for the RND class.

        Parameters
        ----------
        target_network : Model
                Model class for the target network
        predictor_network : Model
                Model class for the predictor.
        distance_metric : SimilarityMeasures
                Metric to use in the representation comparison
        data_generator : objector
                Class to generate or select new points from the point cloud
                being studied.
        point_selector : PointSelection
                Class to select points from the data pool.
        visualizer : TSNEVisualizer
                Class for the representation visualization.
        optimizers : list
                A list of optimizers to use during the training.
        target_size : int
                A size of a target set you want to enforce
        tolerance : int
                Number of stationary iterations to go through before ending the
                run.
        seed_point : list
                Choose to start with an initial point as seed point
        """
        # User defined attributes.
        self.target = target_network
        self.predictor = predictor_network
        self.metric = distance_metric
        self.generator = data_generator
        self.point_selector = point_selector
        self.optimizers = optimizers
        self.tolerance = tolerance
        self.target_size = target_size
        self.seed_point = seed_point
        self.visualizer = visualizer

        # Class defined attributes
        self.target_set: list = []
        self.historical_length = None
        self.iterations = 0
        self.stationary_iterations = 0
        self.metric_results = None
        self.metric_results_storage: list = []

        # Run the class initialization
        self._set_defaults()

        self.point_selector.agent = self

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
            self.point_selector = znrnd.point_selection.GreedySelection(self)
        # Update the metric
        if self.metric is None:
            self.metric = znrnd.similarity_measures.CosineSim()
        # Update the target
        if self.target is None:
            self.target = znrnd.models.DenseModel()
        # Update the predictor.
        if self.predictor is None:
            self.predictor = znrnd.models.DenseModel()
        if self.visualizer is None:
            self.visualizer = TSNEVisualizer()

    def compute_distance(self, points: tf.Tensor) -> tf.Tensor:
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
        predictor_predictions = self.predictor.predict(points)
        target_predictions = self.target.predict(points)

        self.metric_results = self.metric(target_predictions, predictor_predictions)
        return self.metric_results

    def generate_points(self, n_points: int) -> tf.Tensor:
        """
        Call the data generator and get new data.

        Parameters
        ----------
        n_points : int
                Number of points to generate.

        Returns
        -------
        points : tf.Tensor
                A tensor of data points.
        """
        return self.generator.get_points(n_points)

    def _choose_points(self):
        """
        Call compute distances on all generated points and decide what to
        do with them.
        """
        points = self.point_selector.select_points()
        self._update_target_set(points)

    def _update_target_set(self, points: Union[np.ndarray, None]):
        """
        Add point/s to the target set.

        Parameters
        ----------
        points : np.ndarray
                Array of points to be added to the target set.

        Returns
        -------
        If there are new points the class is updated, if not, nothing happens.
        """
        self.historical_length = len(self.target_set)
        if points is None:
            return
        else:
            for item in points:
                self.target_set.append(list(item))

    def _retrain_network(self):
        """
        Re-train the predictor network.
        """
        if self.historical_length == len(self.target_set):
            pass
        else:
            domain = tf.convert_to_tensor(self.target_set)
            codomain = self.target.predict(domain)

            self.predictor.train_model(domain, codomain)

    def _seed_process(self):
        """
        Seed an RND process.

        Returns
        -------
        Initializes an RND process.
        """
        if self.seed_point:
            seed_point = self.seed_point
        else:
            seed_point = self.generate_points(1)
        self._update_target_set(np.array(seed_point))
        self._retrain_network()

    def _store_metrics(self):
        """
        Storage of metric calculations

        Returns
        -------
        updates the metric storage.
        """
        if len(self.target_set) == self.historical_length:
            pass
        else:
            self.metric_results_storage.append(
                np.sort(self.metric_results.numpy())[-3:]
            )

    def _evaluate_agent(self) -> bool:
        """
        Determine whether or not it is time to stop the searching.

        Returns
        -------
        condition_met : bool
            Will end the search loop if criteria is met.
        """
        # First iteration handling
        if self.historical_length is None:
            return False
        elif self.historical_length == 0:
            pass
        elif len(self.target_set) == self.historical_length:
            if self.stationary_iterations >= self.tolerance:
                return True  # loop timeout
            else:
                self.stationary_iterations += 1
                return False  # update stationary
        elif self.target_size is not None:
            if len(self.target_set) >= self.target_size:
                return True
        # Stationary iteration handling
        else:
            self.stationary_iterations = 0  # reset stationary
            return False

    def update_visualization(self, data: np.ndarray = None, reference: bool = False):
        """
        Update the visualization state.

        Parameters
        ----------
        data : np.ndarray
                Data on which to produce the representation.
        reference : bool


        Returns
        -------
        Updates the visualizer.
        """
        if data is None:
            data = self.generate_points(-1)
        if reference:
            model = self.target
        else:
            model = self.predictor

        model_representations = model.predict(data)
        self.visualizer.build_representation(model_representations, reference=reference)

    def _report_performance(self, time_elapsed: float):
        """
        Provide a brief report on the RND agent performance.

        Parameters
        ----------
        time_elapsed : float
                Amount of time taken to perform the RND.
        """
        print("\nRND agent report")
        print("----------------")
        print(f"Run time: {time_elapsed / 60: 0.2f} m")
        print(f"Size of point cloud: {len(self.generator)}")
        print(f"Number of points chosen: {len(self.target_set)}")
        print(f"Seed points: {self.seed_point}\n")

    def run_rnd(self, visualize: bool = False, report: bool = True):
        """
        Run the random network distillation methods and build the target set.

        Parameters
        ----------
        visualize : bool (default=False)
                If true, a t-SNE visualization will be performed on the final models.
        report : bool (default=True)
                If true, print a report about the RND performance.
        """
        start = time.time()
        self._seed_process()
        criteria = False
        self.update_visualization(reference=True)

        while not criteria:
            self._choose_points()
            self._store_metrics()
            self._retrain_network()
            criteria = self._evaluate_agent()
            self.update_visualization(reference=False)
            self.iterations += 1

        stop = time.time()
        if visualize:
            self.visualizer.run_visualization()

        if report:
            self._report_performance(stop - start)
