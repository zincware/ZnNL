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

import jax.numpy as np

import znrnd
from znrnd.jax_core.data.data_generator import DataGenerator
from znrnd.jax_core.distance_metrics.distance_metric import DistanceMetric
from znrnd.jax_core.models.model import Model
from znrnd.jax_core.point_selection.point_selection import PointSelection
from znrnd.jax_core.visualization.tsne_visualizer import TSNEVisualizer


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

    historical_length: int = 0
    target_set: list = []
    iterations: int = 0
    stationary_iterations: int = 0
    metric_results = None
    metric_results_storage: list = []

    def __init__(
        self,
        data_generator: DataGenerator,
        target_network: Model = None,
        predictor_network: Model = None,
        distance_metric: DistanceMetric = None,
        point_selector: PointSelection = None,
        visualizer: TSNEVisualizer = None,
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
        self.tolerance = tolerance
        self.target_size = target_size
        self.seed_point = seed_point
        self.visualizer = visualizer

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
        if self.point_selector is None:
            self.point_selector = znrnd.point_selection.GreedySelection(self)
        if self.metric is None:
            self.metric = znrnd.similarity_measures.CosineSim()
        if self.visualizer is None:
            self.visualizer = TSNEVisualizer()

    def compute_distance(self, points: np.ndarray) -> np.ndarray:
        """
        Compute the distance between neural network representations.

        Parameters
        ----------
        points : np.ndarray
                Points on which distances should be computed.

        Returns
        -------
        distances : np.ndarray
                A tensor of distances computed using the attached metric.
        """
        # TODO: Make models do this with a call method.
        predictor_predictions = self.predictor.predict(points)
        target_predictions = self.target.predict(points)

        self.metric_results = self.metric(target_predictions, predictor_predictions)
        return self.metric_results

    def fetch_data(self, n_points: int) -> np.ndarray:
        """
        Call the data generator and get new data.

        Parameters
        ----------
        n_points : int
                Number of points to generate.

        Returns
        -------
        points : np.ndarray
                A tensor of data points.
        """
        return self.generator.get_points(n_points)

    def _choose_points(self):
        """
        Call compute distances on all generated points and decide what to
        do with them.
        """
        data = self.fetch_data(-1)  # get all points in the pool.
        distances = self.compute_distance(np.array(data))
        points = self.point_selector.select_points(distances)
        selected_points = data[points]
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
            domain = np.array(self.target_set)
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
            seed_point = self.fetch_data(1)
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
        if self.historical_length == 0:
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
            self.stationary_iterations = 0  # reset stationary iterations
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
            data = self.fetch_data(-1)
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
