"""
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincware Project.

Description: Module for the implementation of random network distillation.
"""
import time
from typing import Any, Callable, Union

import jax.numpy as np
import numpy as onp
from jax.random import PRNGKeyArray

import znrnd
from znrnd.agents.agent import Agent
from znrnd.data import DataGenerator
from znrnd.distance_metrics.distance_metric import DistanceMetric
from znrnd.models import Model
from znrnd.point_selection import PointSelection
from znrnd.visualization.tsne_visualizer import TSNEVisualizer


class RND(Agent):
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
        epochs: int = 10,
        tolerance: int = 100,
        seed_point: list = None,
        disable_loading_bar: bool = False,
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
        epochs: int
                Number of epochs to train the model per recursion.
        tolerance : int
                Number of stationary iterations to go through before ending the
                run.
        seed_point : list
                Choose to start with an initial point as seed point
        disable_loading_bar : bool
                Disable the output visualization of the loading par.
        """
        # User defined attributes.
        self.target = target_network
        self.predictor = predictor_network
        self.metric = distance_metric
        self.data_generator = data_generator
        self.point_selector = point_selector
        self.tolerance = tolerance
        self.seed_point = seed_point
        self.visualizer = visualizer
        self.epochs = epochs
        self.disable_loading_bar = disable_loading_bar

        self.historical_length: int = 0
        self.target_set: list = []
        self.target_indices: list = []
        self.iterations: int = 0
        self.stationary_iterations: int = 0
        self.metric_results = None
        self.metric_results_storage: list = []
        self.target_size: int = None

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
            self.point_selector = znrnd.point_selection.GreedySelection()
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
        predictor_predictions = self.predictor(points)
        target_predictions = self.target(points)

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
        return self.data_generator.get_points(n_points)

    def _choose_points(self):
        """
        Call compute distances on all generated points and decide what to
        do with them.
        """
        data = self.fetch_data(-1)  # get all points in the pool.
        distances = self.compute_distance(np.array(data))
        points = self.point_selector.select_points(distances)
        selected_points = data[points]
        self._update_target_set([selected_points])
        self.target_indices.append(int(points))

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
            codomain = self.target(domain)
            dataset = {"inputs": domain, "targets": codomain}
            self.predictor.train_model_recursively(
                train_ds=dataset,
                test_ds=dataset,
                disable_loading_bar=self.disable_loading_bar,
                epochs=self.epochs,
            )

    def _seed_process(self, visualize: bool):
        """
        Seed an RND process.

        Parameters
        ----------
        visualize : bool
                if true, visualization is performed on the data. This parameter is
                populated by the build_dataset method.

        Returns
        -------
        Initializes an RND process.
        """
        if self.seed_point:
            seed_point = self.seed_point
            # TODO get seed number
        else:
            # seed_point = self.fetch_data(1)
            seed_number = onp.random.randint(0, len(self.data_generator))
            seed_point = self.data_generator.take_index(seed_number)

        self._update_target_set(np.array(seed_point))
        self._retrain_network()
        self.target_indices.append(seed_number)
        if visualize:
            self.update_visualization(reference=False)

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
            self.metric_results_storage.append(np.sort(self.metric_results)[-3:])

    def _evaluate_agent(self) -> bool:
        """
        Determine whether or not it is time to stop the searching.

        Returns
        -------
        condition_met : bool
            Will end the search loop if criteria is met.
        """
        condition = False

        # Check if the target set is the correct size
        if self.target_size is not None:
            if len(self.target_set) >= self.target_size:
                condition = True

        # Check if timeout condition is met
        if self.historical_length > 0:
            if len(self.target_set) == self.historical_length:
                if self.stationary_iterations >= self.tolerance:
                    condition = True
                else:
                    self.stationary_iterations += 1

            else:
                self.stationary_iterations = 0  # reset stationary iterations

        return condition

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

        model_representations = model(data)
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
        print(f"Size of point cloud: {len(self.data_generator)}")
        print(f"Number of points chosen: {len(self.target_set)}")
        print(f"Seed points: {self.seed_point}\n")

    def build_dataset(
        self,
        target_size: int = None,
        seed_randomly: bool = False,
        visualize: bool = False,
        report: bool = False,
        store_metrics: bool = False,
    ):
        """
        Run the random network distillation methods and build the target set.

        Parameters
        ----------
        target_size : int
                Target size of the operation.
        seed_randomly : bool:
                If true, the RND process is seeded by a particular point.
                If false, the RND process chooses the first point according to a metric.
        visualize : bool (default=False)
                If true, a t-SNE visualization will be performed on the final models.
        report : bool (default=True)
                If true, print a report about the RND performance.
        store_metrics : bool (default=True)
                If true, store the RND metrics.

        Returns
        -------
        target_set : np.ndarray
                Returns the newly constructed target set.
        """
        # Allow for optional target_sizes.
        self.target_size = target_size
        start = time.time()
        if seed_randomly:
            self._seed_process(visualize=visualize)
        criteria = False

        if visualize:
            self.update_visualization(reference=True)
            self.update_visualization(reference=False)

        while not criteria:
            self._choose_points()
            if store_metrics:
                self._store_metrics()
            self._retrain_network()

            criteria = self._evaluate_agent()
            if visualize:
                self.update_visualization(reference=False)
            self.iterations += 1

        stop = time.time()
        if visualize:
            self.visualizer.run_visualization()

        if report:
            self._report_performance(stop - start)

        return np.array(self.target_set)
