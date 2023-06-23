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
import copy
import logging
import time
from typing import List, Optional, Union

import jax.numpy as np
import numpy as onp

import znnl
from znnl.agents.agent import Agent
from znnl.data import DataGenerator
from znnl.distance_metrics.distance_metric import DistanceMetric
from znnl.models import JaxModel
from znnl.point_selection import PointSelection
from znnl.training_strategies.simple_training import SimpleTraining
from znnl.visualization.tsne_visualizer import TSNEVisualizer

logger = logging.getLogger(__name__)


class RND(Agent):
    """
    Class for the implementation of random network distillation.

    Attributes
    ----------
    target : Jax_Model
                Model class for the target network
    predictor : Jax_Model
            Model class for the predictor.
    target_set : list
            Target set to be built iteratively.
    """

    def __init__(
        self,
        data_generator: DataGenerator,
        target_network: JaxModel,
        predictor_network: JaxModel,
        training_strategy: SimpleTraining,
        distance_metric: DistanceMetric = None,
        point_selector: PointSelection = None,
        visualizer: TSNEVisualizer = None,
        tolerance: int = 100,
        seed_point: list = None,
    ):
        """
        Constructor for the RND class.

        Parameters
        ----------
        target_network : Jax_Model
                Model class for the target network
        predictor_network : Jax_Model
                Model class for the predictor.
        training_strategy : SimpleTraining
                Training strategy for training the predictor model.
        distance_metric : SimilarityMeasures
                Metric to use in the representation comparison
        data_generator : objector
                Class to generate or select new points from the point cloud
                being studied.
        point_selector : PointSelection
                Class to select points from the data pool.
        visualizer : TSNEVisualizer
                Class for the representation visualization.
        tolerance : int
                Number of stationary iterations to go through before ending the
                run.
        seed_point : list
                Choose to start with an initial point as seed point
        """
        # User defined attributes.
        self.target = target_network
        self.predictor = predictor_network
        self.training_strategy = training_strategy
        self.training_strategy.model = self.predictor
        self.metric = distance_metric
        self.data_generator = data_generator
        self.point_selector = point_selector
        self.tolerance = tolerance
        self.seed_point = seed_point
        self.visualizer = visualizer

        self.historical_length: int = 0
        self.target_set: list = []
        self.target_indices: list = []
        self.iterations: int = 0
        self.stationary_iterations: int = 0
        self.metric_results = None
        self.target_size: Optional[int] = None
        self.epochs: Optional[Union[int, List[int]]] = None
        self.batch_size: Optional[Union[int, List[int]]] = None
        self.training_kwargs: Optional[dict] = None

        # Run the class initialization
        self._check_and_update_defaults()

    def _check_and_update_defaults(self):
        """
        Set the default arguments if necessary and check for model correct model
        inputs.

        Updates the class state for the following:
           * self.point_selector (default = greedy selection)
           * self.metric (default = cosine similarity)
           * self.visualizer (default = TSNE visualization)

        Checks whether the model input in the training strategy is None.
        Giving feedback otherwise.

        Returns
        -------
        Updates defaults and gives feedback if necessary.
        """
        if self.point_selector is None:
            self.point_selector = znnl.point_selection.GreedySelection()
        if self.metric is None:
            self.metric = znnl.similarity_measures.CosineSim()
        if self.visualizer is None:
            self.visualizer = TSNEVisualizer()
        if self.training_strategy.model is not None:
            logger.info(
                "The model defined in the training strategy will be ignored. "
                "The defined training strategy will operate on the predictor "
                "model. \n "
                "You can set the model in the training strategy to None. "
            )

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
        data = self.fetch_data(-1) # get all points in the pool.
        # Distinguish between points that are already in the target set and others
        data_idx = np.arange(np.shape(data)[0], dtype=int) 
        non_target_idx = np.setdiff1d(data_idx, np.array(self.target_indices))
        unseen_data = data[non_target_idx, ...]
        # Compute distances and select points
        distances = self.compute_distance(unseen_data)
        points = self.point_selector.select_points(distances)
        selected_idx = non_target_idx[points]
        selected_points = data[selected_idx]
        self._update_target_set([i for i in selected_points])
        self.target_indices.extend([int(i) for i in selected_idx])

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
            self.training_strategy.train_model(
                train_ds=dataset,
                test_ds=dataset,
                epochs=self.epochs,
                batch_size=copy.deepcopy(self.batch_size),
                **self.training_kwargs,
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
        epochs: Union[int, list] = None,
        batch_size: Union[int, list] = None,
        **training_kwargs,
    ):
        """
        Run the random network distillation methods and build the target set.

        This method runs the process of constructing a target set by training the
        predictor model using the train_model method of the training strategy.
        Depending on the training strategy, besides epochs and batch_size additional
        kwargs can be used by writing them into training_kwargs.

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
        epochs : Optional[Union[int, List[int]]]
                Epochs to train the predictor model. This argument is passed to the
                train_model method. The defaults are set in the individual training
                strategy.
        batch_size : Optional[Union[int, List[int]]]
                Size of the batch to use in training. This argument is passed to the
                train_model method. The defaults are set in the individual training
                strategy.
        training_kwargs: dict
                Additional kwargs (besides epochs and batch_size) used for the
                train_model method of the according training strategy.
                These kwargs are passed to the train_model method. The defaults are set
                in the individual training strategy.

        Returns
        -------
        target_set : np.ndarray
                Returns the newly constructed target set.
        """
        # Allow for optional target_sizes.
        self.target_size = target_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.training_kwargs = training_kwargs

        start = time.time()
        if seed_randomly:
            self._seed_process(visualize=visualize)
        criteria = False

        if visualize:
            self.update_visualization(reference=True)
            self.update_visualization(reference=False)

        while not criteria:
            self._choose_points()
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
