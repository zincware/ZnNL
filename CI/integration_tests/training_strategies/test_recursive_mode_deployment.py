"""
ZnRND: A Zincwarecode package.

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
Integration test for the recursive mode.
"""
import copy
import os

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import jax.random as random
import optax
from neural_tangents import stax

from znrnd.loss_functions import MeanPowerLoss
from znrnd.models import NTModel
from znrnd.training_strategies import (
    LossAwareReservoir,
    PartitionedTraining,
    RecursiveMode,
)


class TestRecursiveMode:
    """
    Unit test suite of the recursive mode.
    """

    @classmethod
    def setup_class(cls):
        """
        Create model and data for the tests.
        """
        network = stax.serial(
            stax.Flatten(), stax.Dense(10), stax.Relu(), stax.Dense(10)
        )
        cls.model = NTModel(
            nt_module=network,
            optimizer=optax.adam(learning_rate=0.05),
            input_shape=(1, 8),
            batch_size=10,
        )
        cls.old_points_trainer = PartitionedTraining(
            model=cls.model,
            loss_fn=MeanPowerLoss(order=2),
        )
        cls.new_points_trainer = LossAwareReservoir(
            model=cls.model,
            loss_fn=MeanPowerLoss(order=2),
            reservoir_size=0,
            latest_points=1,
        )
        mode = RecursiveMode(update_type="rnd")
        cls.old_points_trainer.set_recursive_mode(recursive_mode=copy.deepcopy(mode))
        cls.new_points_trainer.set_recursive_mode(recursive_mode=copy.deepcopy(mode))

        key1, key2 = random.split(random.PRNGKey(1), 2)
        x = random.normal(key1, (5, 8))
        y = np.zeros(shape=(5, 1))
        cls.all_data = {"inputs": x, "targets": y}
        cls.new_data = {"inputs": x[-1:], "targets": y[-1:]}
        cls.old_data = {"inputs": x[:-1], "targets": y[:-1]}

    def test_threshold_update_fn(self):
        """
        Test the threshold update function.

        The condition of passing a threshold in recursive training is tested.
        """
        self.old_points_trainer.recursive_mode.use_recursive_mode = True
        self.old_points_trainer.recursive_mode.threshold = 0.001
        _ = self.old_points_trainer.train_model(
            self.all_data,
            self.old_data,
            batch_size=[10],
            epochs=[6],
            train_ds_selection=[slice(None, None, None)],
        )
        all_data_loss = self.new_points_trainer.evaluate_model(
            self.model.model_state.params, self.all_data
        )
        assert 0.001 >= all_data_loss["loss"]

    def test_rnd_update_fn(self):
        """
        Test the rnd update function.

        Two conditions are tested:
        1. The loss of the last point is below the loss of all other points.
        2. The loss of all other points is below a threshold.

        This is checked by:
        1. Pre-train a model on all points except of the last one. Afterwards, train the
        last point only and check the loss of all points and the last point.
        2. Pre-train a model on the last point only. The train all points and check for
        stopping. The first condition is fulfilled by pre-training here.
        """
        # Check 1. condition

        # Pre-train the model such that all points except of the last have a small loss.
        self.old_points_trainer.recursive_mode.use_recursive_mode = False
        _ = self.old_points_trainer.train_model(
            self.all_data,
            self.new_data,
            batch_size=[10],
            epochs=[10],
            train_ds_selection=[slice(None, -1, None)],
        )
        # Make sure that the old loss is smaller than the new loss now
        old_data_loss = self.new_points_trainer.evaluate_model(
            self.model.model_state.params, self.old_data
        )
        new_data_loss = self.new_points_trainer.evaluate_model(
            self.model.model_state.params, self.new_data
        )
        assert old_data_loss["loss"] < new_data_loss["loss"]

        # Train last point only using the recursive mode.
        # Due to the big threshold, the stopping will be determined by the intersection
        # of losses only.
        self.new_points_trainer.recursive_mode.use_recursive_mode = True
        self.new_points_trainer.recursive_mode.threshold = 100
        _ = self.new_points_trainer.train_model(
            self.all_data,
            self.old_data,
            batch_size=10,
            epochs=1,
        )
        # Check that after stopping the new loss is now smaller than ne old loss.
        old_data_loss = self.new_points_trainer.evaluate_model(
            self.model.model_state.params, self.old_data
        )
        new_data_loss = self.new_points_trainer.evaluate_model(
            self.model.model_state.params, self.new_data
        )
        assert old_data_loss["loss"] >= new_data_loss["loss"]

        # Check 2. condition
        # Pre-train the model with the last point only.
        self.new_points_trainer.recursive_mode.use_recursive_mode = False
        _ = self.new_points_trainer.train_model(
            self.all_data,
            self.old_data,
            batch_size=10,
            epochs=10,
        )

        # Train all points using the recursive mode.
        # Due to the small loss of the last point, the stopping will only be determined
        # by the threshold.
        self.old_points_trainer.recursive_mode.threshold = 0.001
        self.old_points_trainer.recursive_mode.use_recursive_mode = True
        _ = self.old_points_trainer.train_model(
            self.all_data,
            self.all_data,
            batch_size=[10],
            epochs=[10],
            train_ds_selection=[slice(None, None, None)],
        )
        all_data_loss = self.new_points_trainer.evaluate_model(
            self.model.model_state.params, self.all_data
        )
        assert 0.001 >= all_data_loss["loss"]
