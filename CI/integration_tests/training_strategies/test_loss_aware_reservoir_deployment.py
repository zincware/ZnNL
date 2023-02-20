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
Test for the loss aware reservoir training strategy.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import copy

import numpy as np
import optax
from neural_tangents import stax
from numpy.testing import assert_array_almost_equal, assert_array_equal

from znrnd.loss_functions import MeanPowerLoss
from znrnd.models import NTModel
from znrnd.training_strategies import LossAwareReservoir, SimpleTraining


class TestLossAwareReservoir:
    """
    Integration test suite for the loss aware reservoir training strategy.
    """

    @classmethod
    def setup_class(cls):
        """
        Create data for the tests.
        """

        raw_x = np.linspace(1, 0, 11)
        x = np.expand_dims(raw_x, axis=-1)
        y = np.zeros_like(x)
        cls.train_ds = {"inputs": x, "targets": y}
        cls.test_ds = {"inputs": x, "targets": y}

    def test_comparison_to_simple_training(self):
        """
        Test the equivalence of loss aware reservoir and simple training.

        The loss aware reservoir has to be identical to simple training for choosing the
        reservoir big enough to capture the whole training data.
        Also compare the random number generators after training. They should have
        generated the same number of keys. Given the same seed they have to be
        identical when calling them again.
        """

        model = NTModel(
            nt_module=stax.serial(stax.Dense(5), stax.Relu(), stax.Dense(1)),
            optimizer=optax.adam(learning_rate=0.001),
            input_shape=(1, 1),
        )

        lar_trainer = LossAwareReservoir(
            model=copy.deepcopy(model),
            loss_fn=MeanPowerLoss(order=2),
            reservoir_size=500,
            seed=17,
            disable_loading_bar=True,
            latest_points=0,
        )
        reservoir_out = lar_trainer.train_model(
            train_ds=self.train_ds,
            test_ds=self.test_ds,
            epochs=3,
        )
        lar_rng = lar_trainer.rng()
        simple_trainer = SimpleTraining(
            model=copy.deepcopy(model),
            loss_fn=MeanPowerLoss(order=2),
            seed=17,
            disable_loading_bar=True,
        )
        simple_out = simple_trainer.train_model(
            train_ds=self.train_ds,
            test_ds=self.test_ds,
            epochs=3,
        )
        simple_rng = simple_trainer.rng()

        assert simple_out == reservoir_out
        assert_array_equal(simple_rng, lar_rng)

    def test_comparison_to_simple_training_latest_points(self):
        """
        Test the equivalence of loss aware reservoir and simple training using only the
        latest points.

        Using latest_points, the user can select points that will be trained in every
        epoch.
        Selecting all data into latest_points will lead to a simple training of those
        points (but without shuffling the data).
        """

        model = NTModel(
            nt_module=stax.serial(stax.Dense(5), stax.Relu(), stax.Dense(1)),
            optimizer=optax.adam(learning_rate=0.001),
            input_shape=(1, 1),
        )

        lar_trainer = LossAwareReservoir(
            model=copy.deepcopy(model),
            loss_fn=MeanPowerLoss(order=2),
            reservoir_size=500,
            seed=17,
            disable_loading_bar=True,
            latest_points=11,
        )
        reservoir_out = lar_trainer.train_model(
            train_ds=self.train_ds,
            test_ds=self.test_ds,
            epochs=5,
        )
        simple_trainer = SimpleTraining(
            model=copy.deepcopy(model),
            loss_fn=MeanPowerLoss(order=2),
            seed=17,
            disable_loading_bar=True,
        )
        simple_out = simple_trainer.train_model(
            train_ds=self.train_ds,
            test_ds=self.test_ds,
            epochs=5,
        )

        for key in simple_out.keys():
            assert_array_almost_equal(simple_out[key], reservoir_out[key])
