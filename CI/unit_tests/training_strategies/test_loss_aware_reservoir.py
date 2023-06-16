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
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import optax
from flax import linen as nn
from jax import random
from neural_tangents import stax
from numpy.testing import assert_array_equal

from znnl.loss_functions import MeanPowerLoss
from znnl.models import FlaxModel, NTModel
from znnl.training_strategies import LossAwareReservoir


class FlaxArchitecture(nn.Module):
    """
    Test model for the Flax tests.
    """

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(5, use_bias=True)(x)
        x = nn.relu(x)
        x = nn.Dense(features=1, use_bias=True)(x)
        return x


class TestPartitionedSelection:
    """
    Unit test suite of the loss aware reservoir training strategy.
    """

    @classmethod
    def setup_class(cls):
        """
        Create models for the tests.
        """
        cls.nt_model = NTModel(
            nt_module=stax.serial(stax.Dense(5), stax.Relu(), stax.Dense(1)),
            optimizer=optax.adam(learning_rate=0.001),
            input_shape=(1, 8),
        )
        cls.flax_model = FlaxModel(
            flax_module=FlaxArchitecture(),
            optimizer=optax.adam(learning_rate=0.001),
            input_shape=(1, 8),
        )

    def test_reservoir_sorting(self):
        """
        Test the sorting of the reservoir.

        This test asserts if
            * the size of the reservoir is the selected one
            * the reservoir sorts correctly
        """

        # Create linearly spaced data from -1 to 1 and shuffle them
        key1, key2 = random.split(random.PRNGKey(1), 2)
        raw_x = random.permutation(key1, np.linspace(0, 1, 11), axis=0)
        x = np.expand_dims(raw_x, axis=-1)
        y = np.zeros_like(x)
        train_ds = {"inputs": x, "targets": y}

        # Use a linear network (without activation function) to test the selection
        model = self.nt_model

        trainer = LossAwareReservoir(
            model=model,
            loss_fn=MeanPowerLoss(order=2),
            reservoir_size=4,
            disable_loading_bar=True,
            latest_points=0,
        )

        # Test if a smaller reservoir selects the correctly sorted points
        trainer.train_data_size = len(train_ds["inputs"])
        reservoir = trainer._update_reservoir(train_ds=train_ds)
        selection_idx = np.argsort(np.abs(raw_x))[::-1][:4]
        assert_array_equal(reservoir, selection_idx)

    def test_update_reservoir(self):
        """
        Test the method _update_reservoir.

        Test whether the method excludes the latest points from train_ds.
        When selecting latest_points > 0, this number of points is separated from
        the train data. The selected points will be appended to every batch.
        This test checks if the method _update_reservoir removes the latest_points
        from the data, as they cannot be part of the reservoir. The reservoir must
        only consist of already seen data.
            1. For reservoir_size = len(train_ds)
                * Shrinking reservoir for latest_points = 1
                * Shrinking reservoir for latest_points = 4
                * Shrink the reservoir to include not points for latest_points = 10
            2. For reservoir_size = 5 and len(train_ds) = 10
                * No shrinking reservoir size for latest_points = 4

        Perform both tests for nt and flax models.
        """

        # Create some test data
        key1, key2 = random.split(random.PRNGKey(1), 2)
        x = random.normal(key1, (10, 8))
        y = random.normal(key1, (10, 1))
        train_ds = {"inputs": x, "targets": y}

        nt_model = self.nt_model
        flax_model = self.flax_model

        # Test for latest_points = 1
        nt_trainer = LossAwareReservoir(
            model=nt_model,
            loss_fn=MeanPowerLoss(order=2),
            disable_loading_bar=True,
            reservoir_size=10,
            latest_points=1,
        )
        flax_trainer = LossAwareReservoir(
            model=flax_model,
            loss_fn=MeanPowerLoss(order=2),
            disable_loading_bar=True,
            reservoir_size=10,
            latest_points=1,
        )
        nt_trainer.train_data_size = len(x)
        flax_trainer.train_data_size = len(x)
        reservoir = nt_trainer._update_reservoir(train_ds=train_ds)
        assert len(x) - 1 == len(reservoir)
        reservoir = flax_trainer._update_reservoir(train_ds=train_ds)
        assert len(x) - 1 == len(reservoir)

        # Test for latest_points = 4
        nt_trainer = LossAwareReservoir(
            model=nt_model,
            loss_fn=MeanPowerLoss(order=2),
            disable_loading_bar=True,
            reservoir_size=10,
            latest_points=4,
        )
        flax_trainer = LossAwareReservoir(
            model=flax_model,
            loss_fn=MeanPowerLoss(order=2),
            disable_loading_bar=True,
            reservoir_size=10,
            latest_points=4,
        )

        nt_trainer.train_data_size = len(x)
        flax_trainer.train_data_size = len(x)
        reservoir = nt_trainer._update_reservoir(train_ds=train_ds)
        assert len(x) - 4 == len(reservoir)
        reservoir = flax_trainer._update_reservoir(train_ds=train_ds)
        assert len(x) - 4 == len(reservoir)

        # Test for latest_points = 10
        nt_trainer = LossAwareReservoir(
            model=nt_model,
            loss_fn=MeanPowerLoss(order=2),
            disable_loading_bar=True,
            reservoir_size=10,
            latest_points=10,
        )
        flax_trainer = LossAwareReservoir(
            model=flax_model,
            loss_fn=MeanPowerLoss(order=2),
            disable_loading_bar=True,
            reservoir_size=10,
            latest_points=10,
        )

        nt_trainer.train_data_size = len(x)
        flax_trainer.train_data_size = len(x)
        reservoir = nt_trainer._update_reservoir(train_ds=train_ds)
        assert 0 == len(reservoir)
        reservoir = flax_trainer._update_reservoir(train_ds=train_ds)
        assert 0 == len(reservoir)

        # Test for latest_points = 2 but for reservoir_size = 5. The reservoir size
        # should not be affected now.
        nt_trainer = LossAwareReservoir(
            model=nt_model,
            loss_fn=MeanPowerLoss(order=2),
            disable_loading_bar=True,
            reservoir_size=5,
            latest_points=4,
        )
        flax_trainer = LossAwareReservoir(
            model=flax_model,
            loss_fn=MeanPowerLoss(order=2),
            disable_loading_bar=True,
            reservoir_size=5,
            latest_points=4,
        )

        nt_trainer.train_data_size = len(x)
        flax_trainer.train_data_size = len(x)
        reservoir = nt_trainer._update_reservoir(train_ds=train_ds)
        assert 5 == len(reservoir)
        reservoir = flax_trainer._update_reservoir(train_ds=train_ds)
        assert 5 == len(reservoir)
