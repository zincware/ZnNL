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
Test for the model recording module.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import copy

import numpy as np
import optax
from jax import random
from neural_tangents import stax
from numpy.testing import assert_array_equal

from znrnd.loss_functions import MeanPowerLoss
from znrnd.models import NTModel
from znrnd.training_strategies import LossAwareReservoir, SimpleTraining


class TestLossAwareReservoir:
    """
    Integration test suite for the loss aware reservoir training strategy.
    """

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
        test_ds = train_ds

        # Use a linear network (without activation function) to test the selection
        model = NTModel(
            nt_module=stax.serial(stax.Dense(128), stax.Dense(1)),
            optimizer=optax.adam(learning_rate=0.001),
            input_shape=(1, 1),
        )

        trainer = LossAwareReservoir(
            model=model,
            loss_fn=MeanPowerLoss(order=2),
            reservoir_size=4,
            disable_loading_bar=True,
        )

        # Test if a smaller reservoir selects the correctly sorted points
        _ = trainer.train_model(train_ds=train_ds, test_ds=test_ds, epochs=1)
        selection_idx = np.argsort(np.abs(raw_x))[::-1][:4]
        assert_array_equal(trainer.reservoir["inputs"], x[selection_idx])

    def test_comparison_to_simple_training(self):
        """
        Test the comparison to simple training strategy.

        The loss aware reservoir has to be identical to simple training for choosing the
        reservoir big enough to capture the whole training data.
        """

        # Create data as they will be put into the reservoir
        raw_x = np.linspace(0, 1, 11)[::-1]
        x = np.expand_dims(raw_x, axis=-1)
        y = np.zeros_like(x)
        train_ds = {"inputs": x, "targets": y}
        test_ds = train_ds

        model = NTModel(
            nt_module=stax.serial(stax.Dense(5), stax.Relu(), stax.Dense(1)),
            optimizer=optax.adam(learning_rate=0.001),
            input_shape=(1, 1),
        )

        recursive_trainer = LossAwareReservoir(
            model=copy.deepcopy(model),
            loss_fn=MeanPowerLoss(order=2),
            reservoir_size=500,
            seed=17,
            disable_loading_bar=True,
        )
        reservoir_out = recursive_trainer.train_model(
            train_ds=train_ds,
            test_ds=test_ds,
            epochs=3,
        )
        simple_trainer = SimpleTraining(
            model=copy.deepcopy(model),
            loss_fn=MeanPowerLoss(order=2),
            seed=17,
            disable_loading_bar=True,
        )
        simple_out = simple_trainer.train_model(
            train_ds=train_ds,
            test_ds=test_ds,
            epochs=3,
            batch_size=len(x),
        )
        assert simple_out == reservoir_out
