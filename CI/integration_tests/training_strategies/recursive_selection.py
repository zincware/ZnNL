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

import optax
from jax import random
from neural_tangents import stax
from numpy.testing import assert_raises

from znrnd.loss_functions import MeanPowerLoss
from znrnd.models import NTModel
from znrnd.training_strategies import RecursiveSelection, SimpleTraining


class TestRecursiveSelection:
    """
    Unit test suite for the recursive selection training strategy.
    """

    def test_parameter_setting(self):
        """
        Test the initialization of the training strategy.
        """
        key1, key2 = random.split(random.PRNGKey(1), 2)
        x = random.normal(key1, (3, 8))
        y = random.normal(key1, (3, 1))
        train_ds = {"inputs": x, "targets": y}
        test_ds = train_ds

        model = NTModel(
            nt_module=stax.serial(stax.Dense(5), stax.Relu(), stax.Dense(1)),
            optimizer=optax.adam(learning_rate=0.001),
            input_shape=(1, 8),
        )

        trainer = RecursiveSelection(
            model=model,
            loss_fn=MeanPowerLoss(order=2),
            disable_loading_bar=True,
        )

        # Check if default train for 200 epochs
        batch_metric = trainer.train_model(
            train_ds=train_ds, test_ds=test_ds,
        )
        assert len(batch_metric["train_losses"]) == 200

        # Check optional kwargs
        batch_metric = trainer.train_model(
            train_ds=train_ds,
            test_ds=test_ds,
            epochs=[2, 5],
            train_ds_selection=[[-1], slice(1, -1, 3)],
        )
        assert len(batch_metric["train_losses"]) == 7

        # Check Error handling for not matching lengths
        assert_raises(KeyError, trainer.train_model, train_ds, test_ds, [2])

        # Check Error handling for passing an int for epochs instead of a list
        assert_raises(KeyError, trainer.train_model, train_ds, test_ds, 50)

    def test_comparison_to_simple_training(self):
        """
        Test the comparison to simple training strategy.

        The recursive selection has to be identical to simple training for selecting
        train_ds_selection=[slice(None, None, None)]
        """

        key1, key2 = random.split(random.PRNGKey(1), 2)
        x = random.normal(key1, (3, 8))
        y = random.normal(key1, (3, 1))
        train_ds = {"inputs": x, "targets": y}
        test_ds = train_ds

        model = NTModel(
            nt_module=stax.serial(stax.Dense(5), stax.Relu(), stax.Dense(1)),
            optimizer=optax.adam(learning_rate=0.001),
            input_shape=(1, 8),
        )

        recursive_trainer = RecursiveSelection(
            model=copy.deepcopy(model),
            loss_fn=MeanPowerLoss(order=2),
            seed=17,
            disable_loading_bar=True,
        )
        recursive_out = recursive_trainer.train_model(
            train_ds=train_ds,
            test_ds=test_ds,
            epochs=[3],
            train_ds_selection=[slice(None, None, None)],
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
        )
        assert simple_out == recursive_out
