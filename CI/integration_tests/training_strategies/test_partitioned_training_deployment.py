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

import copy

import optax
from jax import random
from neural_tangents import stax

from znnl.loss_functions import MeanPowerLoss
from znnl.models import NTModel
from znnl.training_strategies import PartitionedTraining, SimpleTraining


class TestPartitionedSelection:
    """
    Integration test suite for the partitioned training strategy.
    """

    @classmethod
    def setup_class(cls):
        """
        Create data for the tests.
        """
        key1, key2 = random.split(random.PRNGKey(1), 2)
        x = random.normal(key1, (3, 8))
        y = random.normal(key1, (3, 1))
        cls.train_ds = {"inputs": x, "targets": y}
        cls.test_ds = {"inputs": x, "targets": y}

    def test_metric_length(self):
        """
        Test the length of the metric output when training with the partitioned training
        strategy.

        The output of training a model provides a metric on the performance.
        The number of epochs that has actually been trained can be checked by measuring
        the length of this metric.
        The length has to be adjustable by the number of epochs.
        """

        model = NTModel(
            nt_module=stax.serial(stax.Dense(5), stax.Relu(), stax.Dense(1)),
            optimizer=optax.adam(learning_rate=0.001),
            input_shape=(1, 8),
        )

        trainer = PartitionedTraining(
            model=model,
            loss_fn=MeanPowerLoss(order=2),
            disable_loading_bar=True,
        )

        # Check if default train for 200 epochs
        batch_metric = trainer.train_model(
            train_ds=self.train_ds,
            test_ds=self.test_ds,
        )
        assert len(batch_metric["train_losses"]) == 200

        # Check optional kwargs
        batch_metric = trainer.train_model(
            train_ds=self.train_ds,
            test_ds=self.test_ds,
            epochs=[2, 5],
            train_ds_selection=[[-1], slice(1, -1, 3)],
        )
        assert len(batch_metric["train_losses"]) == 7

    def test_comparison_to_simple_training(self):
        """
        Test the comparison to simple training strategy.

        The partitioned training has to be identical to simple training for selecting
        train_ds_selection=[slice(None, None, None)]
        """

        model = NTModel(
            nt_module=stax.serial(stax.Dense(5), stax.Relu(), stax.Dense(1)),
            optimizer=optax.adam(learning_rate=0.001),
            input_shape=(1, 8),
        )

        partitioned_trainer = PartitionedTraining(
            model=copy.deepcopy(model),
            loss_fn=MeanPowerLoss(order=2),
            seed=17,
            disable_loading_bar=True,
        )
        partitioned_out = partitioned_trainer.train_model(
            train_ds=self.train_ds,
            test_ds=self.test_ds,
            epochs=[3],
            train_ds_selection=[slice(None, None, None)],
            batch_size=[2],
        )
        simple_trainer = SimpleTraining(
            model=copy.deepcopy(model),
            loss_fn=MeanPowerLoss(order=2),
            seed=17,
            disable_loading_bar=True,
        )
        simple_out = simple_trainer.train_model(
            train_ds=self.train_ds, test_ds=self.test_ds, epochs=3, batch_size=2
        )
        assert simple_out == partitioned_out
