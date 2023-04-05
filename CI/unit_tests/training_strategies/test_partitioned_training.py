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
Unit tests for the pertitioned training class.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import optax
from jax import random
from neural_tangents import stax
from numpy.testing import assert_raises

from znnl.loss_functions import MeanPowerLoss
from znnl.models import NTModel
from znnl.training_strategies import PartitionedTraining


class TestPartitionedSelection:
    """
    Unit test suite of the partitioned training strategy.
    """

    def test_parameter_error(self):
        """
        Test error raises for wrong keyword arguments.

        Assert a KeyError when using the partitioned training strategy with an integer
        instead of a list input for epochs.
        Assert a KeyError for differing list lengths for epochs compared to batch_size
        and train_ds_selection.
        """

        # Create some test data
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

        trainer = PartitionedTraining(
            model=model,
            loss_fn=MeanPowerLoss(order=2),
            disable_loading_bar=True,
        )

        # Check Error handling for passing an int for epochs instead of a list
        assert_raises(KeyError, trainer.train_model, train_ds, test_ds, 50)

        # Check Error handling for not matching lengths
        assert_raises(KeyError, trainer.train_model, train_ds, test_ds, [2])
