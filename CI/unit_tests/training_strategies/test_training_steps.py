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
Test the TrainStep class.
"""

import jax
import jax.numpy as np
import optax
from flax import linen as nn

from znnl.models.jax_model import TrainState
from znnl.training_strategies.training_steps import TrainStep


class FlaxTestModule(nn.Module):
    """
    Example flax model.
    """

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(5, use_bias=True)(x)
        x = nn.relu(x)
        x = nn.Dense(features=1, use_bias=True)(x)
        return x


class FlaxTestModuleBatchStats(nn.Module):
    """
    Example flax model with batch statistics.
    """

    @nn.compact
    def __call__(self, x, train: bool):
        x = nn.Dense(5, use_bias=True)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        x = nn.Dense(features=1, use_bias=True)(x)
        return x


class TestTrainStep:
    """
    Class to test the TrainStep class.
    """

    @classmethod
    def setup_class(cls):
        """
        Create a train state.
        """
        x = np.ones((1, 3))

        flax_module = FlaxTestModule()
        params = flax_module.init(jax.random.key(0), x)
        cls.state = TrainState.create(
            apply_fn=flax_module.apply,
            params=params,
            tx=optax.adam(learning_rate=0.001),
        )

        flax_module_batchstats = FlaxTestModuleBatchStats()
        variables = flax_module_batchstats.init(jax.random.key(0), x, train=True)
        cls.state_batchstats = TrainState.create(
            apply_fn=flax_module_batchstats.apply,
            params=variables["params"],
            batch_stats=variables["batch_stats"],
            tx=optax.adam(learning_rate=0.001),
            use_batch_stats=True,
        )

    def test_train_step_constructor(self):
        """
        Test the constructor of the TrainStep class.
        """
        # Test a model without batch statistics.
        stepper = TrainStep(self.state)
        assert stepper.use_batch_stats == False
        assert stepper.train_step == stepper.vanilla_step

        # Test a model with batch statistics.
        stepper = TrainStep(self.state_batchstats)
        assert stepper.use_batch_stats == True
        assert stepper.train_step == stepper.batch_stat_step
