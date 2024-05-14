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
from jax.flatten_util import ravel_pytree
from neural_tangents import stax

from znnl.loss_functions import MeanPowerLoss
from znnl.models import FlaxModel, NTModel
from znnl.training_strategies import SimpleTraining


class TestNTModelSeed:
    """
    Class to test the reproducibility of the NT models.
    """

    def test_models(self):
        """
        Creates two (identical) test models.
        """
        test_model = stax.serial(stax.Dense(5, b_std=0.5), stax.Relu(), stax.Dense(1))

        nt_model_1 = NTModel(
            nt_module=test_model,
            optimizer=optax.adam(learning_rate=0.001),
            input_shape=(1,),
            seed=17,
        )

        nt_model_2 = NTModel(
            nt_module=test_model,
            optimizer=optax.adam(learning_rate=0.001),
            input_shape=(1,),
            seed=17,
        )
        return nt_model_1, nt_model_2

    def test_initialization(self):
        """
        Tests that the two initialized models have identical parameters.
        """
        test_models = self.test_models()
        params_1, _ = ravel_pytree(test_models[0].model_state.params)
        params_2, _ = ravel_pytree(test_models[1].model_state.params)

        np.testing.assert_array_equal(params_1, params_2)

    def test_reinitalization(self):
        """
        Tests that reinitialization produces the same model when seeding.
        """
        test_models = self.test_models()
        old_params, _ = ravel_pytree(test_models[0].model_state.params)
        training_strategy = SimpleTraining(
            model=test_models[0],
            loss_fn=MeanPowerLoss(order=2),
            seed=17,
        )
        training_strategy.train_model(
            train_ds={"inputs": np.ones((10, 1)), "targets": np.zeros((10, 1))},
            test_ds={"inputs": np.ones((10, 1)), "targets": np.zeros((10, 1))},
            batch_size=1,
            epochs=1,
        )

        params_after_training, _ = ravel_pytree(test_models[0].model_state.params)

        # Test that params are different from before training
        with np.testing.assert_raises(AssertionError):
            np.testing.assert_array_equal(old_params, params_after_training)

        test_models[0].init_model(seed=17)
        params_after_reinit, _ = ravel_pytree(test_models[0].model_state.params)

        # Check that the model is in the initial state after reinitialization
        np.testing.assert_array_equal(params_after_reinit, old_params)

        # Check that retraining leads at the same configuration
        training_strategy = SimpleTraining(
            model=test_models[0],
            loss_fn=MeanPowerLoss(order=2),
            seed=17,
        )
        training_strategy.train_model(
            train_ds={"inputs": np.ones((10, 1)), "targets": np.zeros((10, 1))},
            test_ds={"inputs": np.ones((10, 1)), "targets": np.zeros((10, 1))},
            batch_size=1,
            epochs=1,
        )
        params_after_retraining, _ = ravel_pytree(test_models[0].model_state.params)

        np.testing.assert_array_equal(params_after_training, params_after_retraining)


class Flax_test_model(nn.Module):
    """
    Test model for the Flax tests.
    """

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(5, use_bias=True)(x)
        x = nn.relu(x)
        x = nn.Dense(features=1, use_bias=True)(x)
        return x


class TestFlaxModelSeed:
    """
    Class to test the reproducibility of the Flax models.
    """

    def test_models(self):
        """
        Creates two (identical) test models.
        """
        flax_model_1 = FlaxModel(
            flax_module=Flax_test_model(),
            optimizer=optax.adam(learning_rate=0.001),
            input_shape=(1,),
            seed=17,
        )

        flax_model_2 = FlaxModel(
            flax_module=Flax_test_model(),
            optimizer=optax.adam(learning_rate=0.001),
            input_shape=(1,),
            seed=17,
        )
        return flax_model_1, flax_model_2

    def test_initialization(self):
        """
        Tests that the two initialized models have identical parameters.
        """
        test_models = self.test_models()
        params_1, _ = ravel_pytree(test_models[0].model_state.params)
        params_2, _ = ravel_pytree(test_models[1].model_state.params)

        np.testing.assert_array_equal(params_1, params_2)

    def test_reinitalization(self):
        """
        Tests that reinitialization produces the same model when seeding.
        """
        test_models = self.test_models()
        old_params, _ = ravel_pytree(test_models[0].model_state.params)
        training_strategy = SimpleTraining(
            model=test_models[0],
            loss_fn=MeanPowerLoss(order=2),
            seed=17,
        )
        training_strategy.train_model(
            train_ds={"inputs": np.ones((10, 1)), "targets": np.zeros((10, 1))},
            test_ds={"inputs": np.ones((10, 1)), "targets": np.zeros((10, 1))},
            batch_size=1,
            epochs=1,
        )

        params_after_training, _ = ravel_pytree(test_models[0].model_state.params)

        # Test that params are different from before training
        with np.testing.assert_raises(AssertionError):
            np.testing.assert_array_equal(old_params, params_after_training)

        test_models[0].init_model(seed=17)
        params_after_reinit, _ = ravel_pytree(test_models[0].model_state.params)

        # Check that the model is in the initial state after reinitalization
        np.testing.assert_array_equal(params_after_reinit, old_params)

        # Check that retraining leads at the same configuration
        training_strategy = SimpleTraining(
            model=test_models[0],
            loss_fn=MeanPowerLoss(order=2),
            seed=17,
        )
        training_strategy.train_model(
            train_ds={"inputs": np.ones((10, 1)), "targets": np.zeros((10, 1))},
            test_ds={"inputs": np.ones((10, 1)), "targets": np.zeros((10, 1))},
            batch_size=1,
            epochs=1,
        )
        params_after_retraining, _ = ravel_pytree(test_models[0].model_state.params)

        np.testing.assert_array_equal(params_after_training, params_after_retraining)
