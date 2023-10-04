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
Test the trace regularizer class.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# from typing import Callable, List, Optional, Union

import jax
import jax.numpy as np
import optax
from flax import linen as nn
from flax.training.train_state import TrainState
from neural_tangents import stax
from jax import random

from znnl.models.flax_model import FlaxModel
from znnl.models.nt_model import NTModel
from znnl.regularizers import TraceRegularizer


class Network(nn.Module):
    """
    Simple flax module.
    """

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=5)(x)
        x = nn.relu(x)
        x = nn.Dense(features=1)(x)
        return x


class TestNormRegularizer:
    """
    Unit test suite for the norm regularizer class.
    """

    @classmethod
    def create_nt_model(cls, key: int) -> NTModel:
        """
        Create data for the tests.
        """
        nt_model = NTModel(
            nt_module=stax.serial(
                stax.Dense(5, b_std=1, parameterization="standard"),
                stax.Relu(),
                stax.Dense(1, b_std=1, parameterization="standard"),
            ),
            optimizer=optax.adam(learning_rate=0.001),
            input_shape=(1, 1),
            seed=key,
        )
        return nt_model

    @classmethod
    def create_flax_model(cls, key: int) -> FlaxModel:
        """
        Create data for the tests.
        """
        flax_model = FlaxModel(
            flax_module=Network(),
            optimizer=optax.adam(learning_rate=0.001),
            input_shape=(1, 1),
            seed=key,
        )
        return flax_model
    
    @classmethod
    def setup_data(cls):
        """
        Create data for the tests.
        """
        key1, key2 = random.split(random.PRNGKey(0), 2)
        # x = random.normal(key1, (3, 1))
        # y = random.normal(key2, (3, 1))
        x = np.ones((3, 1))
        y = np.zeros((3, 1))
        cls.train_ds = {"inputs": x, "targets": y}
        cls.test_ds = {"inputs": x, "targets": y}

    def test_constructor(self):
        """
        Test the constructor of the norm regularizer class.
        """
        regularizer = NormRegularizer(reg_factor=1e-2)
        assert regularizer.reg_factor == 1e-2

        regularizer = NormRegularizer(reg_factor=1e-4)
        assert regularizer.reg_factor == 1e-4

    def test_calculate_regularization(self):
        """
        Test the calculate regularization function.

        The function should return the norm of the parameters, which is tested by
        applying the default norm and the Euclidean norm.

        The test is performed for both the flax and the neural tangents model.
        """
        key = 0
        nt_model = self.create_nt_model(key)
        flax_model = self.create_flax_model(key)

        nt_params = jax.tree_util.tree_map(
            lambda x: jax.numpy.ones_like(x), nt_model.model_state.params
        )
        nt_model.model_state = TrainState.create(
            apply_fn=nt_model.apply_fn, params=nt_params, tx=nt_model.optimizer
        )

        flax_params = jax.tree_util.tree_map(
            lambda x: jax.numpy.ones_like(x), flax_model.model_state.params
        )
        flax_model.model_state = TrainState.create(
            apply_fn=flax_model.apply_fn, params=flax_params, tx=flax_model.optimizer
        )

        # Test the default norm (mean squared norm).
        regularizer = TraceRegularizer(reg_factor=1.0)
        nt_norm = regularizer(
            nt_model.apply_fn, nt_model.model_state.params, batch=self.train_ds, epoch=1
        )
        assert nt_norm == 4.0
        flax_norm = regularizer(
            flax_model.apply_fn, flax_model.model_state.params, batch=self.train_ds, epoch=1
        )
        assert flax_norm == 4.0

        regularizer = TraceRegularizer(reg_factor=1.0)
        nt_norm = regularizer(
            nt_model.apply_fn, nt_model.model_state.params, batch=None, epoch=1
        )
        print(nt_norm)
        # assert nt_norm == 2.0
        # flax_norm = regularizer(
        #     flax_model.apply_fn, flax_model.model_state.params, batch=None, epoch=1
        # )
        # assert flax_norm == 2.0

