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
from jax import random
from neural_tangents import stax

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


class TestTraceRegularizer:
    """
    Unit test suite for the trace regularizer class.
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
            batch_size=3,
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

    def test_constructor(self):
        """
        Test the constructor of the norm regularizer class.
        """
        regularizer = TraceRegularizer(reg_factor=1e-2)
        assert regularizer.reg_factor == 1e-2

        regularizer = TraceRegularizer(reg_factor=1e-4)
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

        x = np.ones((3, 1))
        y = np.zeros((3, 1))
        batch = {"inputs": x, "targets": y}

        regularizer = TraceRegularizer(reg_factor=1.0)

        # Calculate norm from regularizer
        ntk_norm_regularizer = regularizer(
            model=nt_model, params=nt_model.model_state.params, batch=batch, epoch=1
        )
        # Calculate norm from NTK
        ntk = nt_model.compute_ntk(batch["inputs"], infinite=False)["empirical"]
        num_parameters = jax.flatten_util.ravel_pytree(nt_model.model_state.params)[
            0
        ].shape[0]
        normed_ntk = ntk / num_parameters
        diag_ntk = np.diagonal(normed_ntk)
        mean_trace = np.mean(diag_ntk)
        assert mean_trace == ntk_norm_regularizer

        # Calculate norm from regularizer
        flax_norm_regularizer = regularizer(
            model=flax_model, params=flax_model.model_state.params, batch=batch, epoch=1
        )
        # Calculate norm from NTK
        ntk = flax_model.compute_ntk(batch["inputs"], infinite=False)["empirical"]
        num_parameters = jax.flatten_util.ravel_pytree(flax_model.model_state.params)[
            0
        ].shape[0]
        normed_ntk = ntk / num_parameters
        diag_ntk = np.diagonal(normed_ntk)
        mean_trace = np.mean(diag_ntk)
        assert mean_trace == flax_norm_regularizer
