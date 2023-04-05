"""
ZnRND: A zincwarecode package.

License
-------
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the zincwarecode Project.

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
Unit tests for the flax models.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import optax
import pytest
from flax import linen as nn
from jax import random

from znnl.models import FlaxModel


class FlaxTestModule(nn.Module):
    """
    Test model for the Flax tests.
    """

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(5, use_bias=True)(x)
        x = nn.relu(x)
        x = nn.Dense(features=1, use_bias=True)(x)
        return x


class TestFlaxModule:
    """
    Test suite for the neural tangents module.
    """

    def test_ntk_shape(self):
        """
        Creates two models with different NTK tracing, which results in a different
        shape of the NTK.
        """
        model = FlaxModel(
            flax_module=FlaxTestModule(),
            optimizer=optax.adam(learning_rate=0.001),
            input_shape=(8,),
            seed=17,
        )

        key1, key2 = random.split(random.PRNGKey(1), 2)
        x = random.normal(key1, (3, 8))
        ntk = model.compute_ntk(x)["empirical"]
        assert ntk.shape == (3, 3)

    def test_infinite_failure(self):
        """
        Test that the call to the infinite NTK fails.
        """
        model = FlaxModel(
            flax_module=FlaxTestModule(),
            optimizer=optax.adam(learning_rate=0.001),
            input_shape=(8,),
            seed=17,
        )

        key1, key2 = random.split(random.PRNGKey(1), 2)
        x = random.normal(key1, (3, 8))

        with pytest.raises(NotImplementedError):
            model.compute_ntk(x, infinite=True)
