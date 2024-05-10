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

import optax
import pytest
from flax import linen as nn
from jax import random

from znnl.models import FlaxModel
from znnl.ntk_computation import JAXNTKComputation


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

        ntk_computation = JAXNTKComputation(model.ntk_apply_fn, trace_axes=(-1,))

        key1, key2 = random.split(random.PRNGKey(1), 2)
        x = random.normal(key1, (3, 8))
        ntk = ntk_computation.compute_ntk({"params": model.model_state.params}, x)
        assert ntk.shape == (3, 3)
