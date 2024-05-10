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
from jax import random
from neural_tangents import stax

from znnl.models import NTModel
from znnl.ntk_computation import JAXNTKComputation

class TestNTModule:
    """
    Test suite for the neural tangents module.
    """

    def test_ntk_shape(self):
        """
        Creates two models with different NTK tracing, which results in a different
        shape of the NTK.
        """
        test_model = stax.serial(
            stax.Dense(5, b_std=0.5),
            stax.Relu(),
            stax.Dense(5, b_std=0.5),
            stax.Relu(),
            stax.Dense(5),
        )

        key1, key2 = random.split(random.PRNGKey(1), 2)
        x1 = random.normal(key1, (3, 8))

        nt_model_1 = NTModel(
            nt_module=test_model,
            optimizer=optax.adam(learning_rate=0.001),
            input_shape=(1, 8),
            seed=17,
        )

        nt_model_2 = NTModel(
            nt_module=test_model,
            optimizer=optax.adam(learning_rate=0.001),
            input_shape=(1, 8),
            seed=17,
        )

        ntk_computation_1 = JAXNTKComputation(nt_model_1.ntk_apply_fn, trace_axes=(-1,))
        ntk_1 = ntk_computation_1.compute_ntk({"params": nt_model_1.model_state.params}, x1)
        ntk_computation_2 = JAXNTKComputation(nt_model_2.ntk_apply_fn, trace_axes=())
        ntk_2 = ntk_computation_2.compute_ntk({"params": nt_model_2.model_state.params}, x1)

        assert ntk_1.shape == (3, 3)
        assert ntk_2.shape == (3, 3, 5, 5)
