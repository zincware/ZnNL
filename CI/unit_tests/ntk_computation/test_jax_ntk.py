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

import jax.numpy as np
import neural_tangents as nt
import optax
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
        x = nn.Dense(features=2, use_bias=True)(x)
        return x


class TestJAXNTKComputation:
    """
    Test class for the JAX NTK computation class.
    """

    @classmethod
    def setup_class(cls):
        """
        Setup the test class.
        """
        cls.flax_model = FlaxModel(
            flax_module=FlaxTestModule(),
            optimizer=optax.adam(learning_rate=0.001),
            input_shape=(8,),
            seed=17,
        )

        cls.dataset = {
            "inputs": random.normal(random.PRNGKey(0), (10, 8)),
            "targets": random.normal(random.PRNGKey(1), (10, 2)),
        }

    def test_constructor(self):
        """
        Test the constructor of the JAX NTK computation class.
        """
        apply_fn = lambda x: x
        batch_size = 10
        ntk_implementation = None
        trace_axes = (-1,)
        store_on_device = False
        flatten = True
        data_keys = ["image", "label"]

        jax_ntk_computation = JAXNTKComputation(
            apply_fn=apply_fn,
            batch_size=batch_size,
            ntk_implementation=ntk_implementation,
            trace_axes=trace_axes,
            store_on_device=store_on_device,
            flatten=flatten,
            data_keys=data_keys,
        )

        assert jax_ntk_computation.apply_fn == apply_fn
        assert jax_ntk_computation.batch_size == batch_size
        assert jax_ntk_computation.trace_axes == trace_axes
        assert jax_ntk_computation.store_on_device == store_on_device
        assert jax_ntk_computation.flatten == flatten
        assert jax_ntk_computation.data_keys == data_keys

    def test_constructor_default(self):
        """
        Test the default setting of the constructor of the JAX NTK computation class.
        """
        apply_fn = lambda x: x

        jax_ntk_computation = JAXNTKComputation(
            apply_fn=apply_fn,
        )

        assert jax_ntk_computation.apply_fn == apply_fn
        assert jax_ntk_computation.batch_size == 10
        assert jax_ntk_computation.trace_axes == ()
        assert jax_ntk_computation.store_on_device == False
        assert jax_ntk_computation.flatten == True
        assert jax_ntk_computation.data_keys == ["inputs", "targets"]

        # Default ntk_implementation should be NTK_VECTOR_PRODUCTS
        assert (
            jax_ntk_computation.ntk_implementation
            == nt.NtkImplementation.NTK_VECTOR_PRODUCTS
        )

    def test_check_shape(self):
        """
        Test the shape checking function.
        """
        jax_ntk_computation = JAXNTKComputation(apply_fn=self.flax_model.ntk_apply_fn)

        ntk = np.ones((10, 10, 3, 3))
        ntk_ = jax_ntk_computation._check_shape(ntk)

        assert ntk_.shape == (30, 30)

    def test_compute_ntk(self):
        """
        Test the computation of the NTK.
        """
        params = {"params": self.flax_model.model_state.params}

        # Trace axes is empty and flatten is True
        jax_ntk_computation = JAXNTKComputation(
            apply_fn=self.flax_model.ntk_apply_fn,
            trace_axes=(),
            flatten=True,
        )
        ntk = jax_ntk_computation.compute_ntk(params, self.dataset)
        assert np.shape(ntk) == (1, 20, 20)

        # Trace axes is empty and flatten is False
        jax_ntk_computation = JAXNTKComputation(
            apply_fn=self.flax_model.ntk_apply_fn,
            trace_axes=(),
            flatten=False,
        )
        ntk = jax_ntk_computation.compute_ntk(params, self.dataset)

        assert np.shape(ntk) == (1, 10, 10, 2, 2)

        # Trace axes is (-1,) and flatten is True
        jax_ntk_computation = JAXNTKComputation(
            apply_fn=self.flax_model.ntk_apply_fn,
            trace_axes=(-1,),
            flatten=True,
        )
        ntk = jax_ntk_computation.compute_ntk(params, self.dataset)

        assert np.shape(ntk) == (1, 10, 10)

        # Trace axes is (-1,) and flatten is False
        jax_ntk_computation = JAXNTKComputation(
            apply_fn=self.flax_model.ntk_apply_fn,
            trace_axes=(-1,),
            flatten=False,
        )
        ntk = jax_ntk_computation.compute_ntk(params, self.dataset)

        assert np.shape(ntk) == (1, 10, 10)
