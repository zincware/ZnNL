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
import optax
from flax import linen as nn
from jax import random

from znnl.models import FlaxModel
from znnl.ntk_computation import JAXNTKSubsampling


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


class TestJAXNTKSubsampling:
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
        jax_ntk = JAXNTKSubsampling(
            apply_fn=self.flax_model.ntk_apply_fn,
            ntk_size=3,
            seed=0,
            batch_size=10,
            trace_axes=(),
            store_on_device=False,
            flatten=True,
        )

        assert jax_ntk.apply_fn == self.flax_model.ntk_apply_fn
        assert jax_ntk.ntk_size == 3
        assert jax_ntk.seed == 0
        assert jax_ntk.batch_size == 10
        assert jax_ntk.trace_axes == ()
        assert jax_ntk.store_on_device is False
        assert jax_ntk.flatten is True

    def test_get_sample_indices(self):
        """
        Test the _get_sample_indices method.
        """
        jax_ntk = JAXNTKSubsampling(
            apply_fn=self.flax_model.ntk_apply_fn,
            ntk_size=3,
            seed=0,
        )

        sample_indices = jax_ntk._get_sample_indices(self.dataset["inputs"])

        assert len(sample_indices) == 3
        assert sample_indices[0].shape == (3,)
        assert sample_indices[1].shape == (3,)
        assert sample_indices[2].shape == (3,)

    def test_subsample_data(self):
        """
        Test the _subsample_data method.
        """
        jax_ntk = JAXNTKSubsampling(
            apply_fn=self.flax_model.ntk_apply_fn,
            ntk_size=3,
            seed=0,
        )

        jax_ntk._sample_indices = jax_ntk._get_sample_indices(self.dataset["inputs"])
        subsampled_data = jax_ntk._subsample_data(self.dataset["inputs"])

        assert len(subsampled_data) == 3
        assert subsampled_data[0].shape == (3, 8)
        assert subsampled_data[1].shape == (3, 8)
        assert subsampled_data[2].shape == (3, 8)

    def test_compute_ntk(self):
        """
        Test the compute_ntk method.
        """

        # Use vmap is False
        jax_ntk = JAXNTKSubsampling(
            apply_fn=self.flax_model.ntk_apply_fn,
            ntk_size=3,
            seed=0,
        )

        params = {"params": self.flax_model.model_state.params}

        ntk = jax_ntk.compute_ntk(params, self.dataset["inputs"])

        assert np.shape(ntk) == (3, 6, 6)
