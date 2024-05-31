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
from znnl.ntk_computation import JAXNTKClassWise


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


class TestJAXNTKClassWise:
    """
    Test class for the class-wise JAX NTK computation.
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

        # Create random labels between zero and two
        targets = np.array([0, 1, 2, 0, 1, 2, 0, 0])
        one_hot_targets = np.eye(3)[targets]

        cls.dataset_int = {
            "inputs": random.normal(random.PRNGKey(0), (8, 8)),
            "targets": np.expand_dims(targets, axis=1),
        }
        cls.dataset_onehot = {
            "inputs": random.normal(random.PRNGKey(0), (8, 8)),
            "targets": one_hot_targets,
        }

    def test_constructor(self):
        """
        Test the constructor of the JAX NTK computation class.
        """
        jax_ntk = JAXNTKClassWise(
            apply_fn=self.flax_model.apply,
        )

        assert jax_ntk.batch_size == 10
        assert jax_ntk._sample_indices == None

    def test_get_label_indices(self):
        """
        Test the _get_label_indices method.
        """
        jax_ntk = JAXNTKClassWise(
            apply_fn=self.flax_model.apply,
        )

        # Test the one-hot targets
        sample_idx_one_hot = jax_ntk._get_label_indices(self.dataset_onehot)
        assert len(sample_idx_one_hot) == 3
        assert len(sample_idx_one_hot[0]) == 4
        assert len(sample_idx_one_hot[1]) == 2
        assert len(sample_idx_one_hot[2]) == 2

        # Test the integer targets
        sample_idx_int = jax_ntk._get_label_indices(self.dataset_int)
        assert len(sample_idx_int) == 3
        assert len(sample_idx_int[0]) == 4
        assert len(sample_idx_int[1]) == 2
        assert len(sample_idx_int[2]) == 2

        # Test upper bound of ntk_size
        jax_ntk.ntk_size = 3
        sample_idx_one_hot = jax_ntk._get_label_indices(self.dataset_onehot)
        assert len(sample_idx_one_hot) == 3
        assert len(sample_idx_one_hot[0]) == 3
        assert len(sample_idx_one_hot[1]) == 2
        assert len(sample_idx_one_hot[2]) == 2

    def test_subsample_data(self):
        """
        Test the _subsample_data method.
        """
        jax_ntk = JAXNTKClassWise(
            apply_fn=self.flax_model.apply,
        )

        # Test the one-hot targets
        subsampled_data_one_hot = jax_ntk._subsample_data(
            self.dataset_onehot["inputs"],
            jax_ntk._get_label_indices(self.dataset_onehot),
        )
        assert len(subsampled_data_one_hot) == 3
        assert subsampled_data_one_hot[0].shape == (4, 8)
        assert subsampled_data_one_hot[1].shape == (2, 8)
        assert subsampled_data_one_hot[2].shape == (2, 8)

        # Test the integer targets
        subsampled_data_int = jax_ntk._subsample_data(
            self.dataset_int["inputs"], jax_ntk._get_label_indices(self.dataset_int)
        )
        assert len(subsampled_data_int) == 3
        assert subsampled_data_int[0].shape == (4, 8)
        assert subsampled_data_int[1].shape == (2, 8)
        assert subsampled_data_int[2].shape == (2, 8)

    def test_compute_ntk(self):
        """
        Test the compute_ntk method.
        """
        jax_ntk = JAXNTKClassWise(
            apply_fn=self.flax_model.ntk_apply_fn,
            batch_size=10,
        )

        params = {"params": self.flax_model.model_state.params}

        # Test the one-hot targets
        ntks = jax_ntk.compute_ntk(params, self.dataset_onehot)
        assert len(ntks) == 3
        assert ntks[0].shape == (8, 8)
        assert ntks[1].shape == (4, 4)
        assert ntks[2].shape == (4, 4)

        # Test the integer targets
        ntks = jax_ntk.compute_ntk(params, self.dataset_int)
        print(ntks)
        assert len(ntks) == 3
        assert ntks[0].shape == (8, 8)
        assert ntks[1].shape == (4, 4)
        assert ntks[2].shape == (4, 4)

        # Test if not all classes are present
        dataset = {
            "inputs": self.dataset_int["inputs"],
            "targets": np.array([0, 0, 0, 0, 0, 0, 0, 0]),
        }
        ntks = jax_ntk.compute_ntk(params, dataset)
        assert len(ntks) == 1
        assert ntks[0].shape == (16, 16)

        dataset = {
            "inputs": self.dataset_int["inputs"],
            "targets": np.array([0, 0, 0, 0, 0, 0, 0, 5]),
        }
        ntks = jax_ntk.compute_ntk(params, dataset)
        assert len(ntks) == 6
        assert ntks[0].shape == (14, 14)
        assert ntks[1].shape == (0, 0)
        assert ntks[2].shape == (0, 0)
        assert ntks[3].shape == (0, 0)
        assert ntks[4].shape == (0, 0)
        assert ntks[5].shape == (2, 2)
