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
from papyrus.utils.matrix_utils import flatten_rank_4_tensor

from znnl.models import FlaxModel
from znnl.ntk_computation import JAXNTKCombinations


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
        jax_ntk = JAXNTKCombinations(
            apply_fn=self.flax_model.apply,
            class_labels=[0, 1, 2],
        )

        assert jax_ntk.class_labels == [0, 1, 2]

    def test_reduce_data_to_labels(self):
        """
        Test the _reduce_data_to_labels method.
        """
        jax_ntk = JAXNTKCombinations(
            apply_fn=self.flax_model.apply,
            class_labels=[0, 1],
        )

        # Test the one-hot targets
        reduced_data = jax_ntk._reduce_data_to_labels(self.dataset_onehot)
        assert reduced_data["inputs"].shape == (6, 8)
        assert reduced_data["targets"].shape == (6, 3)

        # Test the integer targets
        reduced_data = jax_ntk._reduce_data_to_labels(self.dataset_int)
        assert reduced_data["inputs"].shape == (6, 8)
        assert reduced_data["targets"].shape == (6, 1)

    def test_get_label_indices(self):
        """
        Test the _get_label_indices method.
        """
        jax_ntk = JAXNTKCombinations(
            apply_fn=self.flax_model.apply,
            class_labels=[0, 1, 2],
        )

        # Test the one-hot targets
        sample_idx_one_hot = jax_ntk._get_label_indices(self.dataset_onehot)
        assert type(sample_idx_one_hot) == dict
        assert len(sample_idx_one_hot) == 3
        assert len(sample_idx_one_hot[0]) == 4
        assert len(sample_idx_one_hot[1]) == 2
        assert len(sample_idx_one_hot[2]) == 2

        # Test the integer targets
        sample_idx_int = jax_ntk._get_label_indices(self.dataset_int)
        assert type(sample_idx_int) == dict
        assert len(sample_idx_int) == 3
        assert len(sample_idx_int[0]) == 4
        assert len(sample_idx_int[1]) == 2
        assert len(sample_idx_int[2]) == 2

    def test_compute_combinations(self):
        """
        Test the _compute_combinations method.
        """

        jax_ntk = JAXNTKCombinations(
            apply_fn=self.flax_model.apply,
            class_labels=[0, 1],
        )
        combinations = jax_ntk._compute_combinations()
        assert combinations == [(0,), (1,), (0, 1)]

        jax_ntk = JAXNTKCombinations(
            apply_fn=self.flax_model.apply,
            class_labels=[0, 1, 2],
        )
        combinations = jax_ntk._compute_combinations()
        assert combinations == [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]

    def test_take_sub_ntk(self):
        """
        Test the _take_sub_ntk method.
        """
        jax_ntk = JAXNTKCombinations(
            apply_fn=self.flax_model.apply,
            class_labels=[0, 1],
        )

        # Test shape flattened NTK
        jax_ntk._ntk_shape = (8, 8, 2, 2)
        jax_ntk._is_flattened = True
        jax_ntk.flatten = True
        ntk = random.normal(random.PRNGKey(0), (16, 16))
        reduced_data = jax_ntk._reduce_data_to_labels(self.dataset_int)
        label_indices = jax_ntk._get_label_indices(reduced_data)
        combination = (0, 1)
        sub_ntk = jax_ntk._take_sub_ntk(ntk, label_indices, combination)
        assert sub_ntk.shape == (12, 12)

        # Test shape unflattened NTK
        jax_ntk._ntk_shape = (8, 8, 2, 2)
        jax_ntk._is_flattened = False
        jax_ntk.flatten = False
        ntk = random.normal(random.PRNGKey(0), (8, 8, 2, 2))
        reduced_data = jax_ntk._reduce_data_to_labels(self.dataset_int)
        label_indices = jax_ntk._get_label_indices(reduced_data)
        combination = (0, 1)
        sub_ntk = jax_ntk._take_sub_ntk(ntk, label_indices, combination)
        assert sub_ntk.shape == (6, 6, 2, 2)

        # Test entries of the sub-NTK
        jax_ntk._ntk_shape = (4, 4, 2, 2)
        jax_ntk._is_flattened = True
        jax_ntk.flatten = True
        # Create some easier to check the sub-NTK
        targets = np.array([0, 0, 1, 1, 2, 2, 2, 2])
        dataset = {
            "inputs": self.dataset_int["inputs"],
            "targets": np.expand_dims(targets, axis=1),
        }
        # Reduce data to given labels and get label indices
        reduced_data = jax_ntk._reduce_data_to_labels(dataset)
        label_indices = jax_ntk._get_label_indices(reduced_data)
        # NTK of selected labels
        ntk = np.arange(8 * 8).reshape((4, 4, 2, 2))
        combination = (0,)
        # This is what should be extracted
        _sub_ntk = ntk[np.ix_(label_indices[0], label_indices[0])]
        _sub_ntk, _ = flatten_rank_4_tensor(_sub_ntk)
        # Compute the sub-NTK
        ntk, _ = flatten_rank_4_tensor(ntk)
        sub_ntk = jax_ntk._take_sub_ntk(ntk, label_indices, combination)
        # Check if the sub-NTK is correct
        assert np.all(sub_ntk == _sub_ntk)

    def test_compute_ntk(self):
        """
        Test the compute_ntk method.
        """

        jax_ntk = JAXNTKCombinations(
            apply_fn=self.flax_model.apply,
            class_labels=[0, 1],
        )

        params = {"params": self.flax_model.model_state.params}

        ntks = jax_ntk.compute_ntk(params, self.dataset_int)
        assert len(ntks) == 3
        assert ntks[0].shape == (8, 8)
        assert ntks[1].shape == (4, 4)
        assert ntks[2].shape == (12, 12)

        jax_ntk = JAXNTKCombinations(
            apply_fn=self.flax_model.apply,
            class_labels=[0, 1, 2],
        )

        ntks = jax_ntk.compute_ntk(params, self.dataset_int)
        assert len(ntks) == 7
        assert [np.shape(ntk) for ntk in ntks] == [
            (8, 8),
            (4, 4),
            (4, 4),
            (12, 12),
            (12, 12),
            (8, 8),
            (16, 16),
        ]
