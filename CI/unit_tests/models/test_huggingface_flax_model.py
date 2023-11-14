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

import optax
import pytest
from jax import random
from transformers import FlaxResNetForImageClassification, ResNetConfig

from znnl.models import HuggingFaceFlaxModel


class TestFlaxHFModule:
    """
    Integration test suite for the flax Hugging Face (HF) module.

    Train the model for a few steps and check whether the loss decreases.
    """

    @classmethod
    def setup_class(cls):
        """
        Create a model and data for the tests.
        The resnet config has a 1 dimensional input and a 2 dimensional output.
        """

        resnet_config = ResNetConfig(
            num_channels=2,
            embedding_size=64,
            hidden_sizes=[256, 512, 1024, 2048],
            depths=[3, 4, 6, 3],
            layer_type="bottleneck",
            hidden_act="relu",
            downsample_in_first_stage=False,
            out_features=None,
            out_indices=None,
            id2label=dict(zip([1, 2], [1, 2])),
            return_dict=True,
        )
        hf_model = FlaxResNetForImageClassification(
            config=resnet_config,
            input_shape=(1, 8, 8, 2),
            seed=0,
            _do_init=True,
        )
        cls.model = HuggingFaceFlaxModel(
            hf_model,
            optax.adam(learning_rate=0.001),
            batch_size=3,
        )

        key = random.PRNGKey(0)
        cls.x = random.normal(key, (3, 2, 8, 8))

    def test_ntk_shape(self):
        """
        Test whether the NTK shape is correct.
        """
        ntk = self.model.compute_ntk(self.x)["empirical"]
        assert ntk.shape == (3, 3)

    def test_infinite_failure(self):
        """
        Test that the call to the infinite NTK fails.
        """
        with pytest.raises(NotImplementedError):
            self.model.compute_ntk(self.x, infinite=True)
