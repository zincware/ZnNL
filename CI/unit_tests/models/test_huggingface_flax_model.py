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
import pytest
from flax import linen as nn
from jax import random
from transformers import FlaxResNetForImageClassification, ResNetConfig

from znnl.loss_functions import CrossEntropyLoss
from znnl.models import HuggingFaceFlaxModel
from znnl.training_strategies import SimpleTraining


class TestFlaxHFModule:
    """
    Test suite for the flax Hugging Face (HF) module.
    """

    @classmethod
    def setup_class(cls):
        """
        Create a model and data for the tests.
        The resnet config has a 1 dimensional input and a 2 dimensional output.
        """

        resnet50_config = ResNetConfig(
            num_channels=3,
            embedding_size=64,
            hidden_sizes=[256, 512, 1024, 2048],
            depths=[3, 4, 6, 3],
            layer_type="bottleneck",
            hidden_act="relu",
            downsample_in_first_stage=False,
            out_features=None,
            out_indices=None,
            id2label=dict(
                zip(np.arange(10), np.arange(10))
            ),  # Dummy labels to define the output dimension
            return_dict=True,
        )

        # ResNet-18 taken from https://huggingface.co/microsoft/resnet-18/blob/main/config.json
        resnet18_config = ResNetConfig(
            num_channels=3,
            embedding_size=64,
            hidden_sizes=[64, 128, 256, 512],
            depths=[2, 2, 2, 2],
            layer_type="basic",
            hidden_act="relu",
            downsample_in_first_stage=False,
            id2label=dict(
                zip(np.arange(10), np.arange(10))
            ),  # Dummy labels to define the output dimension
            return_dict=True,
        )

        resnet18 = FlaxResNetForImageClassification(
            config=resnet50_config,
            input_shape=(1, 8, 8, 2),
            seed=0,
            _do_init=True,
        )
        resnet50 = FlaxResNetForImageClassification(
            config=resnet18_config,
            input_shape=(1, 8, 8, 2),
            seed=0,
            _do_init=True,
        )
        cls.resnet18 = HuggingFaceFlaxModel(
            resnet18,
            optax.adam(learning_rate=0.001),
            batch_size=3,
        )
        cls.resnet50 = HuggingFaceFlaxModel(
            resnet50,
            optax.adam(learning_rate=0.001),
            batch_size=3,
        )

        key = random.PRNGKey(0)
        cls.train_ds = {
            "inputs": random.normal(key, (3, 2, 8, 8)),
            "targets": np.arange(3),
        }

    def train_model(self, model, x):
        """
        Train the model for a few steps.
        """
        trainer = SimpleTraining(
            model=model,
            loss_fn=CrossEntropyLoss(),
        )
        batched_loss = trainer.train_model(
            train_ds=self.train_ds,
            test_ds=self.train_ds,
            epochs=5,
        )
        return batched_loss

    def test_loss_decreaser(self):
        """
        Test whether the loss decreases.
        """
        batched_loss = self.train_model(self.resnet18, self.train_ds)
        assert np.all(np.diff(batched_loss) < 0)

        batched_loss = self.train_model(self.resnet50, self.train_ds)
        assert np.all(np.diff(batched_loss) < 0)
