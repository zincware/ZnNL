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
import numpy as np
import optax
from jax import random
from transformers import FlaxResNetForImageClassification, ResNetConfig

from znnl.loss_functions import MeanPowerLoss
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
            num_channels=2,
            embedding_size=64,
            hidden_sizes=[256, 512, 1024, 2048],
            depths=[3, 4, 6, 3],
            layer_type="bottleneck",
            hidden_act="relu",
            downsample_in_first_stage=False,
            out_features=None,
            out_indices=None,
            id2label={
                i: i for i in range(3)
            },  # Dummy labels to define the output dimension
            return_dict=True,
        )

        # ResNet-18 taken from https://huggingface.co/microsoft/resnet-18/blob/main/config.json
        resnet18_config = ResNetConfig(
            num_channels=2,
            embedding_size=64,
            hidden_sizes=[64, 128, 256, 512],
            depths=[2, 2, 2, 2],
            layer_type="bottleneck",
            # layer_type="basic",
            hidden_act="relu",
            downsample_in_first_stage=False,
            id2label={
                i: i for i in range(3)
            },  # Dummy labels to define the output dimension
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
            optax.sgd(learning_rate=1e-3),
            batch_size=3,
        )
        cls.resnet50 = HuggingFaceFlaxModel(
            resnet50,
            optax.sgd(learning_rate=1e-3),
            batch_size=3,
        )

        key = random.PRNGKey(0)
        cls.train_ds = {
            "inputs": random.normal(key, (30, 2, 8, 8)),
            "targets": np.repeat(np.eye(3), 10, axis=0),
        }

    def train_model(self, model):
        """
        Train the model for a few steps.
        """
        trainer = SimpleTraining(
            model=model,
            loss_fn=MeanPowerLoss(order=2),
        )
        batched_loss = trainer.train_model(
            train_ds=self.train_ds,
            test_ds=self.train_ds,
            epochs=20,
            batch_size=30,
        )
        return batched_loss

    def test_loss_decreaser(self):
        """
        Test whether the training loss decreases.
        """
        batched_loss = np.array(self.train_model(self.resnet18)["train_losses"])
        diff = batched_loss[10:] - batched_loss[:1]
        np.testing.assert_array_less(diff, 0)

        batched_loss = np.array(self.train_model(self.resnet50)["train_losses"])
        diff = batched_loss[10:] - batched_loss[:1]
        np.testing.assert_array_less(diff, 0)
