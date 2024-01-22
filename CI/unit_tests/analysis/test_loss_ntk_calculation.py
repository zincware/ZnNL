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

import jax.numpy as np
import pytest

from znnl.analysis import loss_ntk_calculation
from znnl.training_recording import JaxRecorder
from znnl.models import NTModel
from znnl.data import MNISTGenerator
from neural_tangents import stax

import optax
import tensorflow_datasets as tfds


class TestLossNTKCalculation:
    """
    Test Suite for the loss NTK calculation module.
    """

    def test_loss_ntk_calculation(self):
        """
        Test the loss NTK calculation.
        """

        # Define a test Network
        dense_network = stax.serial(
            stax.Dense(32),
            stax.Relu(),
            stax.Dense(32),
        )

        # Define a test model
        fuel_model = NTModel(
            nt_module=dense_network,
            optimizer=optax.adam(learning_rate=0.005),
            input_shape=(9,),
            trace_axes=(),
            batch_size=314,
        )

        # Initialize model parameters

        data_generator = MNISTGenerator(ds_size=10)
        data_set = {
            "inputs": data_generator.train_ds["inputs"],
            "targets": data_generator.train_ds["targets"],
        }

        print(fuel_model.model_state.params)


TestLossNTKCalculation().test_loss_ntk_calculation()
