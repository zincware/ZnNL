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
import numpy as onp
import optax
from flax import linen as nn
from neural_tangents import stax
from numpy.testing import assert_array_almost_equal

from znnl.analysis import LossDerivative, loss_ntk_calculation
from znnl.data import MNISTGenerator
from znnl.distance_metrics import LPNorm
from znnl.loss_functions import LPNormLoss
from znnl.models import FlaxModel
from znnl.training_recording import JaxRecorder


# Defines a simple CNN module
class ProductionModule(nn.Module):
    """
    Simple CNN module.
    """

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=16, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2))
        x = nn.Conv(features=16, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=10)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)

        return x


class TestLossNTKCalculation:
    """
    Test Suite for the loss NTK calculation module.
    """

    def test_loss_ntk_calculation(self):
        """
        Test the Loss NTK calculation.
        Here we test if the Loss NTK calculated through the neural tangents module is
        the same as the Loss NTK calculated with the already implemented NTK and loss
        derivatives.
        """

        # Define a test model
        production_model = FlaxModel(
            flax_module=ProductionModule(),
            optimizer=optax.adam(learning_rate=0.01),
            input_shape=(1, 28, 28, 1),
            trace_axes=(),
        )
        # Initialize model parameters

        data_generator = MNISTGenerator(ds_size=20)
        data_set = {
            "inputs": data_generator.train_ds["inputs"],
            "targets": data_generator.train_ds["targets"],
        }

        # Initialize the loss NTK calculation
        loss_ntk_calculator = loss_ntk_calculation(
            metric_fn=LPNorm(order=2),
            model=production_model,
            dataset=data_set,
        )

        # Compute the loss NTK
        loss_ntk = loss_ntk_calculator.compute_loss_ntk(
            x_i=data_set, model=production_model
        )["empirical"]

        # Now for comparison calculate regular ntk
        ntk = production_model.compute_ntk(data_set["inputs"], infinite=False)[
            "empirical"
        ]
        # Calculate Loss derivative fn
        loss_derivative_calculator = LossDerivative(LPNormLoss(order=2))

        # predictions calculation analogous to the one in jax recording
        predictions = production_model(data_set["inputs"])
        if type(predictions) is tuple:
            predictions = predictions[0]

        # calculation of loss derivatives
        # note: here we need the derivatives of the subloss, not the regular loss fn
        loss_derivatives = onp.empty(shape=(len(predictions), len(predictions[0])))
        for i in range(len(loss_derivatives)):
            # The weird indexing here is because of axis constraints in the LPNormLoss module
            loss_derivatives[i] = loss_derivative_calculator.calculate(
                predictions[i : i + 1], data_set["targets"][i : i + 1]
            )[0]

        # Calculate the loss NTK from the loss derivatives and the ntk
        loss_ntk_2 = np.einsum(
            "ik, jl, ijkl-> ij", loss_derivatives, loss_derivatives, ntk
        )

        # Assert that the loss NTKs are the same
        assert_array_almost_equal(loss_ntk, loss_ntk_2, decimal=4)


TestLossNTKCalculation().test_loss_ntk_calculation()
