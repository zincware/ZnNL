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

from znnl.analysis import EigenSpaceAnalysis, LossDerivative, LossNTKCalculation
from znnl.data import MNISTGenerator
from znnl.distance_metrics import LPNorm
from znnl.loss_functions import LPNormLoss
from znnl.models import FlaxModel, NTModel


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
    Test Suite for the LossNTKCalculation module.
    """

    def test_reshaping_methods(self):
        """
        Test the _reshape_dataset and _unshape_dataset methods.
        These are functions used in the loss NTK calculation to
        """
        # Define a dummy model and dataset to be able to define a
        # LossNTKCalculation class
        production_model = FlaxModel(
            flax_module=ProductionModule(),
            optimizer=optax.adam(learning_rate=0.01),
            input_shape=(1, 28, 28, 1),
            trace_axes=(),
        )

        data_generator = MNISTGenerator(ds_size=20)
        data_set = {
            "inputs": data_generator.train_ds["inputs"],
            "targets": data_generator.train_ds["targets"],
        }

        # Initialize the loss NTK calculation
        loss_ntk_calculator = LossNTKCalculation(
            metric_fn=LPNorm(order=2),
            model=production_model,
            dataset=data_set,
        )

        # Setup a test dataset for reshaping
        test_data_set = {
            "inputs": np.array([[1, 2, 3], [4, 5, 6]]),
            "targets": np.array([[7], [10]]),
        }

        # Test the reshaping
        reshaped_test_data_set = loss_ntk_calculator._reshape_dataset(test_data_set)

        assert_array_almost_equal(
            reshaped_test_data_set, np.array([[1, 2, 3, 7], [4, 5, 6, 10]])
        )

        # Test the unshaping
        input_0, target_0 = loss_ntk_calculator._unshape_data(
            reshaped_test_data_set,
            input_dimension=3,
            input_shape=(2, 3),
            target_shape=(2, 1),
            batch_length=reshaped_test_data_set.shape[0],
        )
        assert_array_almost_equal(input_0, test_data_set["inputs"])
        assert_array_almost_equal(target_0, test_data_set["targets"])

    def test_function_for_loss_ntk(self):
        """
        This method tests the function that is used for the correlation matrix
        in the loss NTK calculation. It is supposed to yield the loss per single
        datapoint.
        """
        # Define a simple feed forward test model
        feed_forward_model = stax.serial(
            stax.Dense(5),
            stax.Relu(),
            stax.Dense(2),
            stax.Relu(),
        )

        # Initialize the model
        model = NTModel(
            optimizer=optax.adam(learning_rate=0.01),
            input_shape=(1, 5),
            trace_axes=(),
            nt_module=feed_forward_model,
        )

        # Define a test dataset with only two datapoints
        test_data_set = {
            "inputs": np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 8]]),
            "targets": np.array([[1, 3], [2, 5]]),
        }

        # Initialize loss
        loss = LPNormLoss(order=2)
        # Initialize the loss NTK calculation
        loss_ntk_calculator = LossNTKCalculation(
            metric_fn=loss.metric,
            model=model,
            dataset=test_data_set,
        )

        # Calculate the subloss from the NTK first
        datapoint = loss_ntk_calculator._reshape_dataset(test_data_set)[0:1]
        subloss_from_NTK = loss_ntk_calculator._function_for_loss_ntk(
            {
                "params": model.model_state.params,
                "batch_stats": model.model_state.batch_stats,
            },
            datapoint=datapoint,
        )

        # Now calculate subloss manually
        applied_model = model.apply(
            {
                "params": model.model_state.params,
                "batch_stats": model.model_state.batch_stats,
            },
            test_data_set["inputs"][0],
        )
        subloss = np.linalg.norm(applied_model - test_data_set["targets"][0], ord=2)

        # Check that the two losses are the same
        assert subloss - subloss_from_NTK < 1e-5

    def test_loss_NTK_calculation(self):
        """
        Test the Loss NTK calculation.
        Here we test if the Loss NTK calculated through the neural tangents module is
        the same as the Loss NTK calculated with the already implemented NTK and loss
        derivatives.
        We do this for a small CNN model and the MNIST dataset.
        We also check if the eigenvalues of the two Loss NTKs are the same.

        The current implementation yields a precision of e-4. If these are numerical
        errors or due to a mistake in the implementation is to be decided.
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
        loss_ntk_calculator = LossNTKCalculation(
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
            # The weird indexing here is because of axis constraints in LPNormLoss
            loss_derivatives[i] = loss_derivative_calculator.calculate(
                predictions[i : i + 1], data_set["targets"][i : i + 1]
            )[0]

        # Calculate the loss NTK from the loss derivatives and the ntk
        loss_ntk_2 = np.einsum(
            "ik, jl, ijkl-> ij", loss_derivatives, loss_derivatives, ntk
        )

        # Assert that the loss NTKs are the same
        assert_array_almost_equal(loss_ntk, loss_ntk_2, decimal=4)

        calculator1 = EigenSpaceAnalysis(matrix=loss_ntk)
        calculator2 = EigenSpaceAnalysis(matrix=loss_ntk_2)

        eigenvalues1 = calculator1.compute_eigenvalues(normalize=False)
        eigenvalue2 = calculator2.compute_eigenvalues(normalize=False)

        assert_array_almost_equal(eigenvalues1, eigenvalue2, decimal=4)
