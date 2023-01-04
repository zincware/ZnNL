"""
ZnRND: A Zincwarecode package.

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
Module for testing the trace optimizer
"""
import jax.numpy as np
from neural_tangents import stax

from znrnd.accuracy_functions import LabelAccuracy
from znrnd.data import MNISTGenerator
from znrnd.loss_functions import CrossEntropyLoss
from znrnd.models import NTModel
from znrnd.optimizers import TraceOptimizer


class TestTraceOptimizer:
    """
    Test suite for optimizers.
    """

    def test_optimizer_instantiation(self):
        """
        Unit test for the trace optimizer
        """
        # Test default settings
        my_optimizer = TraceOptimizer(scale_factor=100.0)
        assert my_optimizer.scale_factor == 100.0
        assert my_optimizer.rescale_interval == 1

        # Test custom settings
        my_optimizer = TraceOptimizer(scale_factor=50.0, rescale_interval=5)
        assert my_optimizer.scale_factor == 50.0
        assert my_optimizer.rescale_interval == 5

    def test_apply_operation(self):
        """
        Test the apply operation of the optimizer.

        Returns
        -------

        """
        # Set parameters.
        scale_factor = 10
        rescale_interval = 1

        # Use MNIST data
        data = MNISTGenerator(ds_size=10)

        # Define the optimizer
        optimizer = TraceOptimizer(
            scale_factor=scale_factor, rescale_interval=rescale_interval
        )

        # Use small dense model
        network = stax.serial(
            stax.Flatten(), stax.Dense(5), stax.Relu(), stax.Dense(10)
        )
        # Define the model
        model = NTModel(
            loss_fn=CrossEntropyLoss(),
            optimizer=optimizer,
            input_shape=(1, 28, 28, 1),
            nt_module=network,
            accuracy_fn=LabelAccuracy(),
            batch_size=5,
            training_threshold=0.01,
        )

        # Get theoretical values
        ntk = model.compute_ntk(data.train_ds["inputs"], normalize=False)["empirical"]
        expected_lr = scale_factor / np.trace(ntk)

        # Compute actual values
        actual_lr = optimizer.apply_optimizer(
            model_state=model.model_state,
            data_set=data.train_ds["inputs"],
            ntk_fn=model.compute_ntk,
            epoch=1,
        ).opt_state

        assert actual_lr.hyperparams["learning_rate"] == expected_lr
