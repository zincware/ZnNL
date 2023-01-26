"""
ZnRND: A zincwarecode package.

License
-------
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the zincwarecode Project.

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
Test the model recording.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as onp
import optax
from neural_tangents import stax

import znrnd as rnd


class TestRecorderDeployment:
    """
    Test suite for the model recorder.

    In the setup, we download the data, prepare the recorders and train a model on
    a small data set. The rest of the test is then about checking how the recording
    has turned out.
    """

    @classmethod
    def setup_class(cls):
        """
        Prepare the class for running.
        """
        # Data Generator
        cls.data_generator = rnd.data.MNISTGenerator(ds_size=10)

        # Make a network
        network = stax.serial(
            stax.Flatten(), stax.Dense(128), stax.Relu(), stax.Dense(10)
        )

        # Set the class assigned recorders
        cls.train_recorder = rnd.model_recording.JaxRecorder(
            loss=True, accuracy=True, update_rate=1
        )
        cls.test_recorder = rnd.model_recording.JaxRecorder(
            loss=True, accuracy=True, ntk=True, update_rate=5
        )

        # Define the model
        cls.production_model = rnd.models.NTModel(
            nt_module=network,
            optimizer=optax.adam(learning_rate=0.01),
            input_shape=(1, 28, 28, 1),
        )

        cls.train_recorder.instantiate_recorder(data_set=cls.data_generator.train_ds)
        cls.test_recorder.instantiate_recorder(data_set=cls.data_generator.test_ds)

        # Define training strategy
        cls.training_strategy = rnd.training_strategies.SimpleTraining(
            model=cls.production_model,
            loss_fn=rnd.loss_functions.CrossEntropyLoss(),
            accuracy_fn=rnd.accuracy_functions.LabelAccuracy(),
            recorders=[cls.train_recorder, cls.test_recorder],
        )
        # Train the model with the recorders
        cls.batch_wise_metrics = cls.training_strategy.train_model(
            train_ds=cls.data_generator.train_ds,
            test_ds=cls.data_generator.test_ds,
            batch_size=5,
            epochs=10,
        )

    def test_batch_loss(self):
        """
        Test that the batch_wise_metrics are returned correctly.
        """
        assert onp.sum(self.batch_wise_metrics["train_accuracy"]) > 0
        assert len(self.batch_wise_metrics["train_accuracy"]) == 10
        assert onp.sum(self.batch_wise_metrics["train_losses"]) > 0
        assert len(self.batch_wise_metrics["train_losses"]) == 10

    def test_private_arrays(self):
        """
        Test that the recorder internally holds the correct values.
        """
        assert len(self.train_recorder._loss_array) == 10
        assert onp.sum(self.train_recorder._loss_array) > 0
        assert len(self.train_recorder._accuracy_array) == 10
        assert onp.sum(self.train_recorder._accuracy_array) > 0

        assert len(self.test_recorder._loss_array) == 2
        assert len(self.test_recorder._accuracy_array) == 2
        assert onp.sum(self.test_recorder._loss_array) > 0
        assert onp.sum(self.test_recorder._accuracy_array) > 0

    def test_export_function(self):
        """
        Test that the reports are exported correctly.
        """
        train_report = self.train_recorder.export_dataset()
        test_report = self.test_recorder.export_dataset()

        assert len(train_report.loss) == 10
        assert onp.sum(train_report.loss) > 0
        assert len(train_report.accuracy) == 10
        assert onp.sum(train_report.accuracy) > 0

        # Arrays should be resized now.
        assert len(test_report.loss) == 2
        assert onp.sum(test_report.loss) > 0
        assert len(test_report.accuracy) == 2
        assert onp.sum(test_report.accuracy) > 0
