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

import copy
import tempfile

import h5py as hf
import numpy as onp
import optax
from neural_tangents import stax
from numpy import testing

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
            loss=True, accuracy=True, update_rate=1, chunk_size=11, name="trainer"
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

    def test_data_dump(self):
        """
        Test that the data dumping works correctly.
        """
        with tempfile.TemporaryDirectory() as directory:
            new_model = copy.deepcopy(self.production_model)
            train_recorder = copy.deepcopy(self.train_recorder)
            train_recorder.storage_path = directory
            train_recorder.instantiate_recorder(
                train_recorder._data_set, overwrite=True
            )

            # Define the training strategy
            training_strategy = rnd.training_strategies.SimpleTraining(
                model=new_model,
                loss_fn=rnd.loss_functions.CrossEntropyLoss(),
                accuracy_fn=rnd.accuracy_functions.LabelAccuracy(),
                recorders=[train_recorder],
            )

            # Retrain the model.
            training_strategy.train_model(
                train_ds=self.data_generator.train_ds,
                test_ds=self.data_generator.test_ds,
                batch_size=5,
                epochs=20,
            )

            # Check if there is data in database
            with hf.File(f"{directory}/trainer.h5", "r") as db:
                db_loss = onp.array(db["loss"])
                db_accuracy = onp.array(db["accuracy"])

                class_loss = onp.array(train_recorder._loss_array)
                class_accuracy = onp.array(train_recorder._accuracy_array)

                assert db_loss.shape == (11,)
                assert class_loss.shape == (9,)
                testing.assert_raises(
                    AssertionError,
                    testing.assert_array_equal,
                    db_loss.sum(),
                    class_loss.sum(),
                )

                assert db_accuracy.shape == (11,)
                assert class_accuracy.shape == (9,)
                testing.assert_raises(
                    AssertionError,
                    testing.assert_array_equal,
                    db_accuracy.sum(),
                    class_accuracy.sum(),
                )

    def test_export_function_no_db(self):
        """
        Test that the reports are exported correctly.
        """
        train_report = self.train_recorder.gather_recording()
        test_report = self.test_recorder.gather_recording()

        assert len(train_report.loss) == 10
        assert onp.sum(train_report.loss) > 0
        assert len(train_report.accuracy) == 10
        assert onp.sum(train_report.accuracy) > 0

        # Arrays should be resized now.
        assert len(test_report.loss) == 2
        assert onp.sum(test_report.loss) > 0
        assert len(test_report.accuracy) == 2
        assert onp.sum(test_report.accuracy) > 0

    def test_export_function_db(self):
        """
        Test that the reports are exported correctly.
        """
        with tempfile.TemporaryDirectory() as directory:
            new_model = copy.deepcopy(self.production_model)
            train_recorder = copy.deepcopy(self.train_recorder)
            train_recorder.storage_path = directory
            train_recorder.instantiate_recorder(
                train_recorder._data_set, overwrite=True
            )
            # Define the training strategy
            training_strategy = rnd.training_strategies.SimpleTraining(
                model=new_model,
                loss_fn=rnd.loss_functions.CrossEntropyLoss(),
                accuracy_fn=rnd.accuracy_functions.LabelAccuracy(),
                recorders=[train_recorder],
            )

            # Retrain the model.
            training_strategy.train_model(
                train_ds=self.data_generator.train_ds,
                test_ds=self.data_generator.test_ds,
                batch_size=5,
                epochs=20,
            )

            report = train_recorder.gather_recording()
            assert report.loss.shape[0] == 20
            testing.assert_array_equal(report.loss[11:], train_recorder._loss_array)

    def test_export_function_no_db_custom_selection(self):
        """
        Test that the reports are exported correctly.
        """
        # Note, NTK is not recorded, it should be caught and removed.
        train_report = self.train_recorder.gather_recording(
            selected_properties=["loss", "ntk"]
        )

        assert len(train_report.loss) == 10
        assert onp.sum(train_report.loss) > 0
        assert "ntk" not in list(train_report.__dict__)
