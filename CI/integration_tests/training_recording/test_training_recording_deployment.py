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

import copy
import tempfile

import h5py as hf
import numpy as onp
import optax
from neural_tangents import stax
from numpy import testing
from papyrus.measurements import NTK, Accuracy, Loss

import znnl as nl
from znnl.analysis import JAXNTKComputation


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
        cls.data_generator = nl.data.MNISTGenerator(ds_size=10)

        # Make a network
        network = stax.serial(
            stax.Flatten(), stax.Dense(128), stax.Relu(), stax.Dense(10)
        )

        # Set the class assigned recorders
        cls.train_recorder = nl.training_recording.JaxRecorder(
            storage_path=".",
            name="trainer",
            update_rate=1,
            chunk_size=11,
            measurements=[
                Loss(apply_fn=nl.loss_functions.MeanPowerLoss(order=2)),
                Accuracy(apply_fn=nl.accuracy_functions.LabelAccuracy()),
            ],
        )
        cls.test_recorder = nl.training_recording.JaxRecorder(
            storage_path=".",
            name="tester",
            update_rate=5,
            measurements=[
                Loss(apply_fn=nl.loss_functions.MeanPowerLoss(order=2)),
                Accuracy(apply_fn=nl.accuracy_functions.LabelAccuracy()),
                NTK(),
            ],
        )

        # Define the model
        cls.production_model = nl.models.NTModel(
            nt_module=network,
            optimizer=optax.adam(learning_rate=0.01),
            input_shape=(1, 28, 28, 1),
        )

        cls.train_recorder.instantiate_recorder(
            data_set=cls.data_generator.train_ds,
            model=cls.production_model,
            ntk_computation=JAXNTKComputation(cls.production_model.ntk_apply_fn),
        )
        cls.test_recorder.instantiate_recorder(
            data_set=cls.data_generator.test_ds,
            model=cls.production_model,
            ntk_computation=JAXNTKComputation(cls.production_model.ntk_apply_fn),
        )

        # Define training strategy
        cls.training_strategy = nl.training_strategies.SimpleTraining(
            model=cls.production_model,
            loss_fn=nl.loss_functions.CrossEntropyLoss(),
            accuracy_fn=nl.accuracy_functions.LabelAccuracy(),
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
        assert len(self.train_recorder._results["loss"]) == 10
        assert onp.sum(self.train_recorder._results["loss"]) > 0
        assert len(self.train_recorder._results["accuracy"]) == 10
        assert onp.sum(self.train_recorder._results["accuracy"]) > 0

        assert len(self.test_recorder._results["loss"]) == 2
        assert len(self.test_recorder._results["accuracy"]) == 2
        assert onp.sum(self.test_recorder._results["loss"]) > 0
        assert onp.sum(self.test_recorder._results["accuracy"]) > 0

    def test_data_dump(self):
        """
        Test that the data dumping works correctly.
        """
        with tempfile.TemporaryDirectory() as directory:

            new_model = copy.deepcopy(self.production_model)

            train_recorder = nl.training_recording.JaxRecorder(
                storage_path=directory,
                name="trainer",
                update_rate=1,
                chunk_size=11,
                measurements=[
                    Loss(apply_fn=nl.loss_functions.MeanPowerLoss(order=2)),
                    Accuracy(apply_fn=nl.accuracy_functions.LabelAccuracy()),
                ],
            )
            train_recorder.instantiate_recorder(
                data_set=self.data_generator.train_ds,
                model=new_model,
            )

            # Define the training strategy
            training_strategy = nl.training_strategies.SimpleTraining(
                model=new_model,
                loss_fn=nl.loss_functions.CrossEntropyLoss(),
                accuracy_fn=nl.accuracy_functions.LabelAccuracy(),
                recorders=[train_recorder],
            )

            # Retrain the model.
            training_strategy.train_model(
                train_ds=self.data_generator.train_ds,
                test_ds=self.data_generator.test_ds,
                batch_size=5,
                epochs=20,
            )

            # Print all files in the directory
            print(f"Files in directory: {os.listdir(directory)}")

            # Check if there is data in database
            with hf.File(f"{directory}/trainer.h5", "r") as db:
                db_loss = onp.array(db["loss"])
                db_accuracy = onp.array(db["accuracy"])

                class_loss = onp.array(train_recorder._results["loss"])
                class_accuracy = onp.array(train_recorder._results["accuracy"])

                assert db_loss.shape == (11, 1)
                assert class_loss.shape == (9, 1)
                testing.assert_raises(
                    AssertionError,
                    testing.assert_array_equal,
                    db_loss.sum(),
                    class_loss.sum(),
                )

                assert db_accuracy.shape == (11, 1)
                assert class_accuracy.shape == (9, 1)
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
        train_report = self.train_recorder.gather()
        test_report = self.test_recorder.gather()

        assert len(train_report["loss"]) == 10
        assert onp.sum(train_report["loss"]) > 0
        assert len(train_report["accuracy"]) == 10
        assert onp.sum(train_report["accuracy"]) > 0

        # Arrays should be resized now.
        assert len(test_report["loss"]) == 2
        assert onp.sum(test_report["loss"]) > 0
        assert len(test_report["accuracy"]) == 2
        assert onp.sum(test_report["accuracy"]) > 0

    def test_export_function_db(self):
        """
        Test that the reports are exported correctly.
        """
        with tempfile.TemporaryDirectory() as directory:
            new_model = copy.deepcopy(self.production_model)

            train_recorder = nl.training_recording.JaxRecorder(
                storage_path=directory,
                name="trainer",
                update_rate=1,
                chunk_size=11,
                measurements=[
                    Loss(apply_fn=nl.loss_functions.MeanPowerLoss(order=2)),
                    Accuracy(apply_fn=nl.accuracy_functions.LabelAccuracy()),
                ],
            )
            train_recorder.instantiate_recorder(
                data_set=self.data_generator.train_ds,
                model=new_model,
            )

            # Define the training strategy
            training_strategy = nl.training_strategies.SimpleTraining(
                model=new_model,
                loss_fn=nl.loss_functions.CrossEntropyLoss(),
                accuracy_fn=nl.accuracy_functions.LabelAccuracy(),
                recorders=[train_recorder],
            )

            # Retrain the model.
            training_strategy.train_model(
                train_ds=self.data_generator.train_ds,
                test_ds=self.data_generator.test_ds,
                batch_size=5,
                epochs=20,
            )

            report = train_recorder.gather()
            assert report["loss"].shape[0] == 20
            testing.assert_array_equal(
                report["loss"][11:], train_recorder._results["loss"]
            )
