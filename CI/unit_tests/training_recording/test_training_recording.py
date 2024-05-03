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

import copy
import tempfile
from pathlib import Path

import h5py as hf
import jax.numpy as np
import numpy as onp
import optax
from neural_tangents import stax
from numpy import testing

from znnl import models
from znnl.loss_functions import MeanPowerLoss
from znnl.training_recording import JaxRecorder


class TestModelRecording:
    """
    Unit test suite for the model recording.
    """

    @classmethod
    def setup_class(cls):
        """
        Prepare the test suite.
        """
        dummy_data = onp.random.uniform(size=(5, 2, 3))
        cls.dummy_data_set = {"inputs": dummy_data, "targets": dummy_data}

        # Create some test data for class specific recording.
        class_specific_input = onp.random.uniform(size=(10, 2, 3))
        class_specific_target = np.concatenate([np.eye(5), np.eye(5)], axis=0)
        predictions = onp.random.uniform(size=(10, 5))
        grads = onp.random.uniform(size=(10, 100))
        ntk = np.einsum("ij, kj -> ik", grads, grads)
        cls.class_specific_data = {
            "inputs": class_specific_input,
            "targets": class_specific_target,
        }
        cls.class_specific_parsed_data = {
            "ntk": ntk,
            "predictions": predictions,
            "targets": class_specific_target,
        }

        # Create a model
        cls.architecture = stax.serial(
            stax.Flatten(), stax.Dense(8), stax.Relu(), stax.Dense(5)
        )

    def test_instantiation(self):
        """
        Test the instantiation of the recorder.
        """

        recorder = JaxRecorder(
            loss=True,
            accuracy=True,
            ntk=True,
            covariance_ntk=True,
            magnitude_ntk=True,
            entropy=True,
            magnitude_entropy=True,
            magnitude_variance=True,
            covariance_entropy=True,
            eigenvalues=True,
            trace=True,
            loss_derivative=True,
            network_predictions=True,
            entropy_class_correlations=True,
        )
        recorder.instantiate_recorder(data_set=self.dummy_data_set)
        _exclude_list = [
            "_accuracy_fn",
            "_loss_fn",
            "update_rate",
            "name",
            "storage_path",
            "chunk_size",
            "flatten_ntk",
            "class_specific",
        ]
        for key, val in vars(recorder).items():
            if key[0] != "_" and key not in _exclude_list:
                assert val is True
            if key == "update_rate":
                assert val == 1
            elif key.split("_")[-1] == "array:":
                assert val == []
            elif key == "_selected_properties":
                pass

    def test_data_dump(self):
        """
        Test that data is dumped correctly.
        """
        with tempfile.TemporaryDirectory() as directory:
            recorder = JaxRecorder(
                storage_path=directory,
                name="my_recorder",
                loss=True,
                accuracy=False,
                ntk=False,
                covariance_ntk=True,
                magnitude_ntk=True,
                entropy=False,
                magnitude_entropy=False,
                magnitude_variance=False,
                covariance_entropy=False,
                eigenvalues=False,
            )
            recorder.instantiate_recorder(data_set=self.dummy_data_set)

            # Add some dummy data.
            test_data = onp.random.uniform(size=(200,))
            recorder._loss_array = test_data.tolist()

            recorder.dump_records()  # dump to disk

            # Check that the dump worked.
            assert Path(f"{directory}/my_recorder.h5").exists()
            with hf.File(f"{directory}/my_recorder.h5", "r") as db:
                testing.assert_almost_equal(db["loss"], test_data, decimal=7)

    def test_overwriting(self):
        """
        Test the overwrite function.
        """
        recorder = JaxRecorder(
            loss=False, accuracy=False, ntk=True, entropy=False, eigenvalues=False
        )
        recorder.instantiate_recorder(data_set=self.dummy_data_set)

        # Populate the arrays deliberately.
        recorder._ntk_array = onp.random.uniform(size=(10, 5, 5)).tolist()
        assert onp.sum(recorder._ntk_array) != 0.0  # check the data is there

        # Check normal resizing on instantiation.
        recorder.instantiate_recorder(data_set=self.dummy_data_set, overwrite=False)
        assert onp.shape(recorder._ntk_array) == (10, 5, 5)

        # Test overwriting.
        recorder.instantiate_recorder(data_set=self.dummy_data_set, overwrite=True)
        assert recorder._ntk_array == []

    def test_magnitude_variance(self):
        """
        Test the magnitude variance function.
        """
        recorder = JaxRecorder(
            loss=False,
            accuracy=False,
            ntk=False,
            entropy=False,
            magnitude_variance=True,
            eigenvalues=False,
        )
        recorder.instantiate_recorder(data_set=self.dummy_data_set)

        # Create some test data.
        data = onp.random.uniform(1.0, 2.0, size=(100))
        ntk = onp.eye(100) * data
        # calculate the magnitude variance
        recorder._update_magnitude_variance(parsed_data={"ntk": ntk})
        # calculate the expected variance
        expected_variance = onp.var(onp.sqrt(data) / onp.sqrt(data).mean())
        # check that the variance is correct
        testing.assert_almost_equal(
            recorder._magnitude_variance_array, expected_variance
        )

    def test_index_identification(self):
        """
        Test the index identification of class specific recording.
        """
        # Test one-hot encoding.
        recorder = JaxRecorder(
            class_specific=True,
            loss=False,
            accuracy=False,
            ntk=False,
            entropy=False,
            magnitude_variance=True,
            eigenvalues=False,
        )
        recorder.instantiate_recorder(data_set=self.class_specific_data)
        class_labels, class_indices = recorder._class_idx
        assert np.all(class_labels == np.arange(5))
        assert np.all(np.array(class_indices) == np.arange(10).reshape(2, 5).T)

        # Test non-one-hot encoding.
        recorder = JaxRecorder(
            class_specific=True,
            loss=False,
            accuracy=False,
            ntk=False,
            entropy=False,
            magnitude_variance=True,
            eigenvalues=False,
        )
        dummy_data_input = onp.random.uniform(size=(10, 2, 3))
        x = np.expand_dims(np.arange(5), axis=1)
        dummy_data_target = np.concatenate([x, x], axis=0)
        dummy_data_set = {"inputs": dummy_data_input, "targets": dummy_data_target}
        recorder.instantiate_recorder(data_set=dummy_data_set)
        class_labels, class_indices = recorder._class_idx
        assert np.all(class_labels == x)
        assert np.all(np.array(class_indices) == np.arange(10).reshape(2, 5).T)

    def test_class_specific_update_fn(self):
        """
        Test the class specific update function.
        """
        recorder = JaxRecorder(
            class_specific=True,
            ntk=True,
            trace=True,
            eigenvalues=True,
        )

        # Instantiate the recorder
        recorder.instantiate_recorder(data_set=self.class_specific_data)
        assert recorder._class_idx[1][0].tolist() == [0, 5]

        # Test trace update
        recorder.class_specific_update_fn(
            call_fn=recorder._update_trace,
            indices=recorder._class_idx[1][0],
            parsed_data=self.class_specific_parsed_data,
        )
        # Check that the trace has been selected correctly.
        ntk = self.class_specific_parsed_data["ntk"]
        assert np.array(recorder._trace_array) == (ntk[0, 0] + ntk[5, 5]) / 2

        # Test eigenvalues update
        recorder.class_specific_update_fn(
            call_fn=recorder._update_eigenvalues,
            indices=recorder._class_idx[1][0],
            parsed_data=self.class_specific_parsed_data,
        )
        # Check shape of the eigenvalues
        assert np.shape(recorder._eigenvalues_array) == (1, 2)

    def test_multi_class_update(self):
        """
        Test the multi-class update function.
        """
        recorder = JaxRecorder(
            class_specific=True,
            loss=True,
            ntk=True,
            entropy=True,
            trace=True,
            magnitude_variance=True,
            eigenvalues=True,
        )

        # Instantiate the recorder
        recorder._loss_fn = MeanPowerLoss(order=2)
        recorder.instantiate_recorder(data_set=self.class_specific_data)

        # Define a model
        model = models.NTModel(
            nt_module=self.architecture,
            optimizer=optax.adam(learning_rate=0.01),
            input_shape=(1, 2, 3),
        )

        # Test trace update
        recorder.update_recorder(epoch=1, model=model)

        # Check the shape of the arrays
        assert np.shape(recorder._loss_array) == (1, 5)
        assert np.shape(recorder._entropy_array) == (1, 5)
        assert np.shape(recorder._trace_array) == (1, 5)
        assert np.shape(recorder._magnitude_variance_array) == (1, 5)
        assert np.shape(recorder._eigenvalues_array) == (1, 10)
        # Even though NTK is selected, it should not be updated.
        assert np.shape(recorder._ntk_array) == ()

        # Update the recorder again
        recorder.update_recorder(epoch=2, model=model)

        # Check the shape of the arrays
        assert np.shape(recorder._loss_array) == (2, 5)
        assert np.shape(recorder._entropy_array) == (2, 5)
        assert np.shape(recorder._trace_array) == (2, 5)
        assert np.shape(recorder._magnitude_variance_array) == (2, 5)
        assert np.shape(recorder._eigenvalues_array) == (2, 10)
        # Even though NTK is selected, it should not be updated.
        assert np.shape(recorder._ntk_array) == ()

    def test_class_combinations(self):
        """
        Test the class combinations method.

        This method is used to calculate the combinations of all classes.
        """
        recorder = JaxRecorder(
            entropy_class_correlations=True,
        )

        # Instantiate the recorder
        recorder.instantiate_recorder(data_set=copy.deepcopy(self.class_specific_data))

        # Test for one-hot encoding
        recorder._data_set["targets"] = np.concatenate([np.eye(3), np.eye(3)], axis=0)
        _, class_combinations = recorder._get_class_combinations()
        assert np.all(np.array(class_combinations[:3]) == np.arange(3).reshape(3, 1))
        assert np.all(
            np.array(class_combinations[3:6]) == np.array([[0, 1], [0, 2], [1, 2]])
        )
        assert np.all(np.array(class_combinations[6:]) == np.array([[0, 1, 2]]))

        # Test for non-one-hot encoding
        recorder._data_set["targets"] = np.concatenate(
            [np.arange(3), np.arange(3)], axis=0
        ).reshape(6, 1)
        _, class_combinations = recorder._get_class_combinations()
        assert np.all(np.array(class_combinations[:3]) == np.arange(3).reshape(3, 1))
        assert np.all(
            np.array(class_combinations[3:6]) == np.array([[0, 1], [0, 2], [1, 2]])
        )
        assert np.all(np.array(class_combinations[6:]) == np.array([[0, 1, 2]]))

        # Test for non-consecutive classes
        idx = np.array([0, 2, 3])
        recorder._data_set["targets"] = np.concatenate([idx, idx], axis=0).reshape(6, 1)
        _, class_combinations = recorder._get_class_combinations()
        assert np.all(np.array(class_combinations[:3]) == np.arange(3).reshape(3, 1))
        assert np.all(
            np.array(class_combinations[3:6]) == np.array([[0, 1], [0, 2], [1, 2]])
        )
        assert np.all(np.array(class_combinations[6:]) == np.array([[0, 1, 2]]))

    def test_entropy_class_correlation(self):
        """
        Test the entropy class correlation method.
        """
        recorder = JaxRecorder(
            entropy_class_correlations=True,
        )

        # Instantiate the recorder
        recorder._loss_fn = MeanPowerLoss(order=2)
        recorder.instantiate_recorder(data_set=self.class_specific_data)

        # Define a model
        model = models.NTModel(
            nt_module=self.architecture,
            optimizer=optax.adam(learning_rate=0.01),
            input_shape=(1, 2, 3),
        )

        # Test the correlation
        recorder.update_recorder(epoch=1, model=model)
        assert np.shape(recorder._entropy_class_correlations_array) == (1, 31)

        # Update the recorder again
        recorder.update_recorder(epoch=2, model=model)
        assert np.shape(recorder._entropy_class_correlations_array) == (2, 31)

    def test_read_class_specific_data(self):
        """
        Test the reading of class specific data.
        """
        recorder = JaxRecorder(
            class_specific=True,
            ntk=True,
            trace=True,
            eigenvalues=True,
        )

        # Instantiate the recorder
        recorder.instantiate_recorder(data_set=self.class_specific_data)

        # Test the case of having one entry per sample, e.g. for recording the 
        # eigenvalues.
        test_record = np.arange(20).reshape(2, 10)
        class_specific_dict = recorder.read_class_specific_data(test_record)
        for i, (key, val) in enumerate(class_specific_dict.items()):
            assert key == i
            print(val)
            assert np.all(val == np.array([[i, i + 5], [i+10, i + 15]]))

        # Test the case of having one entry per class, e.g. for recording the trace.
        test_record = np.arange(10).reshape(2, 5)
        class_specific_dict = recorder.read_class_specific_data(test_record)
        for i, (key, val) in enumerate(class_specific_dict.items()):
            assert key == i
            assert np.all(val == np.array([i, i + 5]))