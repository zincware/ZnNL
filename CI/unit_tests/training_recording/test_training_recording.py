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

import tempfile
from pathlib import Path

import h5py as hf
import numpy as onp
from numpy import testing

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
            loss_ntk=True,
            loss_ntk_eigenvalues=True,
            loss_ntk_entropy=True,
        )
        recorder.instantiate_recorder(data_set=self.dummy_data_set)
        _exclude_list = [
            "_accuracy_fn",
            "_loss_fn",
            "update_rate",
            "name",
            "storage_path",
            "chunk_size",
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
