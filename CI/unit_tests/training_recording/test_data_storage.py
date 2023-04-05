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
from dataclasses import dataclass
from os import path
from pathlib import Path

import h5py as hf
import numpy as onp
from numpy import testing

from znnl.training_recording import DataStorage


@dataclass
class DataClass:
    """
    Dummy data class for testing
    """

    vector_data: onp.ndarray
    tensor_data: onp.ndarray


class TestDataStorage:
    """
    Test suite for the storage module.
    """

    @classmethod
    def setup_class(cls):
        """
        Set up the test.
        """
        cls.vector_data = onp.random.uniform(size=(100,))
        cls.tensor_data = onp.random.uniform(size=(100, 10, 10))

        cls.data_object = DataClass(
            vector_data=cls.vector_data, tensor_data=cls.tensor_data
        )

    def test_database_construction(self):
        """
        Test that database groups are built properly.
        """
        # Create temporary directory for safe testing.
        with tempfile.TemporaryDirectory() as directory:
            database_path = path.join(directory, "test_creation")
            data_storage = DataStorage(Path(database_path))
            data_storage.write_data(self.data_object)  # write some data to empty DB.

            with hf.File(data_storage.database_path, "r") as db:
                # Test correct dataset creation.
                keys = list(db.keys())
                testing.assert_equal(keys, ["tensor_data", "vector_data"])
                vector_data = onp.array(db["vector_data"])
                tensor_data = onp.array(db["tensor_data"])

                # Check data structure within the db.
                assert vector_data.shape == (100,)
                assert vector_data.sum() != 0.0

                assert tensor_data.shape == (100, 10, 10)
                assert tensor_data.sum() != 0.0

    def test_resize_dataset_standard(self):
        """
        Test if the datasets are resized properly.
        """
        with tempfile.TemporaryDirectory() as directory:
            database_path = path.join(directory, "test_resize")
            data_storage = DataStorage(Path(database_path))
            data_storage.write_data(self.data_object)  # write some data to empty DB.
            data_storage.write_data(self.data_object)  # force resize.

            with hf.File(data_storage.database_path, "r") as db:
                # Test correct dataset creation.
                vector_data = onp.array(db["vector_data"])
                tensor_data = onp.array(db["tensor_data"])

                # Check data structure within the db.
                assert vector_data.shape == (200,)
                assert vector_data[100:].sum() != 0.0

                assert tensor_data.shape == (200, 10, 10)
                assert tensor_data[100:].sum() != 0.0
