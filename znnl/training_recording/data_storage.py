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

from dataclasses import dataclass
from pathlib import Path

import h5py as hf
import numpy as onp


class DataStorage:
    """
    Parent class for the HDF5 data storage devices.
    """

    def __init__(self, database_path: Path):
        """
        Constructor for the data storage class.

        Parameters
        ----------
        database_path : Path
                File path to the data storage.
        """
        self.database_path = f"{database_path}.h5"

    def _resize_dataset(self, _data_group: str, _chunk_size: int):
        """
        Resize a dataset.

        Parameters
        ----------
        _data_group : str
                Data group to resize
        _chunk_size : int
                Size of the resize.

        Returns
        -------
        Resizes a hdf5 dataset.
        """
        with hf.File(self.database_path, "a") as db:
            current_size = len(db[_data_group])
            db[_data_group].resize(int(_chunk_size + current_size), axis=0)

    def _write_to_dataset(self, data_group: str, data: onp.ndarray):
        """
        Write a numpy array to a dataset.

        Parameters
        ----------
        data_group : str
                Group in the database to which the data belongs.
        data : onp.ndarray
                Data to be stored in the group.

        Returns
        -------
        Adds data to a dataset.
        """
        with hf.File(self.database_path, "a") as db:
            try:
                current_size = len(db[data_group])
                # add a chunk to the dataset.
                self._resize_dataset(data_group, data.shape[0])
                db[data_group][current_size:] = data

            # Add the group if it doesn't exist.
            except KeyError:
                if len(data.shape) == 1:
                    max_shape = (None,)
                else:
                    max_shape = (None,) + data.shape[1:]

                db.create_dataset(name=data_group, shape=data.shape, maxshape=max_shape)
                db[data_group][:] = data

    def fetch_data(self, dataset_list: list):
        """
        Collect data from the database.

        Parameters
        ----------
        dataset_list : list
                List of datasets to load.
        """
        collected_data = {}

        with hf.File(self.database_path, "r") as db:
            for item in dataset_list:
                collected_data[item] = onp.array(db[item])

        return collected_data

    def write_data(self, data: dataclass):
        """
        Save new data to the database.

        Parameters
        ----------
        data : dataclass
                Exported dataclass to write to the disk.

        Returns
        -------
        Updates the database.
        """
        for item, value in data.__dict__.items():
            self._write_to_dataset(item, value)
