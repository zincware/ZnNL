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
Abalone dataset generator.
"""
import urllib.request
import zipfile

import pandas as pd

from znnl.data.data_generator import DataGenerator


class AbaloneDataGenerator(DataGenerator):
    """
    Generator for the Abalone data-set.
    """

    def __init__(self, train_fraction: float):
        """
        Constructor for the abalone dataset.

        Parameters
        ----------
        train_fraction : float
            Fraction of the data to use for training.
        """
        self._load_data()

        self.data_file = "abalone.data"
        self.columns = [
            "Sex",
            "Length",
            "Diameter",
            "Height",
            "Whole weight",
            "Shucked weight",
            "Viscera weight",
            "Shell weight",
            "Rings",
        ]

        # Collect the processed data
        processed_data = self._process_raw_data()

        # Create the data-sets
        train_ds = processed_data.sample(frac=train_fraction, random_state=0)
        train_labels = train_ds.pop("Rings")

        test_ds = processed_data.drop(train_ds.index)
        test_labels = test_ds.pop("Rings")

        self.train_ds = {
            "inputs": train_ds.to_numpy(),
            "targets": train_labels.to_numpy().reshape(-1, 1),
        }
        self.test_ds = {
            "inputs": test_ds.to_numpy(),
            "targets": test_labels.to_numpy().reshape(-1, 1),
        }

        self.data_pool = self.train_ds["inputs"]

    def _load_data(self):
        """
        Download the data.
        """
        filehandle, _ = urllib.request.urlretrieve(
            "http://archive.ics.uci.edu/static/public/1/abalone.zip"
        )
        with zipfile.ZipFile(filehandle, "r") as zip_ref:
            zip_ref.extractall()

    def _process_raw_data(self):
        """
        Process the raw data
        """
        # Process the raw data.
        raw_data = pd.read_csv(
            self.data_file,
            names=self.columns,
            na_values="?",
            comment="#",
            sep=",",
            skipinitialspace=True,
        )
        raw_data.dropna()

        # encode the sex data
        raw_data = pd.get_dummies(raw_data, columns=["Sex"], prefix="", prefix_sep="")
        # Normalize
        raw_data = (raw_data - raw_data.mean()) / raw_data.std()

        return raw_data
