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
Module for the MNIST dataset.
"""
from znrnd.data_sets.data_set import DataSet
import tensorflow_datasets as tfds


class MNIST(DataSet):
    """
    MNIST dataset.
    """
    def __init__(
            self, 
            ds_size: int = 500, 
            one_hot_encoding: bool = True, 
            ttv: tuple = (0.5, 0.3, .2)
        ):
        """
        Constructor for the MNIST dataset.

        Parameters
        ----------
        ds_size : int (default = 500)
                Size of the dataset to download.
        one_hot_encoding : bool (default = True)
                If true, the targets will be one-hot encoded.
        ttv : tuple (0.5, 0.3, 0.2)
                Train, test, validate split of the dataset.
        """
        self.ds_size = ds_size
        self.one_hot_encoding = one_hot_encoding
        self.ttv = ttv

        self.train_ds, self.test_ds, self.validate_ds = tfds.as_numpy(
            tfds.load(
            "mnist:3.*.*",
            split=[f"train[:{}]", f"test[:{}]", f"validate[:{}]"],
            batch_size=-1,
            )
        )

        
