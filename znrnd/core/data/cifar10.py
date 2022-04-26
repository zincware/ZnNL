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
Data generator for the CIFAR 10 dataset.
"""
import tensorflow_datasets as tfds

from znrnd.core.data.data_generator import DataGenerator


class CIFAR10Generator(DataGenerator):
    """
    Data generator for MNIST datasets
    """

    def __init__(self):
        """
        Constructor for the MNIST generator class.
        """
        self.ds_train, self.ds_test = tfds.as_numpy(
            tfds.load(
                "cifar10:3.*.*",
                split=["train[:%d]" % 500, "test[:%d]" % 500],
                batch_size=-1,
            )
        )
        self.data_pool = self.ds_train["image"].astype(float)
