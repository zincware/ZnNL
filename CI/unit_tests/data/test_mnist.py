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
Test MNIST generator.
"""

from znnl.data import MNISTGenerator


class TestMNISTGenerator:
    """
    Class for testing the MNIST generator.
    """

    def test_one_hot_creation(self):
        """
        Test if one can create the generator.
        """
        generator = MNISTGenerator(ds_size=500)

        assert generator is not None
        assert generator.train_ds["inputs"].shape == (500, 28, 28, 1)
        assert generator.train_ds["targets"].shape == (500, 10)

        assert generator.test_ds["inputs"].shape == (500, 28, 28, 1)
        assert generator.test_ds["targets"].shape == (500, 10)

    def test_serial_creation(self):
        """
        Test if one can create the generator.
        """
        generator = MNISTGenerator(ds_size=500, one_hot_encoding=False)

        assert generator is not None
        assert generator.train_ds["inputs"].shape == (500, 28, 28, 1)
        assert generator.train_ds["targets"].shape == (500, 1)

        assert generator.test_ds["inputs"].shape == (500, 28, 28, 1)
        assert generator.test_ds["targets"].shape == (500, 1)
