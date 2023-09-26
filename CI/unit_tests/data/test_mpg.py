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
Test MPG generator.
"""
from znnl.data import MPGDataGenerator


class TestMPGGenerator:
    """
    Class for testing the MPG generator.
    """

    def test_creation(self):
        """
        Test if one can create the generator.
        """
        generator = MPGDataGenerator(train_fraction=0.8)

        assert generator is not None
        assert generator.train_ds["inputs"].shape == (314, 9)
        assert generator.train_ds["targets"].shape == (314, 1)

        assert generator.test_ds["inputs"].shape == (78, 9)
        assert generator.test_ds["targets"].shape == (78, 1)
