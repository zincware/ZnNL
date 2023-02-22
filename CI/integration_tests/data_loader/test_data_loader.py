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
Integration test for the data loader.
"""
from znrnd.data_loading import DataLoader
from znrnd.


class TestDeployedDataLoader:
    """
    Integration test for the data loader.
    """
    @classmethod
    def setup_class(cls):
        """
        Prepare for the test.
        """

        cls.data_loader = DataLoader()
        cls.data_set = 
