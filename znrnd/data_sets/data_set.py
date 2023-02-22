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
Parent class for a ZnRND dataset.
"""
from typing import Any


class DataSet:
    """
    Parent class for ZnRND dataset.
    """

    def __len__(self) -> int:
        """
        Return the length of the dataset.

        Returns
        -------
        length : int
                Length of the dataset.
        """
        raise NotImplementedError("Implemented in child class.")

    def __getitem__(self, index: int) -> Any:
        """
        Get an item from the dataset.

        Parameters
        ----------
        index : int
                Index of the iterm to collect.

        Returns
        -------
        item : Any
                Some object loaded from a data structure.
        """
        raise NotImplementedError("Implemented in child class.")
