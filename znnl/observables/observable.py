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
Observable parent class module.
"""
from typing import Union

import jax.numpy as np


class Observable:
    """
    Parent class for all observables.
    """

    @classmethod
    def __name__(self) -> str:
        """
        Name of the observable.

        Returns
        -------
        name : str
                The name of the observable.
        """
        raise NotImplementedError("Implemented in child class.")

    @classmethod
    def __signature__(self, data_set: dict) -> tuple:
        """
        Name of the observable.

        Parameters
        ----------
        data_set : dict
                The data set to be used for the observable.

        Returns
        -------
        signature : tuple
                The signature of the observable.
        """
        raise NotImplementedError("Implemented in child class.")

    @classmethod
    def __call__(self, data_set: dict) -> Union[str, np.ndarray, float]:
        """
        Compute the observable.

        Parameters
        ----------
        data_set : dict
                The data set to be used for the observable.

        Returns
        -------
        value : Union[str, np.ndarray, float]
                The value of the observable.

        """
        raise NotImplementedError("Implemented in child class.")
