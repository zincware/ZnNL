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
from abc import ABC


class Regularizer(ABC):
    """
    Parent class for a regularizer. All regularizers should inherit from this class.
    """

    def __init__(self, reg_factor) -> None:
        """
        Constructor of the regularizer class.

        Parameters
        ----------
        reg_factor : float
                Regularization factor.
        """
        self.reg_factor = reg_factor

    def __call__(self, params: dict, **kwargs: dict) -> float:
        """
        Call function of the regularizer class.

        Parameters
        ----------
        params : dict
                Parameters of the model.
        kwargs : dict
                Additional arguments. 
                Individual regularizers can define their own arguments.

        Returns
        -------
        reg_loss : float
                Loss contribution from the regularizer.
        """
        raise NotImplementedError
