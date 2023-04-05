"""
ZnRND: A Zincwarecode package.

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
Module for the analysis class parent.

Notes
-----
Inheriting from this class enforces certain structure on the child class. While there
are no constraints so to speak, there are several methods that must be implemented in
order for the code to run.
"""
from abc import ABC, abstractmethod
import numpy as np
from typing import Union


class AnalysisModule(ABC):
    """
    The parent class for all ZnRND analysis modules.

    ZnRND supports many types of analysis during and after training. In order to 
    accomodate all analysis, this parent class enforces certain methods on its
    children such as a signature and compute functions. In doing so, all classes
    inheriting from this one can be used natively in the training recording.
    """
    @abstractmethod
    def __name__(self) -> str:
        """
        The name of the analsyis module.

        This name should be as detailed as possible.

        Returns
        -------
        name : str
                A string representation of the name.
        """
        raise NotImplementedError("Implemented in child class.")
    
    @abstractmethod
    def __signature__(self, data_set: dict) -> tuple:
        """
        The signature of the output of the analysis.

        Parameters
        ----------
        data_set : dict
                Data set on which the analysis will be performed
                during the recording time.

        Returns
        -------
        siganture : tuple
                Output siganture of the analysis.
        """
        raise NotImplementedError("Implemented in child class.")
    
    @abstractmethod
    def compute_fn(self, data_set: dict) -> Union[np.ndarray, float]:
        """
        Compute function of the analysis class.

        Parameters
        ----------
        data_set : dict
                Data set on which to perform the analysis.

        Returns
        -------
        analysis_output : Union[np.ndarray, float]
                The output of the analysis.
        """
        raise NotImplementedError("Implemented in child class.")
