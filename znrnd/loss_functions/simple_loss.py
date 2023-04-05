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
Module for the simple loss for TensorFlow.
"""
from typing import Union
import jax.numpy as np

from znrnd.distance_metrics.distance_metric import DistanceMetric

from znrnd.utils import AnalysisModule


class SimpleLoss(AnalysisModule):
    """
    Class for the simple loss.

    Attributes
    ----------
    metric : DistanceMetric
    """

    def __signature__(self, data_set: dict) -> tuple:
        """
        Signature of the output of the loss.

        Returns
        -------
        signature : tuple
                In this case, the output is always (1, )
        """
        return (1, )
    
    def compute_fn(self, data_set: dict) -> Union[np.ndarray, float]:
        """
        Compute function for the loss methods.

        In this case, they can just call the __call__ method of the 
        loss function. 

        Parameters
        ----------
        data_set : dict
                Data set on which to compute the loss.

        Returns
        -------
        loss : float
                The float loss value.
        
        Notes
        -----
        This method should take over all call methods throughout the code.
        For this, we would need a cleanout of classes so it should be done 
        at a later stage.
        """
        return super().compute_fn(data_set)

    def __init__(self):
        """
        Constructor for the simple loss parent class.
        """
        self.metric: DistanceMetric = None        

    def __call__(self, point_1: np.array, point_2: np.array) -> float:
        """
        Summation over the tensor of the respective similarity measurement
        Parameters
        ----------
        point_1 : np.array
                first neural network representation of the considered points
        point_2 : np.array
                second neural network representation of the considered points

        Returns
        -------
        loss : float
                total loss of all points based on the similarity measurement
        """
        return np.mean(self.metric(point_1, point_2), axis=0)
