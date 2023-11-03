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

import abc
from typing import List, Union

import jax.numpy as np


class PointSelection:
    """
    Parent class for the point selection module.

    Attributes
    ----------
    drawn_points : int
            Number of points drawn from the data pool.
    selected_points : int
            Number of points selected from the drawn points.
    """

    def __init__(self):
        """
        Constructor for the point selection class.
        """
        self.drawn_points = None
        self.selected_points = None

    @abc.abstractmethod
    def select_points(self, distances: np.ndarray) -> Union[List, None]:
        """
        Select points from the pool

        Returns
        -------
        points : list
                A set of points to be used by the RND class.
        """
        raise NotImplementedError("Implemented in child class.")
