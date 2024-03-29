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

import numpy as onp

from znnl.data.data_generator import DataGenerator


class PointsOnLattice(DataGenerator, ABC):
    """
    class to generate points on a circle
    """

    def __init__(self):
        """
        constructor for points on a lattice.
        """
        self.data_pool = None

    def build_pool(
        self,
        x_points: int = 11,
        y_points: int = 11,
        x_boundary: float = -1,
        y_boundary: float = -1,
    ):
        """
        Build the data pool for points on a square lattice with spacing 1.

        Parameters
        ----------
        x_points : int
                Number of points in the x direction.
        y_points : int
                Number of points in the y direction.
        x_boundary : float (default = -1)
                Symmetric boundary of the lattice in x-direction.
                Defines the min and max value of the lattice.
                If -1, boundary is scaled by the number of points.
        y_boundary : float (default = -1)
                Symmetric boundary of the lattice in y-direction.
                Defines the min and max value of the lattice.
                If -1, boundary is scaled by the number of points.

        Returns
        -------
        Will call a method which updates the class state.
        """
        if x_boundary == -1:
            x_boundary = (x_points - 1) / 2
        if y_boundary == -1:
            y_boundary = (y_points - 1) / 2

        x = onp.linspace(-x_boundary, x_boundary, x_points, dtype=float)
        y = onp.linspace(-y_boundary, y_boundary, y_points, dtype=float)

        grid = onp.stack(onp.meshgrid(x, y), axis=2)
        self.data_pool = grid.reshape(-1, grid.shape[-1])
