"""
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincware Project.

Description: Module for the particles in a box generator class.
"""
from abc import ABC
from pyrnd.core.data.data_generator import DataGenerator


class ConfinedParticles(DataGenerator, ABC):
    """
    A class to generate data for particles in a box.
    """

    def __init__(self):
        """
        Constructor for the ConfinedParticles data generator.
        """
        pass

    def get_points(self, n_points: int):
        """
        Fetch N points from the data pool.

        Parameters
        ----------
        n_points : int
                Number of points to fetch.

        Returns
        -------

        """
        pass
