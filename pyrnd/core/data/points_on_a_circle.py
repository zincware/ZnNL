"""
module for generating points on a circle
"""
from abc import ABC
from pyrnd.core.data.data_generator import DataGenerator
import random
import numpy as np


class PointsOnCircle(DataGenerator, ABC):
    """
    class to generate points on a circle
    """

    def __init__(self, radius: float = 1.0, noise: float = 1e-3):
        """
        constructor for points on a circle

        Parameters
        ----------
        radius : float
                euclidian distance from origin
        noise : float
                maximum allowed deviation from the radius
        """
        self.radius = radius
        self.noise = noise
        self.data_pool = None

    def build_pool(self, n_points: int):
        """
        Build the data pool for points on a circle
        Parameters
        ----------
        n_points

        Returns
        -------

        """
        angles = np.linspace(0, 2*np.pi, num=n_points)
        self.data_pool = (self.radius*np.array([np.cos(angles), np.sin(angles)])).T

    def get_points(self, n_points: int) -> np.ndarray:
        """
        Fetch N points from the data pool.

        Parameters
        ----------
        n_points : int
                Number of points to fetch.

        Returns
        -------
        data : np.ndarray
                A numpy array of data points.
        """

        if n_points == -1:
            return self.data_pool
        else:
            indices = random.sample(range(0, len(self.data_pool) - 1), n_points)
            data = self.data_pool[indices]
            return data
