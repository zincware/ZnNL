"""
module for generating points on a circle
"""
from abc import ABC
from znrnd.core.data.data_generator import DataGenerator
import random
import numpy as np


class PointsOnCircle(DataGenerator, ABC):
    """
    class to generate points on circles
    """

    def __init__(self, radius=None, noise: float = 1e-3):
        """
        constructor for points on circles

        Parameters
        ----------
        radius : np.ndarray
                euclidian distance from origin
        noise : float
                maximum allowed deviation from the radius
        """
        if radius is None:
            radius = np.array[1.0]
        self.radius = radius
        self.noise = noise
        self.data_pool = None

    def uniform_sampling(self, n_points: int, noise: bool = False):
        """
        Generate the points uniformly.

        Parameters
        ----------
        n_points : int
                Number of points to generate.
        noise : bool
                If true, noise will be added to the radial values.

        Returns
        -------
        Updates the data pool in the class.
        """
        pool = []
        angles = np.linspace(0, 2 * np.pi, num=n_points, endpoint=False)
        if noise:
            for radius in self.radius:
                radial_values = np.random.uniform(
                    radius - self.noise, radius + self.noise, n_points
                )
                pool.append(
                    (radial_values * np.array([np.cos(angles), np.sin(angles)])).T
                )
        else:
            for radius in self.radius:
                pool.append((radius * np.array([np.cos(angles), np.sin(angles)])).T)
        self.data_pool = np.array(pool).reshape([n_points * len(self.radius), 2])

    def build_pool(self, method: str, n_points: int, noise: bool = False):
        """
        Build the data pool for points on a circle

        Parameters
        ----------
        n_points : int
                Number of points to add to the circle.
        method : str
                Method with which to compute the points.
        noise : bool
                If true, noise will be added to the data.

        Returns
        -------
        Will call a method which updates the class state.
        """
        method_dict = {"uniform": self.uniform_sampling}
        chosen_method = method_dict[method]
        chosen_method(n_points=n_points, noise=noise)

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
