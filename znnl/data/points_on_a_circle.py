"""
Module for generating points on a circle.
"""
from abc import ABC

import jax
import jax.numpy as np

from znnl.data.data_generator import DataGenerator
from znnl.utils.prng import PRNGKey


class PointsOnCircle(DataGenerator, ABC):
    """Class to generate points on circles."""

    def __init__(self, radius=np.array([1.0]), noise: float = 1e-3, seed: int = None):
        """
        Constructor for points on circles.

        Parameters
        ----------
        radius : np.ndarray
                Euclidean distance from origin.
        noise : float
                Maximum allowed deviation from the radius.
        seed : int, default None
                Random seed.
        """
        self.radius = radius
        self.noise = noise
        self.data_pool = None

        self.rng = PRNGKey(seed)

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
                radial_values = jax.random.uniform(
                    key=self.rng(),
                    shape=(n_points,),
                    dtype=float,
                    minval=radius - self.noise,
                    maxval=radius + self.noise,
                )
                pool.append(
                    (radial_values * np.array([np.cos(angles), np.sin(angles)])).T
                )
        else:
            for radius in self.radius:
                pool.append((radius * np.array([np.cos(angles), np.sin(angles)])).T)
        self.data_pool = np.array(pool).reshape([n_points * len(self.radius), 2])

    def build_pool(self, n_points: int, noise: bool = False, method: str = "uniform"):
        """
        Build the data pool for points on a circle.

        Parameters
        ----------
        n_points : int
                Number of points to add to the circle.
        method : str (default=uniform)
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
