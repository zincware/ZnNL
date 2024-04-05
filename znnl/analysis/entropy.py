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

import jax.numpy as np

from znnl.utils import compute_eigensystem


class EntropyAnalysis:
    """
    Analyse the entropy of a matrix.
    """

    def __init__(self, matrix: np.ndarray):
        """
        Constructor for the entropy analysis class.
        """
        self.matrix = matrix
        self.eigenvalues: np.ndarray

    def _compute_eigensystem(self, normalize: bool = True):
        """
        Compute the eigensystem of the matrix.

        Returns
        -------
        eigenvalues : np.ndarray
                Populates the eigenvalues attribute of the class.
        """
        # Computes the reduced eigen-system
        self.eigenvalues, eigenvectors = compute_eigensystem(
            self.matrix, normalize=normalize
        )

    @staticmethod
    def compute_shannon_entropy(dist: np.ndarray, normalize: bool = False) -> float:
        """
        Compute the Shannon entropy of a given probability distribution.

        The Shannon entropy of a given probability distribution is computed using a
        mask to neglect encountered zeros in the logarithm.

        Parameters
        ----------
        dist : np.ndarray
                Array to calculate the entropy of.
        normalize : bool (default = False)
                If true, the Shannon entropy is normalized by re-scaling to the maximum
                entropy. The method will return a value between 0 and 1.

        Returns
        -------
        Entropy of the distribution
        """
        mask = np.nonzero(dist)
        scaled_values = -1 * dist[mask] * np.log(dist[mask])
        entropy = scaled_values.sum()

        if normalize:
            scale_factor = np.log(len(dist))
            entropy /= scale_factor

        return entropy

    def compute_von_neumann_entropy(
        self, effective: bool = True, normalize_eig: bool = True
    ) -> float:
        """
        Compute the von-Neumann entropy of the matrix.

        Parameters
        ----------
        effective : bool (default=True)
                If true, the entropy is divided by the theoretical maximum entropy of
                the system thereby returning the effective entropy.
        normalize_eig : bool (default = True)
                If true, the eigenvalues are scaled to look like probabilities.

        Returns
        -------
        entropy : float
            Entropy of the matrix.
        """
        # if self.eigenvalues is None:
        self._compute_eigensystem(normalize=normalize_eig)

        entropy = self.compute_shannon_entropy(self.eigenvalues)

        if effective:
            maximum_entropy = np.log(len(self.eigenvalues))
            entropy /= maximum_entropy

        return entropy
