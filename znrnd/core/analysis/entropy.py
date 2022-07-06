"""
ZnRND: A zincwarecode package.

License
-------
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the zincwarecode Project.

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
Module for the analysis of the entropy of a matrix.
"""
import jax.numpy as np

from znrnd.core.utils.matrix_utils import compute_eigensystem


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

        log_values = np.log(self.eigenvalues)

        entropy = self.eigenvalues * log_values

        if effective:
            maximum_entropy = np.log(len(self.eigenvalues))
            entropy /= maximum_entropy

        return -1 * entropy.sum()
