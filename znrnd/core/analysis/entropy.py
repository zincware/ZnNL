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
    eigenvalues: np.ndarray = None

    def __init__(self, matrix: np.ndarray):
        """
        Constructor for the entropy analysis class.
        """
        self.matrix = matrix

    def _compute_eigensystem(self):
        """
        Compute the eigensystem of the matrix.

        Returns
        -------
        eigenvalues
        eigenvectors
        """
        # Computes the reduced eigen-system
        eigenvalues, eigenvectors = compute_eigensystem(self.matrix)

        self.eigenvalues = eigenvalues

    def compute_von_neumann_entropy(self, normalize: bool = True):
        """
        Compute the von-Neumann entropy of the matrix.

        Parameters
        ----------
        normalize : bool (default=True)
                If true, the entropy is divided by the theoretical maximum entropy of
                the system.

        Returns
        -------
        entropy : float
            Entropy of the matrix.
        """
        if self.eigenvalues is None:
            self._compute_eigensystem()

        log_vals = np.log(self.eigenvalues)

        entropy = self.eigenvalues * log_vals

        if normalize:
            max = -1 * len(self.eigenvalues) * np.log(1 / len(self.eigenvalues))

            entropy /= max

        return -1 * entropy.sum()
