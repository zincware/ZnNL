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
import plotly.express as px

from znnl.utils import compute_eigensystem


class EigenSpaceAnalysis:
    """
    Analyse the Eigen space of the matrix.
    """

    def __init__(self, matrix: np.ndarray):
        """
        Constructor for the Eigen space class.

        Parameters
        ----------
        matrix : np.ndarray
                Matrix for which the analysis should take place.
        """
        self.matrix = matrix

    def compute_eigenvalues(self, normalize: bool = True) -> np.ndarray:
        """
        Compute the eigenvalues of the matrix.

        Parameters
        ----------
        normalize : bool (default = False)
                If true, the result is divided by the zeroth axis size of the matrix.
                Normalizing destroys this process anyway so it is defaulted to False
                for the occasion in which you wish to compare un-normalized eigenvalues
                directly.
        normalize : bool
                If true, apply the sum to one interpretation of the eigenvalues.
        Returns
        -------
        eigenvalues : np.ndarray
                Eigenvalues of the system.
        """
        eigenvalues, eigenvectors = compute_eigensystem(
            self.matrix, normalize=normalize
        )

        return eigenvalues

    def compute_eigenvalue_density(self, n_bins: int = 500):
        """
        Compute the eigenvalue density of the matrix.

        Parameters
        ----------
        n_bins : int
                Number of bins to use in the histogram.

        Returns
        -------
        Plots the histogram of eigenvalues.
        """
        eigenvalues = np.real(self.compute_eigenvalues())

        fig = px.histogram(eigenvalues, nbins=n_bins)
        fig.show()

    def compute_eigenvalue_spacing_density(self, n_bins: int = 500):
        """
        Compute the density of the spacing between the eigenvalues for comparison with
        the Wigner surmise.

        Parameters
        ----------
        n_bins : int
                Number of bins to use in the histogram.

        Returns
        -------
        Plots the histogram of eigenvalue separation.

        Notes
        -----
        https://robertsweeneyblanco.github.io/Computational_Random_Matrix_Theory/Eigenvalues/Wigner_Surmise.html
        """
        eigenvalues = np.real(self.compute_eigenvalues())

        spacing = np.diff(eigenvalues)

        fig = px.histogram(spacing, nbins=n_bins)
        fig.show()
