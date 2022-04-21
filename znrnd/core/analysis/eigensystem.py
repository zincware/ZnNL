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
Module for the analysis of the eigensystem of a matrix.
"""
import jax.numpy as np
import plotly.express as px

from znrnd.core.utils.matrix_utils import compute_eigensystem


class EigenSpaceAnalysis:
    """
    Analyse the Eigenspace of the matrix.
    """

    def __init__(self, matrix: np.ndarray):
        """
        Constructor for the Eigenspace class.

        Parameters
        ----------
        matrix : np.ndarray
                Matrix for which the analysis should take place.
        """
        self.matrix = matrix

    def compute_eigenvalues(self, reduce: bool = True):
        """
        Compute the eigenvalues of the matrix.

        Parameters
        ----------
        reduce : bool (default = True)
                If true, the result is divided by the zeroth axis size of the matrix.
        Returns
        -------
        eigenvalues : np.ndarray
                Eigenvalues of the system.
        """
        eigenvalues, eigenvectors = compute_eigensystem(self.matrix, reduce=reduce)

        return eigenvalues / eigenvalues.sum()

    def compute_eigenvalue_density(self, nbins: int = 500):
        """
        Compute the eigenvalue density of the matrix.

        Parameters
        ----------
        nbins : int
                Number of bins to use in the histogram.

        Returns
        -------
        Plots the histogram of eigenvalues.
        """
        eigenvalues = np.real(self.compute_eigenvalues())

        fig = px.histogram(eigenvalues, nbins=nbins)
        fig.show()

    def compute_eigenvalue_spacing_density(self, nbins: int = 500):
        """
        Compute the density of the spacing between the eigenvalues for comparison with
        the Wigner surmise.

        Parameters
        ----------
        nbins : int
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

        fig = px.histogram(spacing, nbins=nbins)
        fig.show()
