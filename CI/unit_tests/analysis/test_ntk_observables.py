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
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import jax.numpy as np
import numpy as onp
from numpy.testing import assert_array_almost_equal
from znnl.utils.matrix_utils import compute_magnitude_density, normalize_gram_matrix


class TestNTKObservables:
    """
    Test suite for the calculation of NTK observables.

    All NTK observables are calculated independently of each other, however, they stem
    from a decomposition of one tensor.
    """

    @staticmethod
    def calculate_observables_by_decomposition(ntk: np.ndarray):
        """
        Decompose the NTK into the observables.

        ntk = Tr(ntk) * mag_density * cos_ntk * mag_density

        Parameters
        ----------
        ntk : np.ndarray
                Neural Tangent Kernel to be decomposed
        """
        # Calculate the trace
        trace = np.trace(ntk)

        # Extract the trace from the NTK
        traced_ntk = ntk / trace

        # Calculate the magnitude array and density of NTK gradients.
        diag = np.diagonal(traced_ntk)
        mag_arr = np.sqrt(diag)
        mag_density = mag_arr / mag_arr.sum()

        # Calculate the matrix to make the NTK a gram matrix of cosine similarities.
        normalization_vector = 1 / mag_arr
        normalization_matrix = np.tensordot(
            normalization_vector, normalization_vector, axes=0
        )

        # Calculate the cosine similarity NTK.
        ntk_cos = traced_ntk * normalization_matrix

        return trace, mag_density, ntk_cos

    @staticmethod
    def calculate_observables_independently(ntk: np.ndarray):
        """
        Calculate the NTK observables from build-in functions independently.

        A similar evaluation can be found in jax_recording.JaxRecorder.

        Parameters
        ----------
        ntk : np.ndarray
                Neural Tangent Kernel to be decomposed
        """
        trace = np.trace(ntk)
        mag_density = compute_magnitude_density(gram_matrix=ntk)
        ntk_cos = normalize_gram_matrix(ntk)
        return trace, mag_density, ntk_cos

    def test_ntk_observables(self):
        """
        Test if the build-in NTK observables correspond to the decomposed values.

        An artificial NTK is constructed from random vectors and the observables are
        evaluated and compared.
        """
        array = onp.random.random((7, 10))
        ntk = np.einsum("ij, kj -> ik", array, array)

        trace_d, mag_density_d, ntk_cos_d = self.calculate_observables_by_decomposition(
            ntk
        )
        trace, mag_density, ntk_cos = self.calculate_observables_independently(ntk)

        assert_array_almost_equal(trace, trace_d)
        assert_array_almost_equal(mag_density, mag_density_d)
        assert_array_almost_equal(ntk_cos, ntk_cos_d)
