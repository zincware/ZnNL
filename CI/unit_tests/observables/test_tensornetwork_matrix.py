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
This module tests the implementation of the tensornetwork matrix computation module.
"""

import numpy as np
from numpy.testing import assert_almost_equal

from znnl.observables.tensornetwork_matrix import compute_tensornetwork_matrix


class TestTensornetworkMatrix:
    """
    Class for testing the implementation of the tensornetwork matrix calculation
    """

    def test_tensornetwork_matrix_computation(self):
        """
        Function tests if the fisher trace computation works correctly for an
        example which was calculated by hand before.

        Returns
        -------
        Asserts the calculated fisher trace for the manually defined inputs
        is what it should be.
        """

        ntk = np.array([[1, 2, 3, 4], [1, 1, 1, 1], [0, 1, 2, 4], [1, 0, 0, 8]])
        targets = np.array([2, 2, 1, 1])

        matrix = compute_tensornetwork_matrix(ntk=ntk, targets=targets)
        correctmatrix = [[3.5, 0.5, 2.25, 1.25]]

        assert_almost_equal(matrix, correctmatrix)
