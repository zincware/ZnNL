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
This module tests the implementation of the fisher trace computation module.
"""

import numpy as np

from znnl.observables.fisher_trace import compute_fisher_trace


class TestFisherTrace:
    """
    Class for testing the implementation of the fisher trace calculation
    """

    def test_fisher_trace_computation(self):
        """
        Function tests if the fisher trace computation works correctly for an
        example which was calculated by hand before.

        Returns
        -------
        Asserts the calculated fisher trace for the manually defined inputs
        is what it should be.
        """

        ntk = np.array(
            [
                [[[1, 2, 3], [4, 5, 6], [7, 8, 9]], np.random.rand(3, 3)],
                [np.random.rand(3, 3), [[2, 1, 3], [1, 2, 3], [3, 2, 1]]],
            ]
        )
        loss_derivative = np.array([[5, 4, 3], [2, 1, 0]])

        trace = compute_fisher_trace(loss_derivative=loss_derivative, ntk=ntk)
        assert trace == 638 / 2
