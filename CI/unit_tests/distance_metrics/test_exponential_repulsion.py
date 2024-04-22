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

from znnl.distance_metrics.exponential_repulsion import ExponentialRepulsion
from znnl.distance_metrics.order_n_difference import OrderNDifference


class TestExponentialRepulsion:
    """
    Class to test the ExponentialRepulsion.

    The ExponentialRepulsion is not a distance metric, as a point to itself
    will not be zero.
    It defines a potential field, which is repulsive for points that are close
    to each other using the exponential function.
    """

    def test_particular_values(self):
        """
        Test the exponential repulsion for particular values.

        The test is done for the following values:

        1.  Test if similar points return alpha.
        2.  Test if two points on the diagonal of an N-dimensional hypercube of length 1
            return alpha*exp(-1).
        3.  Test if two points on the diagonal of an N-dimensional hypercube
            of negative length -1 return alpha/beta*exp(-1).
        4. Test if two points on the diagonal of an N-dimensional hypercube
            of length 2 return alpha*exp(-1) return alpha when rescaling with
            temperature=2.

        """
        alpha = 17.0
        temperature = 1.0
        potential = ExponentialRepulsion(alpha=alpha, temperature=temperature)

        # 1.
        point_1 = [[1.0, 1.0, 1.0]]
        point_2 = [[1.0, 1.0, 1.0]]
        result = potential(np.array(point_1), np.array(point_2))
        print(result)
        assert result == alpha

        # 2.
        point_1 = [[0.0, 0.0, 0.0, 0.0]]
        point_2 = [[1.0, 1.0, 1.0, 1.0]]
        result = potential(np.array(point_1), np.array(point_2))
        assert result == alpha * np.exp(-1.0)

        # 3.
        point_1 = [[0.0, 0.0, 0.0, 0.0]]
        point_2 = [[-1.0, -1.0, -1.0, -1.0]]
        result = potential(np.array(point_1), np.array(point_2))
        assert result == alpha * np.exp(-1.0)

        # 4.
        alpha = 17.0
        temperature = 2.0
        potential = ExponentialRepulsion(alpha=alpha, temperature=temperature)
        point_1 = [[0.0, 0.0, 0.0, 0.0]]
        point_2 = [[2.0, 2.0, 2.0, 2.0]]
        result = potential(np.array(point_1), np.array(point_2))
        assert result == alpha * np.exp(-1.0)

    def test_default_parameters(self):
        """
        Test whether the repulsion values, when using the default values, is
        of similar size as the mean squared error.

        This test is done, as the exponential repulsion is used in the contrastive
        loss function as default for repulsion. The mean squared error is used
        as default for attraction.
        The returned values should be of similar magnitude to make the loss function
        not be dominated by one of the two terms.
        """

        se_metric = OrderNDifference(order=2)
        repulsion_metric = ExponentialRepulsion()

        point_1 = [[0.1, 0.1, 0.1, 0.1]]
        point_2 = [[0.0, 0.0, 0.0, 0.0]]

        result_se = se_metric(np.array(point_1), np.array(point_2))
        result_repulsion = repulsion_metric(np.array(point_1), np.array(point_1))

        np.equal(result_se, result_repulsion)

    def test_scaling(self):
        """
        Test whether the repulsion values are invariant to dimensionality of the input.
        """
        repulsion_metric = ExponentialRepulsion()

        point_1 = np.zeros((1, 10))
        point_2 = np.ones((1, 10))
        result_10d = repulsion_metric(point_1, point_2)

        point_1 = np.zeros((1, 100))
        point_2 = np.ones((1, 100))
        result_100d = repulsion_metric(point_1, point_2)

        np.equal(result_10d, result_100d)
