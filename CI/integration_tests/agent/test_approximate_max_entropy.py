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

import optax
from neural_tangents import stax

import znnl


class TestApproximateMaxEntropy:
    """
    Test suite for the approximate maximum entropy agent.
    """

    @classmethod
    def setup_class(cls):
        """
        Prepare the test suite.
        """
        # Model preparation
        model = stax.serial(stax.Dense(12), stax.Relu(), stax.Dense(12))

        target_network = znnl.models.NTModel(
            nt_module=model,
            optimizer=optax.sgd(0.001),
            input_shape=(2,),
        )

        # Data generator preparation
        data_generator = znnl.data.PointsOnLattice()
        data_generator.build_pool(10, 10)

        cls.agent = znnl.agents.ApproximateMaximumEntropy(
            samples=10, target_network=target_network, data_generator=data_generator
        )
