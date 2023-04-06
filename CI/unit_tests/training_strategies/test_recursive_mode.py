"""
ZnRND: A Zincwarecode package.

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
Unit tests for the recursive mode class.
"""
import optax
from neural_tangents import stax

from znrnd.loss_functions import MeanPowerLoss
from znrnd.models import NTModel
from znrnd.training_strategies import RecursiveMode, SimpleTraining


class TestRecursiveMode:
    """
    Unit test suite of the recursive mode.
    """

    @classmethod
    def setup_class(cls):
        """
        Create a model for the tests.
        """
        network = stax.serial(
            stax.Flatten(), stax.Dense(10), stax.Relu(), stax.Dense(10)
        )
        model = NTModel(
            nt_module=network,
            optimizer=optax.adam(learning_rate=0.01),
            input_shape=(1, 8),
            batch_size=10,
        )
        cls.trainer = SimpleTraining(
            model=model,
            loss_fn=MeanPowerLoss(order=2),
        )

    def test_instantiation(self):
        """
        Test the instantiation of the recursive mode.
        """
        # Default
        mode = RecursiveMode()
        mode.instantiate_recursive_mode(self.trainer)
        assert "init_model" == mode.perturb_fn.__name__
        assert "_update_fn_rnd" == mode.update_fn.__name__

        # Manual settings
        mode = RecursiveMode(update_type="rnd")
        mode.instantiate_recursive_mode(self.trainer)
        assert "_update_fn_rnd" == mode.update_fn.__name__

        mode = RecursiveMode(update_type="threshold")
        mode.instantiate_recursive_mode(self.trainer)
        assert "_update_fn_threshold" == mode.update_fn.__name__
