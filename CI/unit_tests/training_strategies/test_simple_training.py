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
Test the RND class.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from jax import random
from numpy.testing import assert_raises

from znrnd.loss_functions import MeanPowerLoss
from znrnd.training_strategies import SimpleTraining


class TestRNDTraining:
    """
    Class to test the initialization of the simple training strategy
    """

    def test_model_error(self):
        """
        The model is optional input in the training strategy, as the model input can be
        handled by frameworks add the model in the workflow.
        """

        # Create some test data
        key1, key2 = random.split(random.PRNGKey(1), 2)
        x = random.normal(key1, (3, 8))
        y = random.normal(key1, (3, 1))
        train_ds = {"inputs": x, "targets": y}
        test_ds = train_ds

        trainer = SimpleTraining(
            model=None,
            loss_fn=MeanPowerLoss(order=2),
            disable_loading_bar=True,
        )

        assert_raises(KeyError, trainer.train_model, train_ds, test_ds, 1)
