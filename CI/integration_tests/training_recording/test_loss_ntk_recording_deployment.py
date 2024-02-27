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

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
from numpy.testing import assert_array_almost_equal
import optax
from neural_tangents import stax

from znnl.loss_functions import LPNormLoss
from znnl.models import NTModel
from znnl.training_recording import JaxRecorder
from znnl.training_strategies import SimpleTraining


class TestLossNTKRecorderDeployment:
    """
    Test suite for the loss and NTK recorder.
    """

    @classmethod
    def setup_class(cls):
        """
        Create a model and data for the tests.
        """

        network = stax.serial(
            stax.Dense(10), stax.Relu(), stax.Dense(10), stax.Relu(), stax.Dense(1)
        )
        cls.model = NTModel(
            nt_module=network, input_shape=(5,), optimizer=optax.adam(1e-3)
        )

        cls.data_set = {
            "inputs": np.random.rand(10, 5),
            "targets": np.random.randint(0, 2, (10, 1)),
        }

        cls.ntk_recorder = JaxRecorder(
            name="ntk_recorder",
            ntk=True,
            update_rate=1,
        )
        cls.loss_ntk_recorder = JaxRecorder(
            name="loss_ntk_recorder",
            ntk=True,
            use_loss_ntk=True,
            update_rate=1,
        )

        cls.ntk_recorder.instantiate_recorder(data_set=cls.data_set)
        cls.loss_ntk_recorder.instantiate_recorder(data_set=cls.data_set)

        cls.trainer = SimpleTraining(
            model=cls.model,
            loss_fn=LPNormLoss(order=2),
            recorders=[cls.ntk_recorder, cls.loss_ntk_recorder],
        )

    def test_loss_ntk_deployment(self):
        """
        Test the deployment of the loss_NTK recorder.
        """

        # train the model
        training_metrics = self.trainer.train_model(
            train_ds=self.data_set,
            test_ds=self.data_set,
            epochs=10,
            batch_size=2,
        )

        # gather the recording
        ntk_recording = self.ntk_recorder.gather_recording()
        loss_ntk_recording = self.loss_ntk_recorder.gather_recording()

        # For LPNormLoss of order 2 and a 1D output Network, the NTK and the loss NTK
        # should be the same up to a factor of +1 or -1.
        assert_array_almost_equal(
            np.abs(ntk_recording.ntk), np.abs(loss_ntk_recording.ntk), decimal=5
        )
