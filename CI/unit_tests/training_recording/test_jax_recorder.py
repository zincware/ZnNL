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

import tempfile
from pathlib import Path

import numpy as onp
import optax
from flax import linen as nn
from numpy.testing import assert_raises
from papyrus.measurements import Accuracy, Loss, NTKTrace

from znnl.models import FlaxModel
from znnl.ntk_computation import JAXNTKComputation
from znnl.training_recording import JaxRecorder


class FlaxTestModule(nn.Module):
    """
    Test model for the Flax tests.
    """

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(5, use_bias=True)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10, use_bias=True)(x)
        return x


class TestJaxRecorder:
    """
    Unit test suite for the JaxRecorder.

    Tests for parent recorder class are implemented in the papyrus package.
    """

    @classmethod
    def setup_class(cls):
        """
        Prepare the test suite.
        """
        dummy_input = onp.random.uniform(size=(5, 3))
        dummy_target = onp.random.uniform(size=(5, 10))
        cls.dummy_data_set = {"inputs": dummy_input, "targets": dummy_target}

        cls.measurements = [
            Loss(apply_fn=lambda x, y: onp.sum((x - y) ** 2)),
            Accuracy(),
            NTKTrace(),
        ]
        cls.neural_state = {
            "accuracy": [onp.random.uniform(size=(1,))],
            "predictions": [onp.random.uniform(size=(5, 10))],
            "targets": [onp.random.uniform(size=(5, 10))],
            "ntk": [onp.random.uniform(size=(5, 5))],
        }
        cls.model = FlaxModel(
            flax_module=FlaxTestModule(),
            optimizer=optax.adam(learning_rate=0.001),
            input_shape=(2, 3),
            seed=17,
        )
        cls.ntk_computation = JAXNTKComputation(
            apply_fn=cls.model.ntk_apply_fn,
            trace_axes=(-1,),
        )

    def test_check_keys(self):
        """
        Test the check_keys method.
        """
        recorder = JaxRecorder(
            name="test",
            storage_path=tempfile.mkdtemp(),
            measurements=self.measurements,
        )

        recorder.neural_state = self.neural_state.copy()

        # Test correct keys
        recorder._check_keys()

        # Test missing keys
        del recorder.neural_state["accuracy"]
        assert_raises(KeyError, recorder._check_keys)

        # Test additional keys (should not raise an error)
        recorder.neural_state["accuracy"] = [onp.random.uniform(size=(1,))]
        recorder.neural_state["additional_key"] = [onp.random.uniform(size=(5,))]
        recorder._check_keys()

    def test_instantiate_recorder(self):
        """
        Test the instantiate_recorder method.
        """
        recorder = JaxRecorder(
            name="test",
            storage_path=tempfile.mkdtemp(),
            measurements=self.measurements,
        )
        recorder.instantiate_recorder(
            data_set=self.dummy_data_set,
            model=self.model,
            ntk_computation=self.ntk_computation,
        )
        assert recorder.neural_state == {}
        assert recorder._data_set == self.dummy_data_set
        assert recorder._model == self.model
        assert recorder._ntk_computation == self.ntk_computation

        # Test errors for missing data
        recorder = JaxRecorder(
            name="test",
            storage_path=tempfile.mkdtemp(),
            measurements=self.measurements,
        )
        assert_raises(
            AttributeError,
            recorder.instantiate_recorder,
            model=self.model,
            ntk_computation=self.ntk_computation,
        )

        # Test errors for missing model
        recorder = JaxRecorder(
            name="test",
            storage_path=tempfile.mkdtemp(),
            measurements=self.measurements,
        )
        assert_raises(
            AttributeError,
            recorder.instantiate_recorder,
            data_set=self.dummy_data_set,
            ntk_computation=self.ntk_computation,
        )

        # Test errors for missing ntk_computation
        recorder = JaxRecorder(
            name="test",
            storage_path=tempfile.mkdtemp(),
            measurements=self.measurements,
        )
        assert_raises(
            AttributeError,
            recorder.instantiate_recorder,
            data_set=self.dummy_data_set,
            model=self.model,
        )

    def test_record(self):
        """
        Test the record method.
        """
        recorder = JaxRecorder(
            name="test",
            storage_path=tempfile.mkdtemp(),
            measurements=self.measurements,
        )
        recorder.instantiate_recorder(
            data_set=self.dummy_data_set,
            model=self.model,
            ntk_computation=self.ntk_computation,
        )

        recorder.record(
            epoch=0, params=self.model.model_state.params, accuracy=[onp.array([0.5])]
        )

        # Check if the neural state is updated
        assert recorder.neural_state["accuracy"] == [onp.array([0.5])]
        assert onp.shape(recorder.neural_state["predictions"]) == (1, 5, 10)
        assert onp.shape(recorder.neural_state["targets"]) == (1, 5, 10)
        assert onp.shape(recorder.neural_state["ntk"]) == (1, 5, 5)
