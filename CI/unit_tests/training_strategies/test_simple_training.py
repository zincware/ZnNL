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
Test the RND class.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from typing import Callable, List, Optional, Union

import optax
from jax import random
from neural_tangents import stax
from numpy.testing import assert_raises

from znnl.accuracy_functions import AccuracyFunction
from znnl.loss_functions import MeanPowerLoss
from znnl.models.jax_model import JaxModel
from znnl.models.nt_model import NTModel
from znnl.training_recording import JaxRecorder
from znnl.training_strategies import RecursiveMode, SimpleTraining
from znnl.training_strategies.training_decorator import train_func


class SimpleDecoratorTester(SimpleTraining):
    """
    Class to test the training decorator of the simple training.
    """

    def __init__(
        self,
        model: Union[JaxModel, None],
        loss_fn: Callable,
        accuracy_fn: AccuracyFunction = None,
        seed: int = None,
        recursive_mode: RecursiveMode = None,
        disable_loading_bar: bool = False,
        recorders: List["JaxRecorder"] = None,
    ):
        super().__init__(
            model=model,
            loss_fn=loss_fn,
            accuracy_fn=accuracy_fn,
            seed=seed,
            recursive_mode=recursive_mode,
            disable_loading_bar=disable_loading_bar,
            recorders=recorders,
        )

    @train_func
    def train_model(
        self,
        train_ds: dict,
        test_ds: dict,
        epochs: Optional[Union[int, List[int]]] = None,
        batch_size: Optional[Union[int, List[int]]] = None,
        **kwargs,
    ):
        """
        Define train method to test how the decorator changes the inputs.

        Returns
        -------
        Epochs and batch_size of the simple training strategy.
        """
        return epochs, batch_size


class TestSimpleTraining:
    """
    Unit test suite for the simple training strategy.
    """

    @classmethod
    def setup_class(cls):
        """
        Create data for the tests.
        """
        key1, key2 = random.split(random.PRNGKey(1), 2)
        x = random.normal(key1, (3, 8))
        y = random.normal(key1, (3, 1))
        cls.train_ds = {"inputs": x, "targets": y}
        cls.test_ds = {"inputs": x, "targets": y}

    def test_model_error(self):
        """
        Test for error raise when no model is available.

        The model is an optional input in the training strategy construction.
        The input of a model can be handled by frameworks adding the model during the
        workflow of that framework (an example is RND).
        Testing for a KeyError if no model was added but the training method is
        executed.
        """

        trainer = SimpleTraining(
            model=None,
            loss_fn=MeanPowerLoss(order=2),
            disable_loading_bar=True,
        )

        assert_raises(KeyError, trainer.train_model, self.train_ds, self.test_ds, 1)

    def test_update_training_kwargs(self):
        """
        Test the parameter adaption of the training strategy when executing the
        training.

        When calling the train_model method, parameters get adapted if necessary.
        This methods checks if the adaption is done correctly.
        """
        model = NTModel(
            nt_module=stax.serial(stax.Dense(5), stax.Relu(), stax.Dense(1)),
            optimizer=optax.adam(learning_rate=0.001),
            input_shape=(1, 8),
        )

        simple_tester = SimpleDecoratorTester(
            model=model,
            loss_fn=MeanPowerLoss(order=2),
        )

        # Test if the default batch size and epochs correct.
        epochs, batch_size = simple_tester.train_model(
            train_ds=self.train_ds,
            test_ds=self.train_ds,
        )
        assert epochs == 50
        assert batch_size == len(self.train_ds["inputs"])

        # Test if the epochs and the batch_siza are changed correctly.
        epochs, batch_size = simple_tester.train_model(
            train_ds=self.train_ds, test_ds=self.train_ds, epochs=100, batch_size=1
        )
        assert epochs == 100
        assert batch_size == 1

        # Test if the batch size is adapted to the size of available data.
        epochs, batch_size = simple_tester.train_model(
            train_ds=self.train_ds, test_ds=self.train_ds, batch_size=100
        )
        assert batch_size == len(self.train_ds["inputs"])
