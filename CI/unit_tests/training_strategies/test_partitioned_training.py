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
Unit tests for the pertitioned training class.
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from typing import Callable, List, Tuple, Union

import jax.numpy as np
import optax
from jax import random
from neural_tangents import stax
from numpy.testing import assert_array_equal, assert_raises

from znrnd.accuracy_functions import AccuracyFunction
from znrnd.loss_functions import MeanPowerLoss
from znrnd.models import NTModel
from znrnd.models.jax_model import JaxModel
from znrnd.training_recording import JaxRecorder
from znrnd.training_strategies import PartitionedTraining, RecursiveMode
from znrnd.training_strategies.training_decorator import train_func


class PartitionedDecoratorTester(PartitionedTraining):
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
        epochs: list = None,
        batch_size: list = None,
        train_ds_selection: list = None,
        **kwargs,
    ):
        """
        Define train method to test how the decorator changes the inputs.

        Returns
        -------
        Epochs, batch_size and train_ds_selection of the partitioned training strategy.
        """
        return epochs, batch_size, train_ds_selection


class TestPartitionedSelection:
    """
    Unit test suite of the partitioned training strategy.
    """

    @classmethod
    def setup_class(cls):
        """
        Create data for the tests.
        """
        key1, key2 = random.split(random.PRNGKey(1), 2)
        x = random.normal(key1, (5, 8))
        y = random.normal(key1, (5, 1))
        cls.train_ds = {"inputs": x, "targets": y}
        cls.test_ds = {"inputs": x, "targets": y}

    def test_parameter_error(self):
        """
        Test error raises for wrong keyword arguments.

        Assert a KeyError when using the partitioned training strategy with an integer
        instead of a list input for epochs.
        Assert a KeyError for differing list lengths for epochs compared to batch_size
        and train_ds_selection.
        """

        model = NTModel(
            nt_module=stax.serial(stax.Dense(5), stax.Relu(), stax.Dense(1)),
            optimizer=optax.adam(learning_rate=0.001),
            input_shape=(1, 8),
        )

        trainer = PartitionedTraining(
            model=model,
            loss_fn=MeanPowerLoss(order=2),
            disable_loading_bar=True,
        )

        # Check Error handling for passing an int for epochs instead of a list
        assert_raises(KeyError, trainer.train_model, self.train_ds, self.test_ds, 50)

        # Check Error handling for not matching lengths
        assert_raises(
            KeyError,
            trainer.train_model,
            self.train_ds,
            self.test_ds,
            [slice(1, 2, None)],
        )

    def test_update_training_kwargs(self):
        """
        est the parameter adaption of the training strategy when executing the
        training.

        When calling the train_model method, parameters get adapted if necessary.
        This methods checks if the adaption is done correctly.
        """
        model = NTModel(
            nt_module=stax.serial(stax.Dense(5), stax.Relu(), stax.Dense(1)),
            optimizer=optax.adam(learning_rate=0.001),
            input_shape=(1, 8),
        )

        par_tester = PartitionedDecoratorTester(
            model=model,
            loss_fn=MeanPowerLoss(order=2),
        )

        # Test if the default batch size and epochs are correct.
        epochs, batch_size, train_ds_selection = par_tester.train_model(
            train_ds=self.train_ds,
            test_ds=self.train_ds,
        )
        assert_array_equal(epochs, [150, 50])
        assert_array_equal(batch_size, [1, 1])
        assert_array_equal(
            train_ds_selection, [slice(-1, None, None), slice(None, None, None)]
        )

        # Test if the batch size and the epochs get chenged correctly.
        epochs, batch_size, train_ds_selection = par_tester.train_model(
            train_ds=self.train_ds,
            test_ds=self.train_ds,
            epochs=[10, 10],
            batch_size=[2, 2],
            train_ds_selection=[slice(None, None, None), slice(None, None, None)],
        )
        assert_array_equal(epochs, [10, 10])
        assert_array_equal(batch_size, [2, 2])
        assert_array_equal(
            train_ds_selection, [slice(None, None, None), slice(None, None, None)]
        )

        # Test if the batch size is adapted to the size of available data.
        epochs, batch_size, train_ds_selection = par_tester.train_model(
            train_ds=self.train_ds,
            test_ds=self.train_ds,
            epochs=[10, 10],
            batch_size=[10, 10],
            train_ds_selection=[slice(None, 3, None), slice(None, 5, None)],
        )
        assert_array_equal(epochs, [10, 10])
        assert_array_equal(batch_size, [3, 5])
        assert_array_equal(
            train_ds_selection, [slice(None, 3, None), slice(None, 5, None)]
        )

        # Test if the adaption also works for arrays of indices.
        epochs, batch_size, train_ds_selection = par_tester.train_model(
            train_ds=self.train_ds,
            test_ds=self.train_ds,
            epochs=[10],
            batch_size=[10],
            train_ds_selection=[np.array([1, 3, 4])],
        )
        assert_array_equal(epochs, [10])
        assert_array_equal(batch_size, [3])
        assert_array_equal(train_ds_selection, [[1, 3, 4]])
