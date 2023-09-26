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
from dataclasses import dataclass
from typing import Callable

import jax.numpy as np
import numpy as onp
import optax
from flax.training.train_state import TrainState


@dataclass
class PartitionedTraceOptimizer:
    """
    Class implementation of the trace optimizer

    Attributes
    ----------
    scale_factor : float
            Scale factor to apply to the optimizer.
    rescale_interval : int
            Number of epochs to wait before re-scaling the learning rate.
    subset : float
            What percentage of data you want to use in the trace calculation.
    """

    scale_factor: float
    rescale_interval: float = 1
    subset: float = None

    _start_value = None

    @optax.inject_hyperparams
    def optimizer(self, learning_rate):
        return optax.sgd(learning_rate)

    def apply_optimizer(
        self,
        model_state: TrainState,
        data_set: np.ndarray,
        ntk_fn: Callable,
        epoch: int,
    ):
        """
        Apply the optimizer to a model state.

        Parameters
        ----------
        model_state : TrainState
                Current state of the model
        data_set : jnp.ndarray
                Data-set to use in the computation.
        ntk_fn : Callable
                Function to use for the NTK computation
        epoch : int
                Current epoch

        Returns
        -------
        new_state : TrainState
                New state of the model
        """
        eps = 1e-8

        partitions = {}

        number_of_classes = np.unique(data_set["targets"], axis=0)

        for i in range(number_of_classes.shape[0]):
            indices = np.where(data_set["targets"].argmax(-1) == i)[0]

            partitions[i] = np.take(data_set["inputs"], indices, axis=0)

        if self._start_value is None:
            if self.subset is not None:
                init_data_set = {}
                for ds in partitions:
                    subset_size = int(self.subset * partitions[ds].shape[0])
                    init_data_set[ds] = np.take(
                        partitions[ds],
                        onp.random.randint(
                            0, partitions[ds].shape[0] - 1, size=subset_size
                        ),
                        axis=0,
                    )
            else:
                init_data_set = data_set

            start_trace = 0

            for ds in init_data_set:
                ntk = ntk_fn(init_data_set[ds])["empirical"]
                start_trace += np.trace(ntk)

            self._start_value = np.trace(ntk)

        # Check if the update should be performed.
        if epoch % self.rescale_interval == 0:
            # Select a subset of the data
            if self.subset is not None:
                data_set = {}

                for ds in partitions:
                    subset_size = int(self.subset * partitions[ds].shape[0])
                    data_set[ds] = np.take(
                        partitions[ds],
                        onp.random.randint(
                            0, partitions[ds].shape[0] - 1, size=subset_size
                        ),
                        axis=0,
                    )

            # Compute the ntk trace.
            trace = 0.0

            for ds in data_set:
                ntk = ntk_fn(data_set[ds])["empirical"]
                trace += np.trace(ntk)

            # Create the new optimizer.
            new_optimizer = self.optimizer(
                (self.scale_factor * self._start_value) / (trace + eps)
            )

            # Create the new state
            new_state = TrainState.create(
                apply_fn=model_state.apply_fn,
                params=model_state.params,
                tx=new_optimizer,
            )
        else:
            # If no update is needed, return the old state.
            new_state = model_state

        return new_state
