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

from typing import Callable, List, Optional

import jax.numpy as np
import jax.tree as jt
import neural_tangents as nt
from jax import random

from znnl.ntk_computation.jax_ntk import JAXNTKComputation


class JAXNTKSubsampling(JAXNTKComputation):
    """
    Class for computing the empirical Neural Tangent Kernel (NTK) using the
    neural-tangents library (implemented in JAX) with subsampling.

    This class is a subclass of JAXNTKComputation and adds the functionality of
    subsampling the data before computing the NTK.
    Subsampling is useful when the data is too large to compute the NTK on the
    entire dataset.
    Subsampling is done by splitting the data randomly into batches of size
    `ntk_size` and computing the NTK on each part separately.
    This is equivalent to computing block-diagonal elements of the NTK of size
    `ntk_size`.
    The `compute_ntk` method of this class will return a list of len(data) // ntk_size
    NTK matrices.
    """

    def __init__(
        self,
        apply_fn: Callable,
        ntk_size: int,
        seed: int = 0,
        batch_size: int = 10,
        ntk_implementation: nt.NtkImplementation = None,
        trace_axes: tuple = (),
        store_on_device: bool = False,
        flatten: bool = True,
        data_keys: Optional[List[str]] = None,
    ):
        """
        Constructor the JAX NTK computation class with subsampling.

        Parameters
        ----------
        apply_fn : Callable
                The function that applies the neural network to an input.
                This function should be implemented using JAX. It should take in a
                dictionary of parameters (and possibly other arguments) and return the
                output of the neural network.
                For models taking in `batch_stats` the apply function should look like::
                    def apply_fn(params, x):
                        return model.apply(
                            params, x, train=False, mutable=['batch_stats']
                        )[0]
        n_parts : int
                Number of sub-samples to use for the NTK computation.
        ntk_size : int
                Size of the NTK sub-samples.
        batch_size : int
                Size of batch to use in the NTk calculation.
                Note that this has to fit with the set `ntk_size`.
        ntk_implementation : Union[None, NtkImplementation] (default = None)
                Implementation of the NTK computation.
                The implementation depends on the trace_axes and the model
                architecture. The default does automatically take into account the
                trace_axes. For trace_axes=() the default is NTK_VECTOR_PRODUCTS,
                for all other cases including trace_axes=(-1,) the default is
                JACOBIAN_CONTRACTION. For more specific use cases, the user can
                set the implementation manually.
                Information about the implementation and specific requirements can be
                found in the neural_tangents documentation.
        trace_axes : Union[int, Sequence[int]]
                Tracing over axes of the NTK.
                The default value is trace_axes(-1,), which reduces the NTK to a tensor
                of rank 2.
                For a full NTK set trace_axes=().
        store_on_device : bool, default True
                Whether to store the NTK on the device or not.
                This should be set False for large NTKs that do not fit in GPU memory.
        flatten : bool, default True
                If True, the NTK shape is checked and flattened into a 2D matrix, if
                required.
        data_keys : List[str], default ["inputs", "targets"]
                The keys used to define inputs and targets in the dataset.
                These keys are used to extract values from the dataset dictionary in
                the `compute_ntk` method.
                Note that the first key has to refer the input data and the second key
                to the targets / labels of the dataset.
        """
        super().__init__(
            apply_fn=apply_fn,
            batch_size=batch_size,
            ntk_implementation=ntk_implementation,
            trace_axes=trace_axes,
            store_on_device=store_on_device,
            flatten=flatten,
            data_keys=data_keys,
        )
        self.ntk_size = ntk_size
        self.key = random.PRNGKey(seed)

        self._sample_indices: List[np.ndarray] = []
        self.n_parts = None

    def _get_sample_indices(self, x: np.ndarray) -> List[np.ndarray]:
        """
        Split the data into `n_parts` parts of size `ntk_size`.

        Parameters
        ----------
        x : np.ndarray
            The input data.

        Returns
        -------
        List[np.ndarray]
            A list of indices for each part of the data. Each index array has
            length `ntk_size`.
        """
        data_len = x.shape[0]
        self.n_parts = data_len // self.ntk_size

        key, self.key = random.split(self.key)

        indices = random.permutation(key, np.arange(data_len))

        return [
            indices[i * self.ntk_size : (i + 1) * self.ntk_size]
            for i in range(self.n_parts)
        ]

    def _subsample_data(self, x: np.ndarray) -> np.ndarray:
        """
        Subsample the data based on self._sample_indices.

        Parameters
        ----------
        x : np.ndarray
            The input data.

        Returns
        -------
        np.ndarray
            The subsampled data.
        """
        return [np.take(x, indices, axis=0) for indices in self._sample_indices]

    def _compute_ntk(
        self, params: dict, x_i: np.ndarray, x_j: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute the NTK for the neural network.

        Parameters
        ----------
        params : dict
            The parameters of the neural network.
        x_i : np.ndarray
            The input to the neural network.
        x_j : np.ndarray
            The input to the neural network.

        Returns
        -------
        np.ndarray
            The NTK matrix.
        """
        ntk = self.empirical_ntk(x_i, x_j, params)
        ntk = self._check_shape(ntk)
        return ntk

    def compute_ntk(
        self, params: dict, dataset_i: dict, dataset_j: Optional[dict] = None
    ) -> List[np.ndarray]:
        """
        Compute the Neural Tangent Kernel (NTK) for the neural network.

        Parameters
        ----------
        params : dict
                The parameters of the neural network.
        dataset_i : dict
                The input dataset for the NTK computation.
        dataset_j : Optional[dict]
                Optional input dataset for the NTK computation.

        Returns
        -------
        List[np.ndarray]
                The NTK matrix.
        """
        x_i = dataset_i[self.data_keys[0]]
        x_j = dataset_j[self.data_keys[0]] if dataset_j is not None else None

        self._sample_indices = self._get_sample_indices(x_i)
        x_i = self._subsample_data(x_i)

        x_j = self._subsample_data(x_j) if x_j is not None else [None] * self.n_parts

        ntks = jt.map(lambda x_i, x_j: self._compute_ntk(params, x_i, x_j), x_i, x_j)

        return ntks
