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
from jax import random, vmap

from znnl.ntk_computation.jax_ntk import JAXNTKComputation


class JAXNTKClassWise(JAXNTKComputation):
    """
    Class for computing the empirical Neural Tangent Kernel (NTK) using the
    neural-tangents library (implemented in JAX) with class-wise subsampling.

    This class is a subclass of JAXNTKComputation and adds the functionality of
    subsampling the data according to the classes before computing the NTK.
    In this way, the NTK is computed for each class separately.

    Note
    ----
    This class is only implemented for the computing the NTK of a single dataset.
    This menas that axis 0 and 1 of the NTK matrix correspond to the same dataset.
    More information can be found in the `compute_ntk` method.
    """

    def __init__(
        self,
        apply_fn: Callable,
        batch_size: int = 10,
        ntk_implementation: nt.NtkImplementation = None,
        trace_axes: tuple = (),
        store_on_device: bool = False,
        flatten: bool = True,
        data_keys: Optional[List[str]] = None,
        ntk_size: int = None,
    ):
        """
        Constructor the JAX NTK computation class.

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
        batch_size : int
                Size of batch to use in the NTk calculation.
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
        ntk_size : int (default = None)
                Upper limit for the number of samples used for the NTK computation.
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

        self._sample_indices = None
        self.ntk_size = ntk_size

    def _get_sample_indices(self, dataset: dict) -> List[np.ndarray]:
        """
        Group the data by class and return the indices of the samples to use for the
        NTK computation.

        Parameters
        ----------
        dataset : dict
                The dataset containing the inputs and targets.

        Returns
        -------
        sample_indices : dict
                A dictionary containing the indices of the samples for each class, with
                the class label as the key.
        """
        targets = dataset[self.data_keys[1]]

        if len(targets.shape) > 1:
            # If one-hot encoding is used, convert it to class labels
            if targets.shape[1] > 1:
                targets = np.argmax(targets, axis=1)
            # If the targets are already class labels, squeeze the array
            elif targets.shape[1] == 1:
                targets = np.squeeze(targets, axis=1)

        unique_classes = np.unique(targets)
        _indices = np.arange(targets.shape[0])
        sample_indices = {}

        for class_label in unique_classes:
            # Create mask for samples of the current class
            mask = targets == class_label
            indices = np.compress(mask, _indices, axis=0)
            if self.ntk_size is not None:
                indices = indices[: self.ntk_size]
            sample_indices[int(class_label)] = indices

        return sample_indices

    def _subsample_data(self, x: np.ndarray, sample_indices: dict) -> np.ndarray:
        """
        Subsample the data based on indices.

        Parameters
        ----------
        x : np.ndarray
            The input data.
        sample_indices : dict
            The indices of the samples to use for the NTK computation.

        Returns
        -------
        np.ndarray
            The subsampled data.
        """
        return jt.map(lambda indices: np.take(x, indices, axis=0), sample_indices)

    def _compute_ntk(self, params: dict, x_i: np.ndarray) -> np.ndarray:
        """
        Compute the NTK for the neural network.

        Parameters
        ----------
        params : dict
            The parameters of the neural network.
        x_i : np.ndarray
            The input to the neural network.

        Returns
        -------
        np.ndarray
            The NTK matrix.
        """
        ntk = self.empirical_ntk(x_i, None, params)
        ntk = self._check_shape(ntk)
        return ntk

    def compute_ntk(self, params: dict, dataset: dict) -> List[np.ndarray]:
        """
        Compute the Neural Tangent Kernel (NTK) for the neural network.

        Note
        ----
        This method only accepts a single dataset for the NTK computation. This means
        both axes of the NTK matrix correspond to the same dataset.

        Parameters
        ----------
        params : dict
                The parameters of the neural network.
        dataset_i : dict
                The input dataset for the NTK computation.

        Returns
        -------
        List[np.ndarray]
                The NTK matrix.
        """

        self._sample_indices = self._get_sample_indices(dataset)

        x_i = self._subsample_data(dataset[self.data_keys[0]], self._sample_indices)

        ntks = jt.map(lambda x_i: self._compute_ntk(params, x_i), x_i)

        ntks = list(ntks.values())

        # Get the maximum key in the sample indices i
        max_key = max(self._sample_indices.keys())

        # Fill in the missing classes with empty NTKs
        for i in range(max_key):
            if i not in self._sample_indices.keys():
                ntks.insert(i, np.zeros((0, 0)))

        return ntks
