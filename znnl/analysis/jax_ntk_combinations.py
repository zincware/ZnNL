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

from itertools import combinations
from typing import Callable, List, Optional

import jax.numpy as np
import neural_tangents as nt
from papyrus.utils.matrix_utils import flatten_rank_4_tensor, unflatten_rank_4_tensor

from znnl.analysis.jax_ntk import JAXNTKComputation


class JAXNTKCombinations(JAXNTKComputation):
    """
    Class for computing the empirical Neural Tangent Kernel (NTK) using the
    neural-tangents library (implemented in JAX) of all possible class combinations.

    This can be understood in the following way:
    For a dataset of n labels, and a given selection of labels (e.g. 0, 2), the NTK
    will be computed for all possible combinations made from the selected labels.
    This means that the NTKs for the combinations
    (0, 0), (0, 2), (2, 0), (2, 2) and (0+2, 0+2) will be computed.

    Note
    ----
    This class is only implemented for the computing the NTK of a single dataset.
    This menas that axis 0 and 1 of the NTK matrix correspond to the same dataset.
    More information can be found in the `compute_ntk` method.
    """

    def __init__(
        self,
        apply_fn: Callable,
        class_labels: List[int],
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
        class_labels : List[int]
                List of class labels to use for
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

        self.class_labels = class_labels

        # Compute all possible combinations of the class labels
        self.label_combinations = self._compute_combinations()

    def _reduce_data_to_labels(self, dataset: dict) -> dict:
        """
        Reduce the dataset to only contain the selected class labels.

        Parameters
        ----------
        dataset : dict
                The dataset containing the inputs and targets.

        Returns
        -------
        dict
                The dataset containing only the selected class labels.
        """
        targets = dataset[self.data_keys[1]]

        if len(targets.shape) > 1:
            # If one-hot encoding is used, convert it to class labels
            if targets.shape[1] > 1:
                targets = np.argmax(targets, axis=1)
            # If the targets are already class labels, squeeze the array
            elif targets.shape[1] == 1:
                targets = np.squeeze(targets, axis=1)

        mask = np.isin(targets, np.array(self.class_labels))
        dataset_reduced = {}
        for key, value in dataset.items():
            dataset_reduced[key] = np.compress(mask, value, axis=0)

        return dataset_reduced

    def _get_label_indices(self, dataset: dict) -> List[np.ndarray]:
        """
        Group the data by label and return the indices of the samples to use for the
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

        _indices = np.arange(targets.shape[0])
        sample_indices = {}

        for class_label in self.class_labels:
            # Create mask for samples of the current class
            mask = targets == class_label
            indices = np.compress(mask, _indices, axis=0)
            sample_indices[int(class_label)] = indices

        return sample_indices

    def _compute_combinations(self) -> List[np.ndarray]:
        """
        Compute all possible combinations of the class labels.

        The combinations are computed for all possible pairs of class labels contained
        in the `

        Parameters
        ----------
        sample_indices : dict
                The indices of the samples to use for the NTK computation.

        Returns
        -------
        List[np.ndarray]
                The NTK matrix.
        """
        label_combinations = []
        # Compute all possible combinations of the class labels
        for i in range(1, len(self.class_labels) + 1):
            label_combinations.extend(combinations(self.class_labels, i))

        return label_combinations

    def _take_sub_ntk(
        self, ntk: np.ndarray, label_indices: dict, combination: tuple
    ) -> np.ndarray:
        """
        Take a submatrix of the NTK matrix using np.ix_.

        Parameters
        ----------
        ntk : np.ndarray
                The NTK matrix.
        label_indices : dict
                A dictionary containing the indices of the samples for each class, with
                the class label as the key.
        combinations : tuple
                The combination of class labels to use for the submatrix.

        Returns
        -------
        np.ndarray
                The submatrix of the NTK matrix.
        """
        indices = [label_indices[label] for label in combination]
        indices = np.concatenate(indices)

        # Check if flattening was performed
        if self._is_flattened:
            ntk = unflatten_rank_4_tensor(ntk, self._ntk_shape)

        ntk_sub = ntk[np.ix_(indices, indices)]

        if self.flatten:
            ntk_sub, _ = flatten_rank_4_tensor(ntk_sub)

        return ntk_sub

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

    def compute_ntk(self, params: dict, dataset_i: dict) -> List[np.ndarray]:
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
                List of NTK matrices for all possible class combinations.
                What class combinations each NTK corresponds to can be found in the
                `label_combinations` attribute.
        """

        # Reduce the dataset to the selected class labels
        dataset_reduced = self._reduce_data_to_labels(dataset_i)

        # Compute the NTK for the reduced dataset
        ntk = self._compute_ntk(params, dataset_reduced[self.data_keys[0]])

        # Get the label indices referencing to the reduced dataset
        label_indices = self._get_label_indices(dataset_reduced)

        # Create copies of the NTK for all possible class combinations
        ntks = []
        for combination in self.label_combinations:
            sub_ntk = self._take_sub_ntk(ntk, label_indices, combination)
            ntks.append(sub_ntk)

        return ntks
