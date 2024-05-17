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
import neural_tangents as nt
from jax import random, vmap

from znnl.ntk_computation.jax_ntk import JAXNTKComputation


class JAXNTKClassWise(JAXNTKComputation):
    def __init__(
        self,
        apply_fn: Callable,
        batch_size: int = 10,
        ntk_implementation: nt.NtkImplementation = None,
        trace_axes: tuple = (),
        store_on_device: bool = False,
        flatten: bool = True,
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
        """
        super().__init__(
            apply_fn=apply_fn,
            batch_size=batch_size,
            ntk_implementation=ntk_implementation,
            trace_axes=trace_axes,
            store_on_device=store_on_device,
            flatten=flatten,
        )
