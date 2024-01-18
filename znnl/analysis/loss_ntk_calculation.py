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

import neural_tangents as nt
from typing import Callable


class loss_ntk_calculation:
    def __init__(
        self,
        metric_fn: Callable,
        dataset: dict,
        ntk_batch_size: int = 10,
        store_on_device: bool = True,
    ):
        """Constructor for the loss ntk calculation class."""
        self.metric_fn = metric_fn
        self.dataset = dataset
        self.ntk_batch_size = ntk_batch_size
        self.store_on_device = store_on_device

    def _function_for_loss_ntk_helper(params, dataset, metric_fn, apply_fn):
        return metric_fn(apply_fn(params, dataset["inputs"]), dataset["targets"])

    def calculate_loss_ntk(
        self,
        model,
    ):
        _function_for_loss_ntk = lambda x, y: self._function_for_loss_ntk_helper(
            x, y, metric_fn, model._ntk_apply_fn
        )

        # Prepare NTK calculation
        empirical_ntk = nt.batch(
            nt.empirical_ntk_fn(f=_function_for_loss_ntk, trace_axes=trace_axes),
            batch_size=ntk_batch_size,
            store_on_device=store_on_device,
        )
        empirical_ntk_jit = jax.jit(empirical_ntk)
