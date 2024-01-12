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


class ntk_calculation:
    def function_for_loss_ntk(params)
    def calculate_loss_ntk(
        self,
        model,
        metric_fn,
        dataset,
        ntk_batch_size: int = 10,
        store_on_device: bool = True,
    ):
        # Prepare NTK calculation
        empirical_ntk = nt.batch(
            nt.empirical_ntk_fn(f=model._ntk_apply_fn, trace_axes=trace_axes),
            batch_size=ntk_batch_size,
            store_on_device=store_on_device,
        )
        empirical_ntk_jit = jax.jit(empirical_ntk)
