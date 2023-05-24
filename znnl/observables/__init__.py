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
Module for the observables.
"""
from znnl.observables.covariance_entropy import compute_covariance_entropy
from znnl.observables.entropy import compute_entropy
from znnl.observables.fisher_trace import compute_fisher_trace
from znnl.observables.magnitude_entropy import compute_magnitude_density
from znnl.observables.tensornetwork_matrix import compute_tensornetwork_matrix

__all__ = [
    compute_fisher_trace.__name__,
    compute_tensornetwork_matrix.__name__,
    compute_entropy.__name__,
    compute_magnitude_density.__name__,
    compute_covariance_entropy.__name__,
]
