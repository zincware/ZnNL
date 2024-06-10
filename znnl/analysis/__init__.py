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

from znnl.analysis.eigensystem import EigenSpaceAnalysis
from znnl.analysis.entropy import EntropyAnalysis
from znnl.analysis.jax_ntk import JAXNTKComputation
from znnl.analysis.jax_ntk_classwise import JAXNTKClassWise
from znnl.analysis.jax_ntk_combinations import JAXNTKCombinations
from znnl.analysis.jax_ntk_subsampling import JAXNTKSubsampling
from znnl.analysis.loss_fn_derivative import LossDerivative

__all__ = [
    EntropyAnalysis.__name__,
    EigenSpaceAnalysis.__name__,
    LossDerivative.__name__,
    JAXNTKComputation.__name__,
    JAXNTKClassWise.__name__,
    JAXNTKSubsampling.__name__,
    JAXNTKCombinations.__name__,
]
