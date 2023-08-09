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
import logging

import znnl
from znnl import (
    accuracy_functions,
    agents,
    analysis,
    data,
    distance_metrics,
    loss_functions,
    models,
    optimizers,
    point_selection,
    similarity_measures,
    training_recording,
    training_strategies,
    utils,
)
from znnl.utils.machine_properties import print_local_device_properties
from znnl.visualization import TSNEVisualizer

logging.getLogger(znnl.__name__).addHandler(logging.NullHandler())


__all__ = [
    distance_metrics.__name__,
    loss_functions.__name__,
    accuracy_functions.__name__,
    optimizers.__name__,
    models.__name__,
    point_selection.__name__,
    similarity_measures.__name__,
    data.__name__,
    TSNEVisualizer.__name__,
    agents.__name__,
    analysis.__name__,
    utils.__name__,
    training_recording.__name__,
    training_strategies.__name__,
]

print_local_device_properties()  # Report local hardware information.
