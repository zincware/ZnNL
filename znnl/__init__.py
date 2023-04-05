"""
main init file for the project.
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
