"""
main init file for the project.
"""
import logging

import znrnd
from znrnd.core import (
    agents,
    analysis,
    data,
    distance_metrics,
    loss_functions,
    models,
    point_selection,
    similarity_measures,
    utils,
)
from znrnd.core.visualization import TSNEVisualizer

logging.getLogger(znrnd.__name__).addHandler(logging.NullHandler())


__all__ = [
    distance_metrics.__name__,
    loss_functions.__name__,
    models.__name__,
    point_selection.__name__,
    similarity_measures.__name__,
    data.__name__,
    TSNEVisualizer.__name__,
    agents.__name__,
    analysis.__name__,
    utils.__name__
]
