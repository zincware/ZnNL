"""
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincware Project.

Description:
"""
from znrnd.training_strategies.loss_aware_reservoir import LossAwareReservoir
from znrnd.training_strategies.partitioned_training import PartitionedTraining
from znrnd.training_strategies.recursive_mode import RecursiveMode
from znrnd.training_strategies.simple_training import SimpleTraining

__all__ = [
    SimpleTraining.__name__,
    LossAwareReservoir.__name__,
    PartitionedTraining.__name__,
    RecursiveMode.__name__,
]
