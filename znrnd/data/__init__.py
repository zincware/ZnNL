"""
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincware Project.

Description:
"""
from znrnd.data.cifar10 import CIFAR10Generator
from znrnd.data.confined_particles import ConfinedParticles
from znrnd.data.data_generator import DataGenerator
from znrnd.data.mnist import MNISTGenerator
from znrnd.data.points_on_a_circle import PointsOnCircle
from znrnd.data.points_on_a_lattice import PointsOnLattice


__all__ = [
    ConfinedParticles.__name__,
    DataGenerator.__name__,
    PointsOnLattice.__name__,
    PointsOnCircle.__name__,
    MNISTGenerator.__name__,
    CIFAR10Generator.__name__,
]
