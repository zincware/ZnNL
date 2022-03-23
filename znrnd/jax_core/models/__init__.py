"""
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincware Project.

Description:
"""
from znrnd.jax_core.models.model import Model
from znrnd.jax_core.models.flax_model import FlaxModel

__all__ = [Model.__name__, FlaxModel.__name__]
