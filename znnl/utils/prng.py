"""
ZnRND: A zincwarecode package.

License
-------
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the zincwarecode Project.

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
Wrapper class for the jax random PRNG key for conveniently generating subkeys.
"""
import jax.random
import numpy as onp


class PRNGKey:
    """Wrapper class for convenient subkey generation.

    Use rng.call() as key in jax.numpy.random methods in order to get random
    sequences every time.
    """

    def __init__(self, seed: int = None):
        """Initialize a new random number generator.

        Parameter
        ---------
        seed : int, default None
            seed to use, use `onp.random.randint` if None.
        """
        if seed is None:
            seed = onp.random.randint(onp.iinfo(onp.int32).max)
        self._rng = jax.random.PRNGKey(seed)

    def __call__(self):
        """Generate and return a new subkey.

        For a given seed this will always produce the same deterministic sequence of
        subkeys, while still providing random sequences everytime the instance is used.
        """
        _, self._rng = jax.random.split(self._rng)
        return self._rng
