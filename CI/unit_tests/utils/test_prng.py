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
Unit tests for the PRNG class.
"""
from itertools import combinations

import jax.random
from numpy.testing import assert_array_equal, assert_raises

from znrnd.utils import PRNGKey


class TestPRNG:
    """Test the PRNG key generation."""

    def test_deterministic_sequence(self):
        """Test that a given seed produces the same sequence."""
        rng_1 = PRNGKey(42)
        array_1 = jax.random.uniform(key=rng_1.key, shape=(100,))

        rng_2 = PRNGKey(42)
        array_2 = jax.random.uniform(key=rng_2.key, shape=(100,))

        assert_array_equal(array_1, array_2)

    def test_random_sequence(self):
        """Test that random sequences can still be obtained."""
        rng = PRNGKey(42)
        random_list = []
        for i in range(10):
            random_list.append(jax.random.uniform(key=rng.key, shape=(10,)))
        # Assert that no equal sequences exist
        for array_1, array_2 in combinations(random_list, 2):
            assert_raises(AssertionError, assert_array_equal(array_1, array_2))

    def test_deterministic_split(self):
        """Test that a given seed also produces the same sequence after splitting."""
        rng_1 = PRNGKey(42)
        random_list_1 = []
        for i in range(10):
            random_list_1.append(jax.random.uniform(key=rng_1.key, shape=(10,)))

        rng_2 = PRNGKey(42)
        random_list_2 = []
        for i in range(10):
            random_list_2.append(jax.random.uniform(key=rng_2.key, shape=(10,)))

        assert_array_equal(random_list_1, random_list_2)
