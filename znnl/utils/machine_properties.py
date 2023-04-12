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
import jax
from rich import print


def print_local_device_properties():
    """
    A helper function to print local device properties.

    This function is called at instantiation to report to the user what devices are used
    by default on their local computer.

    Returns
    -------
    Prints a list of devices being used.
    """
    backend = jax.default_backend()  # get used backend
    device_list = jax.devices()  # collect a list of available devices

    print(f"Using backend: {backend}")
    print("Available hardware:")
    for item in device_list:
        print(item)
