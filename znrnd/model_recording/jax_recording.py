"""
ZnRND: A Zincwarecode package.

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
Module for recording jax training.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class JaxRecorder:
    """
    Class for recording jax training.

    Each property is has an accompanying array which is set to None.
    At instantiation, this can be populated correctly or resized in the event of
    re-training.
    Only loss is set default True.

    Attributes
    ----------
    """
    # Model Loss
    loss: bool = True
    _loss_array: np.ndarray = None

    # Model accuracy
    accuracy: bool = False
    _accuracy_array: np.ndarray = None

    # NTK Matrix
    ntk: bool = False
    _ntk_array: np.ndarray = None

    # Entropy of the model
    entropy: bool = False
    _entropy_array: np.ndarray = None

    # Model eigenvalues
    eigenvalues: bool = False
    _eigenvalues_array: np.ndarray = None

    def instantiate_recorder(self):
        """
        Prepare the recorder for training.

        Returns
        -------

        """
        pass

    def update_recorder(self):
        """
        Update the values stored in the recorder.

        Returns
        -------

        """
        pass

    def dump_records(self):
        """
        Dump recorded properties to hdf5 database.

        Returns
        -------

        """
        pass

    def visualize_recorder(self):
        """
        Display recorded values as web app.

        Returns
        -------

        """
        pass
