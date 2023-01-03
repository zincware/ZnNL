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

    Notes
    -----
    Currently the options are hard-coded. In the future, we will work towards allowing
    for arbitrary computations to be added, for example, two losses.
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

    # Class helpers
    _selected_properties: list = None

    def instantiate_recorder(
            self, data_length: int, data_shape: tuple = None, overwrite: bool = False
    ):
        """
        Prepare the recorder for training.

        Parameters
        ----------
        data_length : int
                Length of time series to store currently.
        data_shape : tuple (default=None)
                If the NTK is being recorded, the shape of the data should be provided.
        overwrite : bool (default=False)
                If true and there is data already in the array, this will be removed and
                a new array created.

        Returns
        -------
        Populates the array attributes of the dataclass.
        """
        # populate the class attribute
        self._selected_properties = [
            value for value in list(vars(self)) if value[0] != "_" and vars(self)[value] is True
        ]

        all_attributes = self.__dict__
        for item in self._selected_properties:
            if item is "ntk":
                all_attributes[f"_{item}_array"] = np.zeros(
                    (data_length, data_shape[0], data_shape[0])
                )
            else:
                all_attributes[f"_{item}_array"] = np.zeros((data_length,))

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
        raise NotImplementedError("Not yet available in ZnRND.")

    def visualize_recorder(self):
        """
        Display recorded values as web app.

        Returns
        -------

        """
        raise NotImplementedError("Not yet available in ZnRND.")
