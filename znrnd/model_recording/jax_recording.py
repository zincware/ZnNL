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
from dataclasses import dataclass, make_dataclass

import numpy as np
from rich import print

from znrnd.analysis.eigensystem import EigenSpaceAnalysis
from znrnd.analysis.entropy import EntropyAnalysis
from znrnd.models.jax_model import JaxModel


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
    loss : bool (default=True)
            If true, loss will be recorded.
    accuracy : bool (default=False)
            If true, accuracy will be recorded.
    ntk : bool (default=False)
            If true, the ntk will be recorded. Warning, large overhead.
    entropy : bool (default=False)
            If true, entropy will be recorded. Warning, large overhead.
    eigenvalues : bool (default=False)
            If true, eigenvalues will be recorded. Warning, large overhead.
    update_rate : int (default=1)
            How often the values are updated.

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
    update_rate: int = 1
    _selected_properties: list = None
    _model: JaxModel = None
    _data_set: dict = None
    _compute_ntk: bool = False  # Helps to know so we can compute it once and share.
    _index_count: int = 0  # Helps to avoid problems with non-1 update rates.

    def _read_selected_attributes(self):
        """
        Helper function to read selected attributes.
        """
        # populate the class attribute
        self._selected_properties = [
            value
            for value in list(vars(self))
            if value[0] != "_" and vars(self)[value] is True
        ]

    def instantiate_recorder(
        self, data_length: int, data_set: dict, overwrite: bool = False
    ):
        """
        Prepare the recorder for training.

        Parameters
        ----------
        model : JaxModel
                Model to monitor and record.
        data_length : int
                Length of time series to store currently.
        data_set : dict (default=None)
                Data to record during training.
        overwrite : bool (default=False)
                If true and there is data already in the array, this will be removed and
                a new array created.

        Returns
        -------
        Populates the array attributes of the dataclass.

        Notes
        -----
        * TODO: Add check for previous array and extend
        """
        # Update simple attributes
        self._data_set = data_set

        # Update involved attributes
        data_shape = data_set["inputs"].shape

        # populate the class attribute
        self._read_selected_attributes()

        # Instantiate arrays
        all_attributes = self.__dict__
        for item in self._selected_properties:
            if item == "ntk":
                all_attributes[f"_{item}_array"] = np.zeros(
                    (data_length, data_shape[0], data_shape[0])
                )
            else:
                all_attributes[f"_{item}_array"] = np.zeros((data_length,))

        # Check if we need an NTK computation and update the class accordingly
        test_array = np.array(
            [
                "ntk" in self._selected_properties,
                "entropy" in self._selected_properties,
                "eigenvalues" in self._selected_properties,
            ]
        )
        if test_array.sum() > 0:
            self._compute_ntk = True

    def update_recorder(self, epoch: int, model: JaxModel):
        """
        Update the values stored in the recorder.

        Parameters
        ----------
        epoch : int
                Current epoch of the model.
        model : JaxModel
                Model to use in the update.

        Returns
        -------
        Updates the chosen class attributes depending on the user requirements.
        """
        # Check if we need to record and if so, record
        if epoch % self.update_rate == 0:
            # Update here to expose to other methods.
            self._model = model

            parsed_data = {}

            # Add epoch to the parsed data
            parsed_data["epoch"] = self._index_count
            # Compute representations here to avoid repeated computation.
            predictions = self._model(self._data_set["inputs"])
            parsed_data["predictions"] = predictions

            # Compute ntk here to avoid repeated computation.
            if self._compute_ntk:
                try:
                    ntk = self._model.compute_ntk(
                        self._data_set["inputs"], normalize=False
                    )
                    parsed_data["ntk"] = ntk
                except NotImplementedError:
                    print(
                        "NTK calculation is not yet available for this model. Removing "
                        "it from this recorder."
                    )
                    self.ntk = False
                    self.entropy = False
                    self.eigenvalues = False
                    self._read_selected_attributes()

            for item in self._selected_properties:
                call_fn = getattr(self, f"_update_{item}")  # get the callable function
                call_fn(parsed_data)  # call the function and update the property

            self._index_count += 1  # Update the index count.
        else:
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

    def _update_loss(self, parsed_data: dict):
        """
        Update the loss array.

        Parameters
        ----------
        parsed_data : dict
                Data computed before the update to prevent repeated calculations.
        """
        self._loss_array[parsed_data["epoch"]] = self._model.loss_fn(
            parsed_data["predictions"], self._data_set["targets"]
        )

    def _update_accuracy(self, parsed_data: dict):
        """
        Update the accuracy array.

        Parameters
        ----------
        parsed_data : dict
                Data computed before the update to prevent repeated calculations.
        """
        try:
            self._accuracy_array[parsed_data["epoch"]] = self._model.accuracy_fn(
                parsed_data["predictions"], self._data_set["targets"]
            )
        except AttributeError:
            print(
                "There is no accuracy function in the model class, switching this "
                "recording option off."
            )
            self.accuracy = False
            self._read_selected_attributes()

    def _update_ntk(self, parsed_data: dict):
        """
        Update the ntk array.

        Parameters
        ----------
        parsed_data : dict
                Data computed before the update to prevent repeated calculations.
        """
        self._ntk_array[parsed_data["epoch"]] = parsed_data["ntk"]

    def _update_entropy(self, parsed_data: dict):
        """
        Update the entropy array.

        Parameters
        ----------
        parsed_data : dict
                Data computed before the update to prevent repeated calculations.
        """
        calculator = EntropyAnalysis(matrix=parsed_data["ntk"])
        entropy = calculator.compute_von_neumann_entropy(
            effective=False, normalize_eig=True
        )
        self._entropy_array[parsed_data["epoch"]] = entropy

    def _update_eigenvalues(self, parsed_data: dict):
        """
        Update the eigenvalue array.

        Parameters
        ----------
        parsed_data : dict
                Data computed before the update to prevent repeated calculations.
        """
        calculator = EigenSpaceAnalysis(matrix=parsed_data["ntk"])
        eigenvalues = calculator.compute_eigenvalues(normalize=False)
        self._eigenvalues_array[parsed_data["epoch"]] = eigenvalues

    def export_dataset(self):
        """
        Export a dataclass of used properties.

        Returns
        -------
        dataset : object
                A dataclass of only the data recorder during the training.
        """
        DataSet = make_dataclass(
            "DataSet",
            [(item, np.ndarray) for item in self._selected_properties]
        )
        selected_data = {
            item: vars(self)[f"_{item}_array"][:self._index_count] for item in self._selected_properties
        }

        return DataSet(**selected_data)
