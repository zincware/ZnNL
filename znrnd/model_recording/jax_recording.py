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
import logging
from dataclasses import dataclass, make_dataclass

import numpy as onp

from znrnd.analysis.eigensystem import EigenSpaceAnalysis
from znrnd.analysis.entropy import EntropyAnalysis
from znrnd.models.jax_model import JaxModel
from znrnd.utils.matrix_utils import calculate_l_pq_norm

logger = logging.getLogger(__name__)


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
    loss_derivative : bool (default=False)
            If true, the derivative of the loss function with respect to the network
            output will be recorded.
    update_rate : int (default=1)
            How often the values are updated.

    Notes
    -----
    Currently the options are hard-coded. In the future, we will work towards allowing
    for arbitrary computations to be added, for example, two losses.
    """

    # Model Loss
    loss: bool = True
    _loss_array: list = None

    # Model accuracy
    accuracy: bool = False
    _accuracy_array: list = None

    # NTK Matrix
    ntk: bool = False
    _ntk_array: list = None

    # Entropy of the model
    entropy: bool = False
    _entropy_array: list = None

    # Model eigenvalues
    eigenvalues: bool = False
    _eigenvalues_array: list = None

    # Model eigenvalues
    loss_derivative: bool = False
    _loss_derivative_array: list = None

    # Class helpers
    update_rate: int = 1
    _selected_properties: list = None
    _model: JaxModel = None
    _data_set: dict = None
    _compute_ntk: bool = False  # Helps to know so we can compute it once and share.
    _compute_loss_derivative: bool = False
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

    def _build_or_resize_array(self, name: str, overwrite: bool):
        """
        Build or resize an array.

        Parameters
        ----------
        name : str
                Name of array. Needed to check for resizing.
        overwrite : bool
                If True, arrays will no be resized but overwritten.

        Returns
        -------
        A np zeros array or a resized array padded with zeros.
        """
        # Check if array exists
        data = getattr(self, name)

        # Create array if none or if overwrite is set true
        if data is None or overwrite:
            data = []

        return data

    def instantiate_recorder(self, data_set: dict, overwrite: bool = False):
        """
        Prepare the recorder for training.

        Parameters
        ----------
        data_set : dict (default=None)
                Data to record during training.
        overwrite : bool (default=False)
                If true and there is data already in the array, this will be removed and
                a new array created.

        Returns
        -------
        Populates the array attributes of the dataclass.
        """
        # Update simple attributes
        self._data_set = data_set

        # populate the class attribute
        self._read_selected_attributes()

        # Instantiate arrays
        all_attributes = self.__dict__
        for item in self._selected_properties:
            if item == "ntk":
                all_attributes[f"_{item}_array"] = self._build_or_resize_array(
                    f"_{item}_array", overwrite
                )
            else:
                all_attributes[f"_{item}_array"] = self._build_or_resize_array(
                    f"_{item}_array", overwrite
                )

        # If over-writing, reset the index count
        if overwrite:
            self._index_count = 0

        # Check if we need an NTK computation and update the class accordingly
        if any(
            [
                "ntk" in self._selected_properties,
                "entropy" in self._selected_properties,
                "eigenvalues" in self._selected_properties,
            ]
        ):
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

            parsed_data = {"epoch": self._index_count}

            # Add epoch to the parsed data
            # Compute representations here to avoid repeated computation.
            predictions = self._model(self._data_set["inputs"])
            if type(predictions) is tuple:
                predictions = predictions[0]
            parsed_data["predictions"] = predictions

            # Compute ntk here to avoid repeated computation.
            if self._compute_ntk:
                try:
                    ntk = self._model.compute_ntk(
                        self._data_set["inputs"], normalize=False, infinite=False
                    )
                    parsed_data["ntk"] = ntk["empirical"]
                except NotImplementedError:
                    logger.info(
                        "NTK calculation is not yet available for this model. Removing "
                        "it from this recorder."
                    )
                    self.ntk = False
                    self.entropy = False
                    self.eigenvalues = False
                    self._read_selected_attributes()

            for item in self._selected_properties:
                call_fn = getattr(self, f"_update_{item}")  # get the callable function

                # Try to add data and resize if necessary.
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
        self._loss_array.append(
            self._model.loss_fn(parsed_data["predictions"], self._data_set["targets"])
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
            self._accuracy_array.append(
                self._model.accuracy_fn(
                    parsed_data["predictions"], self._data_set["targets"]
                )
            )
        except AttributeError:
            logger.info(
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
        self._ntk_array.append(parsed_data["ntk"])

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
        self._entropy_array.append(entropy)

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
        self._eigenvalues_array.append(eigenvalues)

    def _update_loss_derivative(self, parsed_data):
        """
        Update the loss derivative array.

        The loss derivative is normalized by the L_pq matrix norm.

        Parameters
        ----------
        parsed_data : dict
                Data computed before the update to prevent repeated calculations.
        """
        vector_loss_derivative = self._model.calculate_loss_derivative_jit(
            parsed_data["predictions"], self._data_set["targets"]
        )
        loss_derivative = calculate_l_pq_norm(vector_loss_derivative)
        self._loss_derivative_array.append(loss_derivative)

    def export_dataset(self):
        """
        Export a dataclass of used properties.

        Returns
        -------
        dataset : object
                A dataclass of only the data recorder during the training.
        """
        DataSet = make_dataclass(
            "DataSet", [(item, onp.ndarray) for item in self._selected_properties]
        )
        selected_data = {
            item: onp.array(vars(self)[f"_{item}_array"])
            for item in self._selected_properties
        }

        return DataSet(**selected_data)
