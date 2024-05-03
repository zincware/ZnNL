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

import logging
from dataclasses import dataclass, make_dataclass
from os import path
from pathlib import Path
from typing import Optional

import jax.numpy as np
import numpy as onp

from znnl.accuracy_functions.accuracy_function import AccuracyFunction
from znnl.analysis.eigensystem import EigenSpaceAnalysis
from znnl.analysis.entropy import EntropyAnalysis
from znnl.analysis.loss_fn_derivative import LossDerivative
from znnl.loss_functions import SimpleLoss
from znnl.models.jax_model import JaxModel
from znnl.training_recording.data_storage import DataStorage
from znnl.utils.matrix_utils import (
    calculate_trace,
    compute_magnitude_density,
    flatten_rank_4_tensor,
    normalize_gram_matrix,
)

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
    name : str
            Name of the recorder.
    storage_path : str
            Where to store the data on disk.
    chunk_size : int
            Amount of data to keep in memory before it is dumped to a hdf5 database.
    loss : bool (default=True)
            If true, loss will be recorded.
    accuracy : bool (default=False)
            If true, accuracy will be recorded.
    network_predictions : bool (default=False)
            If true, network predictions will be recorded.
    ntk : bool (default=False)
            If true, the ntk will be recorded. Warning, large overhead.
    covariance_ntk : bool (default = False)
            If true, the covariance of the ntk will be recorded.
            Warning, large overhead.
    magnitude_ntk : bool (default = False)
            If true, gradient magnitudes of the ntk will be recorded.
            Warning, large overhead.
    entropy : bool (default=False)
            If true, entropy will be recorded. Warning, large overhead.
    covariance_entropy : bool (default=False)
            If true, the entropy of the covariance ntk will be recorded.
            Warning, large overhead.
    magnitude_variance : bool (default=False)
            If true, the variance of the gradient magnitudes of the ntk will be
            recorded.
    magnitude_entropy : bool (default=False)
            If true, the entropy of the gradient magnitudes of the ntk will be recorded.
            Warning, large overhead.
    eigenvalues : bool (default=False)
            If true, eigenvalues will be recorded. Warning, large overhead.
    loss_derivative : bool (default=False)
            If true, the derivative of the loss function with respect to the network
            output will be recorded.
    update_rate : int (default=1)
            How often the values are updated.
    flatten_ntk : bool (default=False)
            If true, an NTK of rank 4 will be flattened to a rank 2 tensor.
            In case of an NTK of rank 2, this has no effect.

    Notes
    -----
    Currently the options are hard-coded. In the future, we will work towards allowing
    for arbitrary computations to be added, for example, two losses.
    """

    # Recorder Attributes
    name: str = "my_recorder"
    storage_path: str = "./"
    chunk_size: int = 100
    flatten_ntk: bool = True

    # Model Loss
    loss: bool = True
    _loss_array: list = None

    # Model accuracy
    accuracy: bool = False
    _accuracy_array: list = None

    # Model predictions
    network_predictions: bool = False
    _network_predictions_array: list = None

    # NTK Matrix
    ntk: bool = False
    _ntk_array: list = None

    # Covariance NTK Matrix
    covariance_ntk: bool = False
    _covariance_ntk_array: list = None

    # Magnitude NTK array
    magnitude_ntk: bool = False
    _magnitude_ntk_array: list = None

    # Entropy of the model
    entropy: bool = False
    _entropy_array: list = None

    # Covariance Entropy of the model
    covariance_entropy: bool = False
    _covariance_entropy_array: list = None

    # Magnitude Variance of the model
    magnitude_variance: bool = False
    _magnitude_variance_array: list = None

    # Magnitude Entropy of the model
    magnitude_entropy: bool = False
    _magnitude_entropy_array: list = None

    # Model eigenvalues
    eigenvalues: bool = False
    _eigenvalues_array: list = None

    # Model trace
    trace: bool = False
    _trace_array: list = None

    # Loss derivative
    loss_derivative: bool = False
    _loss_derivative_array: list = None

    # Class helpers
    update_rate: int = 1
    _loss_fn: SimpleLoss = None
    _accuracy_fn: AccuracyFunction = None
    _selected_properties: list = None
    _model: JaxModel = None
    _data_set: dict = None
    _compute_ntk: bool = False  # Helps to know if we can compute it once and share.
    _compute_loss_derivative: bool = False
    _loss_derivative_fn: LossDerivative = False
    _index_count: int = 0  # Helps to avoid problems with non-1 update rates.
    _data_storage: DataStorage = None  # For writing to disk.
    _ntk_rank: Optional[int] = None  # Rank of the NTK matrix.

    def _read_selected_attributes(self):
        """
        Helper function to read selected attributes.
        """
        # populate the class attribute
        self._selected_properties = [
            value
            for value in list(vars(self))
            if value[0] != "_" and vars(self)[value] is True and value != "flatten_ntk"
        ]

    def _build_or_resize_array(self, name: str, overwrite: bool):
        """
        Build or resize an array.

        Parameters
        ----------
        name : str
                Name of array. Needed to check for resizing.
        overwrite : bool
                If True, arrays will not be resized but overwritten.

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

    def instantiate_recorder(self, data_set: dict = None, overwrite: bool = False):
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
        # Create the data storage manager.
        _storage_path = path.join(self.storage_path, self.name)
        self._data_storage = DataStorage(Path(_storage_path))

        if data_set:
            # Update simple attributes
            self._data_set = data_set
        if self._data_set is None and data_set is None:
            raise AttributeError(
                "No data set given for the recording process."
                "Instantiate the recorder with a data set."
            )

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
                "covariance_ntk" in self._selected_properties,
                "magnitude_ntk" in self._selected_properties,
                "entropy" in self._selected_properties,
                "magnitude_entropy" in self._selected_properties,
                "magnitude_variance" in self._selected_properties,
                "covariance_entropy" in self._selected_properties,
                "eigenvalues" in self._selected_properties,
                "trace" in self._selected_properties,
            ]
        ):
            self._compute_ntk = True

        if "loss_derivative" in self._selected_properties:
            self._loss_derivative_fn = LossDerivative(self._loss_fn)

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
                        self._data_set["inputs"], infinite=False
                    )["empirical"]
                    self._ntk_rank = len(ntk.shape)
                    if self.flatten_ntk and self._ntk_rank == 4:
                        ntk = flatten_rank_4_tensor(ntk)
                    parsed_data["ntk"] = ntk
                except NotImplementedError:
                    logger.info(
                        "NTK calculation is not yet available for this model. Removing "
                        "it from this recorder."
                    )
                    self.ntk = False
                    self.covariance_ntk = False
                    self.magnitude_ntk = False
                    self.entropy = False
                    self.magnitude_entropy = False
                    self.magnitude_variance = False
                    self.covariance_entropy = False
                    self.eigenvalues = False
                    self._read_selected_attributes()

            for item in self._selected_properties:
                call_fn = getattr(self, f"_update_{item}")  # get the callable function

                # Try to add data and resize if necessary.
                call_fn(parsed_data)  # call the function and update the property

            self._index_count += 1  # Update the index count.
        else:
            pass

        # Dump records if the index hits the chunk size.
        if self._index_count == self.chunk_size:
            self.dump_records()

    def dump_records(self):
        """
        Dump recorded properties to hdf5 database.
        """
        export_data = self._export_in_memory_data()  # collect the in-memory data.
        self._data_storage.write_data(export_data)
        self.instantiate_recorder(self._data_set, overwrite=True)  # clear data.

    def visualize_recorder(self):
        """
        Display recorded values as web app.

        Returns
        -------

        """
        raise NotImplementedError("Not yet available in ZnRND.")

    @property
    def loss_fn(self):
        """
        The loss function property of a recorder.

        Returns
        -------
        The loss function used in the recorder.
        """
        return self._loss_fn

    @loss_fn.setter
    def loss_fn(self, loss_fn: SimpleLoss):
        """
        Setting a loss function for a recorder.

        Parameters
        ----------
        loss_fn : SimpleLoss
                Loss function used for recording.
        """
        self._loss_fn = loss_fn

    @property
    def accuracy_fn(self):
        """
        The accuracy function property of a recorder.

        Returns
        -------
        The accuracy function used in the recorder.
        """
        return self._accuracy_fn

    @accuracy_fn.setter
    def accuracy_fn(self, accuracy_fn: AccuracyFunction):
        """
        Setting an accuracy function for a recorder.

        Parameters
        ----------
        accuracy_fn : AccuracyFunction
                Accuracy function used for recording.
        """
        self._accuracy_fn = accuracy_fn

    def _update_loss(self, parsed_data: dict):
        """
        Update the loss array.

        Parameters
        ----------
        parsed_data : dict
                Data computed before the update to prevent repeated calculations.
        """
        self._loss_array.append(
            self._loss_fn(parsed_data["predictions"], self._data_set["targets"])
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
                self._accuracy_fn(parsed_data["predictions"], self._data_set["targets"])
            )
        except TypeError:
            logger.info(
                "There is no accuracy function defined in the training procedure, "
                "switching this recording option off."
            )
            self.accuracy = False
            self._read_selected_attributes()

    def _update_network_predictions(self, parsed_data: dict):
        """
        Update the network predictions array.

        Parameters
        ----------
        parsed_data : dict
                Data computed before the update to prevent repeated calculations.
        """
        self._network_predictions_array.append(parsed_data["predictions"])

    def _update_ntk(self, parsed_data: dict):
        """
        Update the ntk array.

        Parameters
        ----------
        parsed_data : dict
                Data computed before the update to prevent repeated calculations.
        """
        self._ntk_array.append(parsed_data["ntk"])

    def _update_covariance_ntk(self, parsed_data: dict):
        """
        Update the covariance ntk array.

        Parameters
        ----------
        parsed_data : dict
                Data computed before the update to prevent repeated calculations.
        """
        cov_ntk = normalize_gram_matrix(parsed_data["ntk"])
        self._covariance_ntk_array.append(cov_ntk)

    def _update_magnitude_ntk(self, parsed_data: dict):
        """
        Update the magnitude ntk array.

        Parameters
        ----------
        parsed_data : dict
                Data computed before the update to prevent repeated calculations.
        """
        magnitude_dist = compute_magnitude_density(gram_matrix=parsed_data["ntk"])
        self._magnitude_ntk_array.append(magnitude_dist)

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

    def _update_covariance_entropy(self, parsed_data: dict):
        """
        Update the entropy of the covariance NTK.

        The covariance ntk is defined as the of cosine similarities. For this each
        entry of the NTK is re-scaled by the gradient amplitudes.

        Parameters
        ----------
        parsed_data : dict
                Data computed before the update to prevent repeated calculations.
        """
        cov_ntk = normalize_gram_matrix(parsed_data["ntk"])
        calculator = EntropyAnalysis(matrix=cov_ntk)
        entropy = calculator.compute_von_neumann_entropy(
            effective=False, normalize_eig=True
        )
        self._covariance_entropy_array.append(entropy)

    def _update_magnitude_entropy(self, parsed_data: dict):
        """
        Update magnitude entropy of the NTK.

        The magnitude entropy is defined as the entropy of the normalized gradient
        magnitudes.

        Parameters
        ----------
        parsed_data : dict
                Data computed before the update to prevent repeated calculations.
        """
        magnitude_dist = compute_magnitude_density(gram_matrix=parsed_data["ntk"])
        entropy = EntropyAnalysis.compute_shannon_entropy(magnitude_dist)
        self._magnitude_entropy_array.append(entropy)

    def _update_magnitude_variance(self, parsed_data: dict):
        """
        Update the magnitude variance of the NTK.

        The magnitude variance is defined as the variance of the normalized gradient
        magnitudes.
        As the normalization to obtain the magnitude distribution is done by dividing
        by the sum of the magnitudes, the variance is calculated as:

            magnitude_variance = var(magnitudes * magnitudes.shape[0])

        This ensures that the variance is not dependent on the number entries in the
        magnitude distribution.
        It is equivalent to the following:

            ntk_diag = sqrt( diag(ntk) )
            magnitude_variance = var( diag / mean(ntk_diag) )

        Parameters
        ----------
        parsed_data : dict
                Data computed before the update to prevent repeated calculations.
        """
        magnitude_dist = compute_magnitude_density(gram_matrix=parsed_data["ntk"])
        magvar = np.var(magnitude_dist * magnitude_dist.shape[0])
        self._magnitude_variance_array.append(magvar)

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

    def _update_trace(self, parsed_data: dict):
        """
        Update the trace of the NTK.

        The trace of the NTK is computed as the mean of the diagonal elements of the
        NTK.

        Parameters
        ----------
        parsed_data : dict
                Data computed before the update to prevent repeated calculations.
        """
        trace = calculate_trace(parsed_data["ntk"], normalize=True)
        self._trace_array.append(trace)

    def _update_loss_derivative(self, parsed_data):
        """
        Update the loss derivative array.

        The loss derivative records the derivative of the loss function with respect to
        the network output, returning a vector of the same shape as the network output.

        Parameters
        ----------
        parsed_data : dict
                Data computed before the update to prevent repeated calculations.
        """
        vector_loss_derivative = self._loss_derivative_fn.calculate(
            parsed_data["predictions"], self._data_set["targets"]
        )
        self._loss_derivative_array.append(vector_loss_derivative)

    def gather_recording(self, selected_properties: list = None) -> dataclass:
        """
        Export a dataclass of used properties.

        Parameters
        ----------
        selected_properties : list default = None
                List of parameters to export. If None, all available are exported.

        Returns
        -------
        dataset : object
                A dataclass of only the data recorder during the training.
        """
        if selected_properties is None:
            selected_properties = self._selected_properties
        else:
            # Check if we can collect all the data.
            comparison = [i in self._selected_properties for i in selected_properties]
            # Throw away data that was not saved in the first place.
            if not all(comparison):
                logger.info(
                    "You have asked for properties that were not recorded. Removing"
                    " the impossible elements."
                )
                selected_properties = onp.array(selected_properties)[
                    onp.array(comparison).astype(int) == 1
                ]

        DataSet = make_dataclass(
            "DataSet", [(item, onp.ndarray) for item in selected_properties]
        )
        selected_data = {
            item: onp.array(vars(self)[f"_{item}_array"])
            for item in selected_properties
        }

        # Try to load some data from the hdf5 database.
        try:
            db_data = self._data_storage.fetch_data(selected_properties)
            # Add db data to the selected data dict.
            for item, data in selected_data.items():
                selected_data[item] = onp.concatenate((db_data[item], data), axis=0)

        except FileNotFoundError:  # There is no database.
            pass

        return DataSet(**selected_data)

    def _export_in_memory_data(self) -> dataclass:
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
