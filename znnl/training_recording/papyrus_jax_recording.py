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

from typing import List

from papyrus.measurements import BaseMeasurement
from papyrus.recorders import BaseRecorder

from znnl.analysis import JAXNTKComputation
from znnl.models import JaxModel


class JaxRecorder(BaseRecorder):
    """
    Recorder for a fixed dataset.

    Attributes
    ----------
    name : str
            The name of the recorder, defining the name of the file the data will be
            stored in.
    storage_path : str
            The path to the storage location of the recorder.
    measurements : List[BaseMeasurement]
            The measurements that the recorder will apply.
    chunk_size : int
            The size of the chunks in which the data will be stored.
    overwrite : bool (default=False)
            Whether to overwrite the existing data in the database.
    update_rate : int (default=1)
            The rate at which the recorder will update the neural state.
    neural_state_keys : List[str]
            The keys of the neural state that the recorder takes as input.
            A neural state is a dictionary of numpy arrays that represent the state of
            a neural network.
    neural_state : Dict[str, onp.ndarray]
            The neural state dictionary containing a state representation of a neural
            network.
    _data_set : Dict[str, onp.ndarray]
            The dataset that will be used to create the neural state.
            It needs to be a dictionary of numpy arrays with the following keys:
            - "inputs": The ionputs of the dataset.
            - "targets": The targets of the dataset.
    _model : JaxModel
            A neural network module. For more information see the JaxModel class.
    _ntk_computation : JAXNTKComputation
            An NTK computation module. For more information see the JAXNTKComputation
            class.
    """

    def __init__(
        self,
        name: str,
        storage_path: str,
        measurements: List[BaseMeasurement],
        chunk_size: int = 1e5,
        overwrite: bool = False,
        update_rate: int = 1,
    ):
        """
        Constructor method of the BaseRecorder class.

        Parameters
        ----------
        name : str
                The name of the recorder, defining the name of the file the data will be
                stored in.
        storage_path : str
                The path to the storage location of the recorder.
        measurements : List[BaseMeasurement]
                The measurements that the recorder will apply.
        chunk_size : int (default=1e5)
                The size of the chunks in which the data will be stored.
        overwrite : bool (default=False)
                Whether to overwrite the existing data in the database.
        update_rate : int (default=1)
                The rate at which the recorder will update the neural state.
        """
        super().__init__(name, storage_path, measurements, chunk_size, overwrite)
        self.update_rate = update_rate

        self.neural_state = {}

        self._data_set = None
        self._model = None
        self._ntk_computation = None

    def instantiate_recorder(
        self,
        model: JaxModel = None,
        data_set: dict = None,
        ntk_computation: JAXNTKComputation = None,
    ):
        """
        Prepare the recorder for training.

        Instantiate the recorder with the required modules and data set.

        The instantiation method performs the following checks:
            - Check if the neural network module is required and provided.
            - Check if the NTK computation module is required and provided.
            - Check if the data set is provided.

        Parameters
        ----------
        model : JaxModel (default=None)
                The neural network module to record during training.
        data_set : dict (default=None)
                Data to record during training. The first key needs to be the input data
                and the second key the target data.
        ntk_computation : JAXNTKComputation (default=None)
                Computation of the NTK matrix.
                If the ntk is to be computed, this is required.

        Returns
        -------
        Populates the array attributes of the dataclass.
        """

        # Check if the neural network module is required and provided
        if "predictions" in self.neural_state_keys:
            if model is None:
                raise AttributeError(
                    "The neural network module is required for the recording process."
                    "Instantiate the recorder with a JaxModel."
                )

        # Check if the NTK computation module is required and provided
        if "ntk" in self.neural_state_keys:
            if ntk_computation is None:
                raise AttributeError(
                    "The NTK computation module is required for the recording process."
                    "Instantiate the recorder with a JAXNTKComputation module."
                )

        # Check if the data set is provided
        if data_set:
            # Update simple attributes
            self._data_set = data_set
        if self._data_set is None and data_set is None:
            raise AttributeError(
                "No data set given for the recording process."
                "Instantiate the recorder with a data set."
            )

        self._model = model
        self._ntk_computation = ntk_computation

    def _check_keys(self):
        """
        Check if the provided keys match the neural state keys.

        This method checks if the provided keys of the neural state match the required
        keys of the neural state, in other words, if the incoming data is complete.

        Parameters
        ----------
        neural_state : dict
                The neural state dictionary.
        """
        if any([key not in self.neural_state.keys() for key in self.neural_state_keys]):
            raise KeyError(
                "The attributes that are computed do not match the required attributes."
                "The required attributes are: "
                f"{self.neural_state_keys}."
                "The provided attributes are: "
                f"{self.neural_state.keys()}."
            )

    def _compute_neural_state(self, model: JaxModel):
        """
        Compute the neural state.

        Parameters
        ----------
        model : JaxModel
                The neural network module containing the parameters to use for
                recording.

        Returns
        -------
        Updates the neural state dictionary.
        """
        if self._ntk_computation:
            ntk = self._ntk_computation.compute_ntk(
                params={
                    "params": model.model_state.params,
                    "batch_stats": model.model_state.batch_stats,
                },
                dataset_i=self._data_set,
            )
            self.neural_state["ntk"] = ntk
        if self._model:
            predictions = model(self._data_set[list(self._data_set.keys())[0]])
            self.neural_state["predictions"] = [predictions]

    def record(self, epoch: int, model: JaxModel, **kwargs):
        """
        Perform the recording of a neural state.

        Recording is done by measuring and storing the measurements to a database.

        Parameters
        ----------
        epoch : int
                The epoch of the training process.
        model : JaxModel
                The neural network module containing the parameters to use for
                recording.
        kwargs : Any
                Additional keyword arguments that are directly added to the neural
                state.

        Returns
        -------
        result : onp.ndarray
                The result of the recorder.
        """
        if epoch % self.update_rate == 0:
            # Compute the neural state
            self._compute_neural_state(model)
            # Add all other kwargs to the neural state dictionary
            self.neural_state.update(kwargs)
            for key, val in self._data_set.items():
                self.neural_state[key] = [val]
            # Check if incoming data is complete
            self._check_keys()
            # Perform measurements
            self._measure(**self.neural_state)
            # Store the measurements
            self.store(ignore_chunk_size=False)
