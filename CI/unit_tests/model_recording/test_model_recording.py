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
Test for the model recording module.
"""
import numpy as onp

from znrnd.model_recording import JaxRecorder


class TestModelRecording:
    """
    Unit test suite for the model recording.
    """

    @classmethod
    def setup_class(cls):
        """
        Prepare the test suite.
        """
        dummy_data = onp.random.uniform(size=(5, 2, 3))
        cls.dummy_data_set = {"inputs": dummy_data, "targets": dummy_data}

    def test_instantiation(self):
        """
        Test the instantiation of the recorder.
        """

        recorder = JaxRecorder(
            loss=True,
            accuracy=True,
            ntk=True,
            entropy=True,
            eigenvalues=True,
            trace=True,
            loss_derivative=True,
        )
        recorder.instantiate_recorder(data_set=self.dummy_data_set)

        for key, val in vars(recorder).items():
            if key[0] != "_" and key != "update_rate":
                assert val is True
            if key == "update_rate":
                assert val == 1
            elif key.split("_")[-1] == "array:":
                assert val == []
            elif key == "_selected_properties":
                pass

    def test_overwriting(self):
        """
        Test the overwrite function.
        """
        recorder = JaxRecorder(
            loss=False, accuracy=False, ntk=True, entropy=False, eigenvalues=False
        )
        recorder.instantiate_recorder(data_set=self.dummy_data_set)

        # Populate the arrays deliberately.
        recorder._ntk_array = onp.random.uniform(size=(10, 5, 5)).tolist()
        assert onp.sum(recorder._ntk_array) != 0.0  # check the data is there

        # Check normal resizing on instantiation.
        recorder.instantiate_recorder(data_set=self.dummy_data_set, overwrite=False)
        assert onp.shape(recorder._ntk_array) == (10, 5, 5)

        # Test overwriting.
        recorder.instantiate_recorder(data_set=self.dummy_data_set, overwrite=True)
        assert recorder._ntk_array == []
