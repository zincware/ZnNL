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
            loss=True, accuracy=True, ntk=True, entropy=True, eigenvalues=True
        )
        recorder.instantiate_recorder(data_set=self.dummy_data_set)

        for key, val in vars(recorder).items():
            if key[0] != "_" and key != "update_rate":
                assert val is True
            if key == "update_rate":
                assert val == 1
            elif key == "_ntk_array":
                assert val.shape == (100, 5, 5)
            elif key.split("_")[-1] == "array:":
                assert val.shape == (100,)
            elif key == "_selected_properties":
                pass

    def test_array_resizing(self):
        """
        Test that the array is resized correctly.
        """
        recorder = JaxRecorder(
            loss=True, accuracy=False, ntk=True, entropy=False, eigenvalues=False
        )
        recorder.instantiate_recorder(data_set=self.dummy_data_set)

        # Populate the arrays deliberately.
        recorder._ntk_array = onp.random.uniform(size=(10, 5, 5))
        recorder._loss_array = onp.random.uniform(size=(10,))

        new_ntk = recorder._build_or_resize_array("_ntk_array", (5, 5), False)
        new_loss = recorder._build_or_resize_array("_loss_array", (), False)

        # Assert correct shapes
        assert new_ntk.shape == (110, 5, 5)
        assert new_loss.shape == (110,)

        # Assert data is not lost and new data is 0.0
        assert new_ntk[:10].sum() != 0.0
        assert new_ntk[10:].sum() == 0.0
        assert new_loss[:10].sum() != 0.0
        assert new_loss[10:].sum() == 0.0

    def test_overwriting(self):
        """
        Test the overwrite function.
        """
        recorder = JaxRecorder(
            loss=False, accuracy=False, ntk=True, entropy=False, eigenvalues=False
        )
        recorder.instantiate_recorder(data_set=self.dummy_data_set)

        # Populate the arrays deliberately.
        recorder._ntk_array = onp.random.uniform(size=(10, 5, 5))
        assert recorder._ntk_array.sum() != 0.0  # check the data is there

        recorder.instantiate_recorder(data_set=self.dummy_data_set, overwrite=False)
        # Check normal resizing on instantiation.
        assert recorder._ntk_array.shape == (110, 5, 5)

        recorder.instantiate_recorder(data_set=self.dummy_data_set, overwrite=True)
        assert recorder._ntk_array.shape == (100, 5, 5)
        assert recorder._ntk_array.sum() == 0.0
