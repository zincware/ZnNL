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
from znrnd.model_recording import JaxRecorder

import numpy as np


class TestModelRecording:
    """
    Unit test suite for the model recording.
    """

    def test_instantiation(self):
        """
        Test the instantiation of the recorder.
        """
        dummy_data = np.random.uniform(size=(5, 2, 3))
        dummy_data_set = {"inputs": dummy_data, "targets": dummy_data}
        recorder = JaxRecorder(
            loss=True, accuracy=True, ntk=True, entropy=True, eigenvalues=True
        )
        recorder.instantiate_recorder(
            model=None,
            data_length=10,
            data_set=dummy_data_set
        )

        for key, val in vars(recorder).items():
            if key[0] != "_" and key != "update_rate":
                assert val is True
            if key == "update_rate":
                assert val == 1
            elif key == "_ntk_array":
                assert val.shape == (10, 5, 5)
            elif key.split("_")[-1] == "array:":
                assert val.shape == (10,)
            elif key == "_selected_properties":
                pass
