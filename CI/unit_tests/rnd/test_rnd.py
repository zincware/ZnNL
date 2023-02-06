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
Test the RND class.
"""
import copy
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import optax
from neural_tangents import stax

import znrnd.data
from znrnd.data import PointsOnLattice
from znrnd.loss_functions import MeanPowerLoss
from znrnd.models import NTModel
from znrnd.training_strategies import SimpleTraining


class TestRNDTraining:
    """
    Class to test the initialization of the RND training
    """

    def test_rnd_predictor_init(self):
        """
        Asserts whether the predictor model and the model inside the training strategy
        are the same.
        """

        predictor_model = NTModel(
            nt_module=stax.serial(stax.Dense(5), stax.Relu(), stax.Dense(1)),
            optimizer=optax.adam(learning_rate=0.001),
            input_shape=(1, 8),
        )
        target_model = copy.deepcopy(predictor_model)

        trainer = SimpleTraining(
            model=None,
            loss_fn=MeanPowerLoss(order=2),
        )
        # Create data generator
        data_gen = PointsOnLattice()
        data_gen.build_pool()

        agent_dict = {
            "data_generator": data_gen,
            "target_network": target_model,
            "predictor_network": predictor_model,
            "distance_metric": znrnd.distance_metrics.OrderNDifference(order=2),
            "point_selector": znrnd.point_selection.GreedySelection(),
        }
        agent = znrnd.agents.RND(
            training_strategy=trainer,
            **agent_dict,
        )
        assert agent.training_strategy.model is predictor_model
