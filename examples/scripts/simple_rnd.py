"""
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincware Project.

Description: an example script to test the RND on functionality
and investigate the symmetry conservation in representation space
"""
import znrnd
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    """
    Main method to run the routine.
    """
    data_generator = znrnd.data.PointsOnLattice()
    data_generator.build_pool(x_points=10, y_points=10)

    target = znrnd.models.DenseModel(
        units=(12, 12, 12),
        in_d=2,
        activation="sigmoid",
        out_d=12,
        tolerance=1e-3,
        loss=znrnd.loss_functions.MeanPowerLoss(order=2),
    )
    predictor = znrnd.models.DenseModel(
        units=(12, 12, 12),
        in_d=2,
        activation="sigmoid",
        out_d=12,
        tolerance=1e-3,
        loss=znrnd.loss_functions.MeanPowerLoss(order=2),
    )

    agent = znrnd.RND(
        point_selector=znrnd.point_selection.GreedySelection(threshold=0.0001),
        distance_metric=znrnd.distance_metrics.OrderNDifference(order=2),
        data_generator=data_generator,
        target_network=target,
        predictor_network=predictor,
        tolerance=8,
        target_size=10,
    )
    agent.run_rnd()
    target_set = np.array(agent.target_set)
    print(target_set)
    # print(agent.metric_results_storage)

    # fig, (plot1, plot2) = plt.subplots(2, figsize=(4, 8))
    # results = np.array(agent.metric_results_storage)
    # plot1.plot(results[:, 0], label="0")
    # plot1.plot(results[:, 1], label="1")
    # plot1.plot(results[:, 2], label="2")
    #
    # # for i, array in enumerate(agent.metric_results_storage):
    # #     plot1.plot(array, label=i)
    # plot1.legend()
    #
    # plot2.plot(data_generator.data_pool[:, 0], data_generator.data_pool[:, 1], ".")
    # plot2.plot(target_set[:, 0], target_set[:, 1], "x", ms=8, mew=3)
    # plot2.set_aspect("equal", adjustable="box")
    #
    # plt.show()
