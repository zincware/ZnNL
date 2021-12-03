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
    # data_generator = znrnd.data.PointsOnCircle(radius=1.0, noise=1e-3)
    # data_generator.build_pool("uniform", n_points=10, noise=False)
    data_generator = znrnd.data.PointsOnLattice()
    data_generator.build_pool(x_points=10, y_points=10)

    target = znrnd.models.DenseModel(
        units=(4, 4, 4),
        in_d=2,
        activation='tanh',
        out_d=4,
        tolerance=1e-2,
        loss=znrnd.loss_functions.LPNormLoss(order=2)
    )
    predictor = znrnd.models.DenseModel(
        units=(4, 4, 4),
        in_d=2,
        activation='tanh',
        out_d=4,
        tolerance=1e-2,
        loss=znrnd.loss_functions.LPNormLoss(order=2)
    )

    agent = znrnd.RND(
        point_selector=znrnd.point_selection.GreedySelection(threshold=10),
        distance_metric=znrnd.distance_metrics.MahalanobisDistance(),
        # distance_metric=znrnd.distance_metrics.LPNorm(order=2),
        data_generator=data_generator,
        target_network=target,
        predictor_network=predictor,
        tolerance=5,
        # target_size=10,
    )
    agent.run_rnd()
    target_set = np.array(agent.target_set)
    print(target_set)

    fig, (plot1, plot2, plot3) = plt.subplots(3, figsize=(4, 12))
    results = np.array(agent.metric_results_storage)
    plot1.plot(results[:, 0], label='0')
    plot1.plot(results[:, 1], label='1')
    plot1.plot(results[:, -1], label='2')

    # for i, array in enumerate(agent.metric_results_storage):
    #     plot1.plot(array, label=i)
    plot1.legend()

    plot2.plot(data_generator.data_pool[:, 0], data_generator.data_pool[:, 1], ".")
    plot2.plot(target_set[:, 0], target_set[:, 1], "x", ms=8, mew=3)
    plot2.set_aspect("equal", adjustable="box")

    ind = np.argpartition(1 / agent.metric_results, -10)[-10:]
    plot2.plot(data_generator.data_pool[:, 0][ind], data_generator.data_pool[:, 1][ind],
               "_", ms=8, mew=2)

    plot3.plot(1/agent.metric_results, ".")
    index = []
    for element in target_set:
        index.append(np.where((data_generator.data_pool == element).all(axis=1))[0][0])
    plot3.plot(index, 1/agent.metric_results.numpy()[index], "x", ms=8, mew=3)
    plot3.plot(ind, 1 / agent.metric_results.numpy()[ind], "_", ms=8, mew=2)

    plt.show()


