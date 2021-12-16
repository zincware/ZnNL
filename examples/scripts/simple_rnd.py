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

    networks = (2, 24, 24, 24)
    tolerance = 1e-4
    loss = znrnd.loss_functions.MeanPowerLoss(order=2)

    target = znrnd.models.DenseModel(
        units=networks[1:-1],
        in_d=networks[0],
        activation="tanh",
        out_d=networks[-1],
        tolerance=tolerance,
        loss=loss,
    )
    predictor = znrnd.models.DenseModel(
        units=networks[1:-1],
        in_d=networks[0],
        activation="tanh",
        out_d=networks[-1],
        tolerance=tolerance,
        loss=loss,
    )

    agent = znrnd.RND(
        point_selector=znrnd.point_selection.GreedySelection(threshold=1e3),
        # distance_metric=znrnd.distance_metrics.MahalanobisDistance(),
        distance_metric=znrnd.distance_metrics.LPNorm(order=2),
        # distance_metric=znrnd.distance_metrics.HyperSphere(order=2),
        # distance_metric=znrnd.distance_metrics.CosineDistance(),
        data_generator=data_generator,
        target_network=target,
        predictor_network=predictor,
        seed_point=[[-5.0, -5.0], [-5.0, 5.0]],
        tolerance=5,
        # target_size=10,
    )
    agent.run_rnd()
    target_set = np.array(agent.target_set)
    print(target_set)
    print(data_generator.data_pool)

    fig, ax = plt.subplots(3, figsize=(4, 12))
    results = np.array(agent.metric_results_storage)

    # Show how biggest loss decreases
    # plot1.plot(results[:, 0], label='0')
    # plot1.plot(results[:, 1], label='1')
    # plot1.plot(results[:, -1], label='2')
    # plot1.legend()

    inv_distance = 1 / agent.metric_results.numpy()
    map_plot = np.log10(np.flip(inv_distance.reshape(11, 11), axis=0))
    ax[0].imshow(map_plot)

    ax[1].plot(data_generator.data_pool[:, 0], data_generator.data_pool[:, 1], ".")
    ax[1].plot(target_set[:, 0], target_set[:, 1], "x", ms=8, mew=3)
    ax[1].set_aspect("equal", adjustable="box")

    add_points = len(target_set) + 30
    ind = np.argpartition(1 / agent.metric_results, -add_points)[-add_points:]
    ax[1].plot(
        data_generator.data_pool[:, 0][ind],
        data_generator.data_pool[:, 1][ind],
        "_",
        ms=8,
        mew=2,
    )

    ax[2].plot(1 / agent.metric_results, ".")
    index = []
    # Detect maximum
    for element in target_set:
        index.append(np.where((data_generator.data_pool == element).all(axis=1))[0][0])
    ax[2].plot(index, inv_distance[index], "x", ms=8, mew=3)
    ax[2].plot(ind, inv_distance[ind], "_", ms=8, mew=2)
    ax[2].set_yscale("log")
    plt.show()
