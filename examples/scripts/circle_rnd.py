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
    data_generator = znrnd.data.PointsOnCircle(radius=np.arange(2., 0., -0.2), noise=1e-3)
    data_generator.build_pool("uniform", n_points=72, noise=False)

    # Define both networks
    networks = (2, 24, 24, 24)
    tolerance = 1e-4
    loss = znrnd.loss_functions.MeanPowerLoss(order=2)

    target = znrnd.models.DenseModel(
        units=networks[1:-1],
        in_d=networks[0],
        activation='tanh',
        out_d=networks[-1],
        tolerance=tolerance,
        loss=loss
    )
    predictor = znrnd.models.DenseModel(
        units=networks[1:-1],
        in_d=networks[0],
        activation='tanh',
        out_d=networks[-1],
        tolerance=tolerance,
        loss=loss
    )

    agent = znrnd.RND(
        point_selector=znrnd.point_selection.GreedySelection(threshold=1e4),
        distance_metric=znrnd.distance_metrics.LPNorm(order=2),
        data_generator=data_generator,
        target_network=target,
        predictor_network=predictor,
        seed_point=[
            data_generator.data_pool[72 * 0 + 0],
            data_generator.data_pool[72 * 0 + 18],
            data_generator.data_pool[72 * 0 + 36],
         ],
        tolerance=5,
        # target_size=10,
    )
    agent.run_rnd()
    target_set = np.array(agent.target_set)
    print(target_set)

    fig, ax = plt.subplots(2, figsize=(4, 8))
    results = np.array(agent.metric_results_storage)

    inv_distance = 1 / agent.metric_results.numpy()
    log_inv_dist = np.log(inv_distance)

    ax[0].scatter(data_generator.data_pool[:, 0], data_generator.data_pool[:, 1],
                  s=80, c=log_inv_dist, cmap="inferno")
    ax[0].set_aspect("equal", adjustable="box")

    # ax[1].plot(data_generator.data_pool[:, 0], data_generator.data_pool[:, 1], ".")
    # ax[1].plot(target_set[:, 0], target_set[:, 1], "x", ms=8, mew=3)
    # ax[1].set_aspect("equal", adjustable="box")

    add_points = len(target_set) + 10
    ind = np.argpartition(inv_distance, -add_points)[-add_points:]
    # ax[1].plot(data_generator.data_pool[:, 0][ind], data_generator.data_pool[:, 1][ind],
    #            "_", ms=8, mew=2)

    ax[1].plot(inv_distance, ".")
    index = []
    # Detect maximum
    for element in target_set:
        index.append(np.where((data_generator.data_pool == element).all(axis=1))[0][0])
    ax[1].plot(index, inv_distance[index], "x", ms=8, mew=3)
    ax[1].plot(ind, inv_distance[ind], "_", ms=8, mew=2)
    ax[1].set_yscale("log")
    plt.show()


