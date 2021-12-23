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


def main():
    """
    Main method to run the routine.
    """

    # Define both networks
    networks = (2, 12, 12, 12, 12)
    tolerance = 1e-4
    loss = znrnd.loss_functions.MeanPowerLoss(order=2)

    target = znrnd.models.DenseModel(
        units=networks[1:-1],
        in_d=networks[0],
        activation="tanh",
        out_d=networks[-1],
        tolerance=tolerance,
        loss=loss,
        iterations=20,
        epochs=100,
    )
    predictor = znrnd.models.DenseModel(
        units=networks[1:-1],
        in_d=networks[0],
        activation="tanh",
        out_d=networks[-1],
        tolerance=tolerance,
        loss=loss,
        iterations=20,
        epochs=100,
    )

    agent = znrnd.RND(
        point_selector=znrnd.point_selection.GreedySelection(threshold=1e4),
        distance_metric=znrnd.distance_metrics.LPNorm(order=1),
        data_generator=data_generator,
        target_network=target,
        predictor_network=predictor,
        seed_point=[
            data_generator.data_pool[72 * 0 + 0],
            data_generator.data_pool[72 * 3 + 0],
            data_generator.data_pool[72 * 7 + 0],
            data_generator.data_pool[72 * 0 + 9],
            data_generator.data_pool[72 * 3 + 9],
            data_generator.data_pool[72 * 7 + 9],
            data_generator.data_pool[72 * 0 + 27],
            data_generator.data_pool[72 * 3 + 27],
            data_generator.data_pool[72 * 7 + 27],
        ],
        tolerance=5,
        # target_size=10,
    )
    agent.run_rnd()

    return agent.target_set, preprocess_data(agent.metric_results.numpy())


def preprocess_data(data):
    distances = np.array(data)
    normed_distances = distances / np.amax(distances)
    membership = 1 - normed_distances
    exp_membership = np.power(membership, 2)
    return exp_membership / np.amax(exp_membership)


if __name__ == "__main__":

    target_set = []
    norm_scaled_membership = []

    data_generator = znrnd.data.PointsOnCircle(
        radius=np.arange(2.0, 0.0, -0.2), noise=1e-3
    )
    data_generator.build_pool("uniform", n_points=72, noise=False)

    runs = 0
    for runs in range(5):
        results = main()
        target_set.append(results[0])
        norm_scaled_membership.append(results[1])

    target_set = np.array(target_set)
    mean_target_set = target_set.mean(axis=0)
    norm_scaled_membership = np.array(norm_scaled_membership)
    mean_norm_scaled_membership = norm_scaled_membership.mean(axis=0)
    print(mean_target_set)

    fig, ax = plt.subplots(2, figsize=(4, 8), constrained_layout=True)

    ax[0].set_aspect("equal", adjustable="box")
    ax[0].scatter(
        data_generator.data_pool[:, 0],
        data_generator.data_pool[:, 1],
        s=80,
        c=mean_norm_scaled_membership,
        cmap="gnuplot2",
    )
    ax[0].plot(mean_target_set[:, 0], mean_target_set[:, 1], "o", ms=8, mew=1,
               fillstyle="none")

    # ax[1].plot(data_generator.data_pool[:, 0], data_generator.data_pool[:, 1], ".")
    # ax[1].plot(target_set[:, 0], target_set[:, 1], "x", ms=8, mew=3)
    # ax[1].set_aspect("equal", adjustable="box")

    ax[1].set_box_aspect(1 / 1)
    add_points = len(mean_target_set) + 10
    ind = np.argpartition(mean_norm_scaled_membership, -add_points)[-add_points:]
    index = []
    # Detect maximum
    for element in target_set[0]:
        index.append(np.where((data_generator.data_pool == element).all(axis=1))[0][0])
    ax[1].plot(mean_norm_scaled_membership, ".")
    ax[1].plot(index, mean_norm_scaled_membership[index], "x", ms=8, mew=3)
    ax[1].plot(ind, mean_norm_scaled_membership[ind], "_", ms=8, mew=2)
    # ax[1].set_yscale("log")

    plt.show()
