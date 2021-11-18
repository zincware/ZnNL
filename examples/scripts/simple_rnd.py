"""
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincware Project.

Description: An example script to compute the number of unique places in a box.
"""
import pyrnd
from pyrnd.core.distance_metrics.distance_metrics import euclidean_distance
from pyrnd.core.point_selection.greedy_selection import GreedySelection
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    """
    Main method to run the routine.
    """
    # data_generator = pyrnd.ConfinedParticles()
    data_generator = pyrnd.PointsOnCircle(radius=1.0, noise=1e-3)
    # Noisy circle.
    # data = pyrnd.PointsOnCircle(noise=0.1)
    data_generator.build_pool("uniform", n_points=100, noise=True)
    # data_generator = pyrnd.ConfinedParticles()

    target = pyrnd.DenseModel(
        units=12,
        layers=4,
        in_d=2,
        out_d=12,
        tolerance=1e-5,
        loss="cosine_similarity",
        activation="tanh",
    )
    predictor = pyrnd.DenseModel(
        units=12,
        layers=4,
        in_d=2,
        out_d=12,
        tolerance=1e-5,
        loss="cosine_similarity",
        activation="tanh",
    )
    # print(target.summary())

    agent = pyrnd.RND(
        point_selector=GreedySelection(threshold=0.1),
        # distance_metric=euclidean_distance,
        data_generator=data_generator,
        target_network=target,
        predictor_network=predictor,
        tolerance=5,
        target_size=10,
    )
    agent.run_rnd()
    target_set = np.array(agent.target_set)
    print(target_set)
    print(agent.metric_results_storage)
    # print(agent.metric_results_storage)

    fig, (plot1, plot2) = plt.subplots(2, figsize=(4, 8))
    results = np.array(agent.metric_results_storage)
    plot1.plot(results[:, 0], label="0")
    plot1.plot(results[:, 1], label="1")
    plot1.plot(results[:, 2], label="2")

    # for i, array in enumerate(agent.metric_results_storage):
    #     plot1.plot(array, label=i)
    plot1.legend()

    plot2.plot(data_generator.data_pool[:, 0], data_generator.data_pool[:, 1], ".")
    plot2.plot(target_set[:, 0], target_set[:, 1], "x", ms=8, mew=3)
    plot2.set_aspect("equal", adjustable="box")

    plt.show()
