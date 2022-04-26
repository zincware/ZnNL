"""
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincware Project.

Description: Demonstrating learning a metric for lattice.
"""
import matplotlib.pyplot as plt
import numpy as np

import znrnd

if __name__ == "__main__":
    """
    Main method to run the routine.
    """
    # Prepare the data generator.
    data_generator = znrnd.data.PointsOnLattice()
    data_generator.build_pool(x_points=10, y_points=10)

    target = znrnd.models.DenseModel(
        units=(12, 12, 12),
        in_d=2,
        out_d=12,
        tolerance=1.0,
        loss=znrnd.loss_functions.MeanPowerLoss(order=2),
    )
    predictor = znrnd.models.DenseModel(
        units=(12, 12, 12),
        in_d=2,
        out_d=12,
        tolerance=1.0,
        loss=znrnd.loss_functions.MeanPowerLoss(order=2),
    )

    # Create and train the lattice metric for the problem.
    lattice_metric = znrnd.distance_metrics.MLPMetric(
        data_generator=data_generator,
        distance_metric=znrnd.distance_metrics.LPNorm(order=2),
        embedding_operator=target,
        training_points=100,
        validation_points=100,
        test_points=100,
        name="lattice_metric",
    )
    lattice_metric.train_model(
        units=(15, 15), activation="sigmoid", normalize=False, epochs=500
    )

    target.loss = lattice_metric
    predictor.loss = lattice_metric

    # Define and run the RND agent.
    agent = znrnd.RND(
        point_selector=znrnd.point_selection.GreedySelection(threshold=1),
        distance_metric=lattice_metric,
        data_generator=data_generator,
        target_network=target,
        predictor_network=predictor,
        tolerance=5,
        target_size=10,
    )
    agent.build_dataset()

    plt.plot(
        data_generator.data_pool[:, 0],
        data_generator.data_pool[:, 1],
        ".",
        label="Data Pool",
    )
    plt.plot(
        np.array(agent.target_set)[:, 0],
        np.array(agent.target_set)[:, 1],
        "x",
        label="Chosen Points",
    )
    plt.legend()
    plt.show()
