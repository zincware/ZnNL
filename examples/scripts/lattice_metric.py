"""
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincware Project.

Description: Demonstrating learning a metric for lattice.
"""
import pyrnd
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    """
    Main method to run the routine.
    """
    # Prepare the data generator.
    data_generator = pyrnd.PointsOnLattice()
    data_generator.build_pool(x_points=10, y_points=10)

    # Build the target and predictor models.
    target = pyrnd.DenseModel(
        units=12, layers=4, in_d=2, out_d=12, tolerance=1e-5, loss="cosine_similarity"
    )
    predictor = pyrnd.DenseModel(
        units=12, layers=4, in_d=2, out_d=12, tolerance=1e-5, loss="cosine_similarity"
    )

    # Create and train the lattice metric for the problem.
    lattice_metric = pyrnd.MLPMetric(
        data_generator=data_generator,
        distance_metric=pyrnd.distance_metrics.euclidean_distance,
        embedding_operator=target,
        training_points=100,
        validation_points=100,
        test_points=100,
        name='lattice_metric'
    )
    lattice_metric.train_model(
        units=(15, 15), activation='relu', normalize=False, epochs=500
    )

    # Define and run the RND agent.
    agent = pyrnd.RND(
        point_selector=pyrnd.GreedySelection(threshold=1),
        distance_metric=lattice_metric,
        data_generator=data_generator,
        target_network=target,
        predictor_network=predictor,
        tolerance=5,
        target_size=10,
    )
    agent.run_rnd()
    print(agent.target_set)

    plt.plot(
        data_generator.data_pool[:, 0],
        data_generator.data_pool[:, 1],
        '.',
        label='Data Pool'
    )
    plt.plot(
        np.array(agent.target_set)[:, 0],
        np.array(agent.target_set)[:, 1],
        'x',
        label='Chosen Points'
    )
    plt.show()
