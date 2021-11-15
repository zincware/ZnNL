"""
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincware Project.

Description: Demonstrating learning a metric for lattice.
"""
import pyrnd
import tensorflow as tf


if __name__ == "__main__":
    """
    Main method to run the routine.
    """
    data_generator = pyrnd.PointsOnLattice()
    data_generator.build_pool(x_points=10, y_points=10)

    points_1 = tf.convert_to_tensor(data_generator.get_points(15), dtype=tf.float64)
    points_2 = tf.convert_to_tensor(data_generator.get_points(15), dtype=tf.float64)

    print(points_1)
    print(points_2)

    print(pyrnd.distance_metrics.euclidean_distance(points_1, points_2))

    # target = pyrnd.DenseModel(
    #     units=12, layers=4, in_d=2, out_d=12, tolerance=1e-5, loss="cosine_similarity"
    # )
    # predictor = pyrnd.DenseModel(
    #     units=12, layers=4, in_d=2, out_d=12, tolerance=1e-5, loss="cosine_similarity"
    # )
    # # print(target.summary())
    #
    # agent = pyrnd.RND(
    #     point_selector=GreedySelection(threshold=0.1),
    #     # distance_metric=euclidean_distance,
    #     data_generator=data_generator,
    #     target_network=target,
    #     predictor_network=predictor,
    #     tolerance=5,
    #     target_size=10,
    # )
    # agent.run_rnd()
