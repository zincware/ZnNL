"""
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincware Project.

Description: An example script to compute the number of unique places in a box.
"""
import pyrnd


if __name__ == "__main__":
    """
    Main method to run the routine.
    """
    # Noisy circle.
    # data = pyrnd.PointsOnCircle(noise=0.1)
    # data.build_pool('uniform', n_points=50, noise=True)
    data_generator = pyrnd.ConfinedParticles()
    data_generator.build_pool(100)

    target = pyrnd.DenseModel(units=12, layers=4, in_d=2, out_d=12, tolerance=1e-3)
    predictor = pyrnd.DenseModel(units=12, layers=4, in_d=2, out_d=12, tolerance=1e-3)
    agent = pyrnd.RND(
        data_generator=data_generator,
        target_network=target,
        predictor_network=predictor,
        tolerance=10,
        target_size=10,
    )
    agent.run_rnd()
