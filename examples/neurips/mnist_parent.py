"""
ZnRND: A zincwarecode package.

License
-------
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the zincwarecode Project.

Contact Information
-------------------
email: zincwarecode@gmail.com
github: https://github.com/zincware
web: https://zincwarecode.com/

Citation
--------
If you use this module please cite us with:

Summary
-------
Parent class for the MNIST experiment
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


import numpy as np
import optax
from flax import linen as nn
from neural_tangents import stax

import znrnd as rnd

import time

set_size = DS_SIZE # -- Set for linting

data_generator = rnd.data.MNISTGenerator()

model = stax.serial(
    stax.Conv(32, (3, 3)),
    stax.Relu(),
    stax.AvgPool(window_shape=(2, 2), strides=(2, 2)),
    stax.Conv(64, (3, 3)),
    stax.Relu(),
    stax.AvgPool(window_shape=(2, 2), strides=(2, 2)),
    stax.Flatten(),
    stax.Dense(256),
)
model1 = stax.serial(
    stax.Conv(32, (3, 3)),
    stax.Relu(),
    stax.AvgPool((2, 2), (2, 2)),
    stax.Conv(64, (3, 3)),
    stax.Relu(),
    stax.AvgPool((2, 2), (2, 2)),
    stax.Flatten(),
    stax.Dense(256),
)


class CustomModule(nn.Module):
    """
    Simple CNN module.
    """

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        x = nn.log_softmax(x)

        return x


def run_experiment(data_set_size: int, ensembling: bool = False, ensembles: int = 10):
    """
    Run an experiment for a specific datasrndsize.

    Parameters
    ----------
    data_set_size : int
            Size of the dataset to produce
    ensembling : bool (default=False)
            If true, the experiment is run several times to produce an error estimate
    ensembles : int
            Number of ensembles to use in the averaging.

    Returns
    -------
    entropy : dict
            A dictionary of the computed entropy:
            e.g {"rnd": 0.68, "random": 0.41, "approximate_maximum": 0.84}
    eigenvalues : dict
            Dictionary of eigenvalues
            e.g {"rnd": np.array(), "random": np.array(), "approximate_maximum": np.array()}

    """
    # Turnoff averaging if required.
    if not ensembling:
        ensembles = 1

    rnd_entropy_arr = []
    random_entropy_arr = []
    apr_max_entropy_arr = []

    rnd_eig_arr = []
    random_eig_arr = []
    apr_max_eig_arr = []

    rnd_losses = []
    random_losses = []
    apr_max_losses = []

    for i in range(ensembles):
        # Define the models
        target = rnd.models.NTModel(
            nt_module=model,
            optimizer=optax.adam(0.01),
            loss_fn=rnd.loss_functions.MeanPowerLoss(order=2),
            input_shape=(1, 28, 28, 1),
            training_threshold=0.01,
        )

        predictor = rnd.models.NTModel(
            nt_module=model1,
            optimizer=optax.adam(0.01),
            loss_fn=rnd.loss_functions.MeanPowerLoss(order=2),
            input_shape=(1, 28, 28, 1),
            training_threshold=0.01,
        )

        # Define the agents for a fresh run.
        rnd_agent = rnd.agents.RND(
            point_selector=rnd.point_selection.GreedySelection(threshold=0.01),
            distance_metric=rnd.distance_metrics.OrderNDifference(order=2),
            data_generator=data_generator,
            target_network=target,
            predictor_network=predictor,
            tolerance=15,
        )
        rnd_agent.target_set = []
        rnd_agent.target_indices = []

        random_agent = rnd.agents.RandomAgent(data_generator=data_generator)
        approximate_max_agent = rnd.agents.ApproximateMaximumEntropy(
            target_network=target,
            data_generator=data_generator,
            samples=20,
            # How many sets it produces in the test. Takes the one with max entropy.
        )

        # Compute the sets
        rnd_set = rnd_agent.build_dataset(target_size=data_set_size, visualize=False)
        random_set = random_agent.build_dataset(
            target_size=data_set_size, visualize=False
        )
        start = time.time()
        apr_max_set = approximate_max_agent.build_dataset(
            target_size=data_set_size, visualize=False
        )
        print(f"Apr max set: {time.time() - start}")
        start = time.time()
        # Compute NTK for each set
        rnd_ntk = target.compute_ntk(x_i=rnd_set)["empirical"]
        random_ntk = target.compute_ntk(x_i=random_set)["empirical"]
        apr_max_ntk = target.compute_ntk(x_i=apr_max_set)["empirical"]

        print(f"NTK Time: {time.time() - start}")

        # Compute the entropy of each set
        rnd_entropy = rnd.analysis.EntropyAnalysis(
            matrix=rnd_ntk
        ).compute_von_neumann_entropy()
        random_entropy = rnd.analysis.EntropyAnalysis(
            matrix=random_ntk
        ).compute_von_neumann_entropy()
        apr_max_entropy = rnd.analysis.EntropyAnalysis(
            matrix=apr_max_ntk
        ).compute_von_neumann_entropy()

        # Compute eigenvalues
        rnd_eigval = rnd.analysis.EigenSpaceAnalysis(
            matrix=rnd_ntk
        ).compute_eigenvalues()
        random_eigval = rnd.analysis.EigenSpaceAnalysis(
            matrix=random_ntk
        ).compute_eigenvalues()
        apr_max_eigval = rnd.analysis.EigenSpaceAnalysis(
            matrix=rnd_ntk
        ).compute_eigenvalues()

        rnd_entropy_arr.append(rnd_entropy)
        random_entropy_arr.append(random_entropy)
        apr_max_entropy_arr.append(apr_max_entropy)

        rnd_eig_arr.append(rnd_eigval)
        random_eig_arr.append(random_eigval)
        apr_max_eig_arr.append(apr_max_eigval)

        # Train production model
        rnd_production = rnd.models.FlaxModel(
            flax_module=CustomModule(),
            optimizer=optax.adam(learning_rate=0.1),
            loss_fn=rnd.loss_functions.CrossEntropyLoss(classes=10),
            input_shape=(1, 28, 28, 1),
            training_threshold=0.001,
        )

        random_production = rnd.models.FlaxModel(
            flax_module=CustomModule(),
            optimizer=optax.adam(learning_rate=0.1),
            loss_fn=rnd.loss_functions.CrossEntropyLoss(classes=10),
            input_shape=(1, 28, 28, 1),
            training_threshold=0.001,
        )

        apr_max_production = rnd.models.FlaxModel(
            flax_module=CustomModule(),
            optimizer=optax.adam(learning_rate=0.1),
            loss_fn=rnd.loss_functions.CrossEntropyLoss(classes=10),
            input_shape=(1, 28, 28, 1),
            training_threshold=0.001,
        )

        rnd_training_ds = {
            "inputs": np.take(
                data_generator.ds_train["image"], rnd_agent.target_indices, axis=0
            ),
            "targets": np.take(
                data_generator.ds_train["label"], rnd_agent.target_indices, axis=0
            ),
        }
        random_training_ds = {
            "inputs": np.take(
                data_generator.ds_train["image"], random_agent.target_indices, axis=0
            ),
            "targets": np.take(
                data_generator.ds_train["label"], random_agent.target_indices, axis=0
            ),
        }
        apr_max_training_ds = {
            "inputs": np.take(
                data_generator.ds_train["image"],
                approximate_max_agent.target_indices,
                axis=0,
            ),
            "targets": np.take(
                data_generator.ds_train["label"],
                approximate_max_agent.target_indices,
                axis=0,
            ),
        }

        test_ds = {
            "inputs": data_generator.ds_test["image"],
            "targets": data_generator.ds_test["label"],
        }

        rnd_losses.append(
            rnd_production.train_model(
                train_ds=rnd_training_ds, test_ds=test_ds, epochs=100, batch_size=10
            )
        )
        random_losses.append(
            random_production.train_model(
                train_ds=random_training_ds, test_ds=test_ds, epochs=100, batch_size=10
            )
        )
        apr_max_losses.append(
            apr_max_production.train_model(
                train_ds=apr_max_training_ds, test_ds=test_ds, epochs=100, batch_size=10
            )
        )

        del rnd_agent
        del random_agent
        del approximate_max_agent

    # Get mean and uncertainty.
    rnd_entropy_arr = np.array(rnd_entropy_arr)
    random_entropy_arr = np.array(random_entropy_arr)
    apr_max_entropy_arr = np.array(apr_max_entropy_arr)

    rnd_eig_arr = np.array(rnd_eig_arr)
    random_eig_arr = np.array(random_eig_arr)
    apr_max_eig_arr = np.array(apr_max_eig_arr)

    rnd_losses = np.array(rnd_losses)
    random_losses = np.array(random_losses)
    apr_max_losses = np.array(apr_max_losses)

    rnd_entropy = np.array(
        [np.mean(rnd_entropy_arr), np.std(rnd_entropy_arr) / np.sqrt(ensembles)]
    )
    random_entropy = np.array(
        [np.mean(random_entropy_arr), np.std(random_entropy_arr) / np.sqrt(ensembles)]
    )
    apr_max_entropy = np.array(
        [np.mean(apr_max_entropy_arr), np.std(apr_max_entropy_arr) / np.sqrt(ensembles)]
    )

    rnd_eigval = np.array(
        [np.mean(rnd_eig_arr, axis=0), np.std(rnd_eig_arr, axis=0) / np.sqrt(ensembles)]
    )
    random_eigval = np.array(
        [
            np.mean(random_eig_arr, axis=0),
            np.std(random_eig_arr, axis=0) / np.sqrt(ensembles),
        ]
    )
    apr_max_eigval = np.array(
        [
            np.mean(apr_max_eig_arr, axis=0),
            np.std(apr_max_eig_arr, axis=0) / np.sqrt(ensembles),
        ]
    )

    rnd_loss = np.array(
        [np.mean(rnd_losses, axis=0), np.std(rnd_losses, axis=0) / np.sqrt(ensembles)]
    )
    random_loss = np.array(
        [
            np.mean(random_losses, axis=0),
            np.std(random_losses, axis=0) / np.sqrt(ensembles),
        ]
    )
    apr_max_loss = np.array(
        [
            np.mean(apr_max_losses, axis=0),
            np.std(apr_max_losses, axis=0) / np.sqrt(ensembles),
        ]
    )

    entropy = {
        "rnd": rnd_entropy,
        "random": random_entropy,
        "approximate_maximum": apr_max_entropy,
    }
    eigenvalues = {
        "rnd": rnd_eigval,
        "random": random_eigval,
        "approximate_maximum": apr_max_eigval,
    }
    losses = {
        "rnd": rnd_loss,
        "random": random_loss,
        "approximate_maximum": apr_max_loss,
    }

    return entropy, eigenvalues, losses


entropy, eigenvalues, loss = run_experiment(
    data_set_size=set_size, ensembling=True, ensembles=3
)

np.save(f"entropy_{set_size}.npy", entropy)
np.save(f"eigenvalues_{set_size}", eigenvalues)
np.save(f"losses_{set_size}", loss)
