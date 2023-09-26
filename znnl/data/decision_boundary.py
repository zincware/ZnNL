"""
ZnNL: A Zincwarecode package.

License
-------
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincwarecode Project.

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
Data generator for decision boundary problems.
"""
from znnl.data.data_generator import DataGenerator
import jax
import jax.numpy as np
import numpy as onp
from znnl.utils.prng import PRNGKey
import matplotlib.pyplot as plt


def linear_boundary(
    data: onp.ndarray, gradient: float, intercept: float
) -> np.ndarray:
    """
    Create a linear boundary between classes.
    
    Parameters
    ----------
    data : np.ndarray (n_samples, 2)
            Data to be converted into classes.
    gradient : float
            Gradient of the line, default 1.0.
    intercept : float
            Intercept of the line, default 1.0.
    """
    # y = m * x + c
    reference_values = gradient * data[:, 0] + intercept
    
    differences = data[:, 1] - reference_values

    differences[differences > 0] = 1
    differences[differences < 0.] = 0
    
    return differences

def circle(
    data: onp.ndarray, radius: float = 0.25
):
    """
    Create a circular classification problem.
    
    For simplicity, assume the points inside the circle are
    class 0 and outside are class 1.
    
    Parameters
    ----------
    data : np.ndarray
        Data to be converted into classes.
    radius : float
        Radius of the circle.
    """
    radii = onp.linalg.norm(data - 0.5, axis=1)
    
    radii[radii < radius] = 0.
    radii[radii > radius] = 1.
    
    return radii


class DecisionBoundaryGenerator(DataGenerator):
    """
    A class to generate data for decision boundary problems.
    """

    def __init__(
            self, 
            n_samples: int, 
            discriminator: str = "line", 
            one_hot: bool = True,
            gradient: float = 1.0,
            y_intercept: float = 0.0,
            radius: float = 0.25,
            seed: int = None
        ):
        """
        Instantiate the class.
       
        Parameters
        ----------
        n_samples : int
                Number of samples to generate per class.
        discriminator : str
                String to define the discriminator to use. 
                Options are "line" and "circle".
        one_hot : bool
                Whether to use one-hot encoding for the classes.
        gradient : float
                Gradient of the line, default 1.0.
        y_intercept : float
                Intercept of the line, default 1.0.
        radius : float
                Radius of the circle, default 0.25.
        seed : int
                Random seed for the RNG. Uses a random int if not specified.
        """
        self.rng = PRNGKey(seed=seed)
        self.one_hot = one_hot

        if discriminator == "line":
            self.discriminator = linear_boundary
            self.args = (gradient, y_intercept)
        elif discriminator == "circle":
            self.discriminator = circle
            self.args = (radius,)
        else:
            raise ValueError("Discriminator not recognised.")
        
        self.train_ds = self._build_dataset(n_samples=n_samples)
        self.test_ds = self._build_dataset(n_samples=n_samples)
        
    def _build_dataset(self, n_samples: int):
        """
        Helper method to create datasets quickly.

        Parameters
        ----------
        n_samples : int
                Number of samples to generate per class.
        """
        # Create the data-sets
        data = onp.array(jax.random.uniform(
            self.rng(), minval=0., maxval=1., shape=(n_samples, 2)
        ))
        data = onp.clip(data, 0., 1.)
        targets = self.discriminator(data, *self.args)  # build classes (0, 1)
        
        class_one_indices = np.where(
            targets == 0
        )[0]

        class_two_indices = np.where(
            targets == 1
        )[0]

        indices = np.hstack((class_one_indices, class_two_indices))
        indices = jax.random.shuffle(self.rng(), indices)

        if self.one_hot:
            targets = np.array(jax.nn.one_hot(targets, num_classes=2))
        else:
            targets = targets.reshape(-1, 1)
            
        return {
            "inputs": np.take(data, indices, axis=0),
            "targets": np.take(targets, indices, axis=0)
        }
    
    def plot(self):
        """
        Plot the training and test datasets.
        """
        fig, ax = plt.subplots(1, 2)

        # Traing data plots.
        ax[0].scatter(
            self.train_ds["inputs"][:, 0],
            self.train_ds["inputs"][:, 1],
            c=self.train_ds["targets"][:, 0]
        )
        ax[0].set_title("Training Data")
        ax[0].set_xlabel("x")
        ax[0].set_ylabel("y")

        # Test data plots.
        ax[1].scatter(
            self.test_ds["inputs"][:, 0],
            self.test_ds["inputs"][:, 1],
            c=self.test_ds["targets"][:, 0]
        )
        ax[1].set_title("Test Data")
        ax[1].set_xlabel("x")
        ax[1].set_ylabel("y")

        plt.show()



