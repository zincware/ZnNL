"""
ZnRND: A Zincwarecode package.

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
TSNE visualizer.
"""
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


class TSNEVisualizer:
    """
    Perform visualization on network representations with TSNE.
    """

    def __init__(self, components: int = 2):
        """
        Constructor for the TSNE visualizer.

        Parameters
        ----------
        components : int
                Number of components to use in the representation. Either 2 or 3.
        """
        self.model = TSNE(n_components=components, init="random")

        self.reference = []
        self.dynamic = []

    def build_representation(self, data: np.ndarray, reference: bool = False):
        """
        Construct a TSNE representation.

        Parameters
        ----------
        data : np.ndarray (n_coordinates, dimension)
                Data on which the representation should be constructed.
        reference : bool
                If true, populate the reference attribute.

        Returns
        -------
        stores a plot
        """
        if reference:
            self.reference.append(self.model.fit_transform(data))
        else:
            self.dynamic.append(self.model.fit_transform(data))

    def data_gen(self):
        """
        Generate next data point for the animation.

        Returns
        -------

        """
        for count in self.dynamic:
            yield count[:, 0], count[:, 1]

    def run_visualization(self):
        """
        Run the visualization.
        """
        # create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2)

        # initialize two line objects (one in each axes)
        ax1.plot(self.reference[0][:, 0], self.reference[0][:, 1], ".")
        ax2.plot(self.dynamic[-1][:, 0], self.dynamic[-1][:, 1], ".")

        plt.show()
