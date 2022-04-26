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
MNIST Data generator.
"""
import jax.numpy as np
import plotly.graph_objects as go
import tensorflow_datasets as tfds
from plotly.subplots import make_subplots

from znrnd.core.data.data_generator import DataGenerator


class MNISTGenerator(DataGenerator):
    """
    Data generator for MNIST datasets
    """

    def __init__(self):
        """
        Constructor for the MNIST generator class.
        """
        self.ds_train, self.ds_test = tfds.as_numpy(
            tfds.load(
                "mnist:3.*.*",
                split=["train[:%d]" % 500, "test[:%d]" % 500],
                batch_size=-1,
            )
        )
        self.ds_train["image"] = self.ds_train["image"] / 255.0
        self.ds_test["image"] = self.ds_test["image"] / 255.0
        self.data_pool = self.ds_train["image"].astype(float)

    def plot_image(self, indices: list = None, data_list: list = None):
        """
        Plot a images from the training dataset.

        Parameters
        ----------
        indices : list (None)
                Indices of the dataset you want to plot.
        data_list : list (None)
                A list of data objects to plot.
        """
        if indices is not None:
            data_length = len(indices)
            data_source = self.ds_train["image"][indices]
        elif data_list is not None:
            data_length = len(data_list)
            data_source = data_list
        else:
            raise TypeError("No valid data provided")

        if data_length <= 4:
            columns = data_length
            rows = 1
        else:
            columns = 4
            rows = int(np.ceil(data_length / 4))

        fig = make_subplots(rows=rows, cols=columns)

        img_counter = 0
        for i in range(1, rows + 1):
            for j in range(1, columns + 1):
                if indices is not None:
                    data = self.ds_train["image"][img_counter].reshape(28, 28)
                else:
                    data = data_list[img_counter].reshape(28, 28)
                fig.add_trace(go.Heatmap(z=data), row=i, col=j)
                if img_counter == len(data_source) - 1:
                    break
                else:
                    img_counter += 1

        fig.show()
