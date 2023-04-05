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
import jax.nn as nn
import jax.numpy as np
import plotly.graph_objects as go
import tensorflow_datasets as tfds
from plotly.subplots import make_subplots

from znnl.data.data_generator import DataGenerator


class MNISTGenerator(DataGenerator):
    """
    Data generator for MNIST datasets
    """

    def __init__(self, ds_size: int = 500, one_hot_encoding: bool = True):
        """
        Constructor for the MNIST generator class.

        Parameters
        ----------
        ds_size : int (default = 500)
                Number of points to download in the train and test set.
        one_hot_encoding : bool (default = True)
                If True, the targets will be one-hot encoded.
        """
        self.train_ds, self.test_ds = tfds.as_numpy(
            tfds.load(
                "mnist:3.*.*",
                split=["train[:%d]" % ds_size, "test[:%d]" % ds_size],
                batch_size=-1,
            )
        )
        self.train_ds["inputs"] = self.train_ds["image"] / 255.0
        self.test_ds["inputs"] = self.test_ds["image"] / 255.0
        self.train_ds["targets"] = self.train_ds["label"]
        self.test_ds["targets"] = self.test_ds["label"]
        self.train_ds.pop("image")
        self.train_ds.pop("label")
        self.test_ds.pop("image")
        self.test_ds.pop("label")
        self.data_pool = self.train_ds["inputs"].astype(float)
        if one_hot_encoding:
            self.train_ds["targets"] = nn.one_hot(
                self.train_ds["targets"], num_classes=10
            )
            self.test_ds["targets"] = nn.one_hot(
                self.test_ds["targets"], num_classes=10
            )

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
            data_source = self.train_ds["inputs"][indices]
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
                    data = self.train_ds["inputs"][img_counter].reshape(28, 28)
                else:
                    data = data_list[img_counter].reshape(28, 28)
                fig.add_trace(go.Heatmap(z=data), row=i, col=j)
                if img_counter == len(data_source) - 1:
                    break
                else:
                    img_counter += 1

        fig.show()
