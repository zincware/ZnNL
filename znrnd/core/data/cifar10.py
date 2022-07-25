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
Data generator for the CIFAR 10 dataset.
"""
import jax.numpy as np
import plotly.graph_objects as go
import tensorflow_datasets as tfds
from plotly.subplots import make_subplots

from znrnd.core.data.data_generator import DataGenerator


class CIFAR10Generator(DataGenerator):
    """
    Data generator for MNIST datasets
    """

    def __init__(self, ds_size: int = 500):
        """
        Constructor for the MNIST generator class.

        Parameters
        ----------
        ds_size : int
                Size of the dataset to load.
        """
        self.train_ds, self.test_ds = tfds.as_numpy(
            tfds.load(
                "cifar10:3.*.*",
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
                    data = self.train_ds["inputs"][img_counter] * 255.0
                else:
                    data = data_list[img_counter] * 255.0
                fig.add_trace(go.Image(z=data), row=i, col=j)
                if img_counter == len(data_source) - 1:
                    break
                else:
                    img_counter += 1

        fig.show()
