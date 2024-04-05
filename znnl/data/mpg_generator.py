"""
Data generator for the MPG data.
"""

import pandas as pd

from znnl.data.data_generator import DataGenerator


class MPGDataGenerator(DataGenerator):
    """
    MPG data generator.
    """

    def __init__(self, train_fraction: float):
        """
        Constructor for the MPG data generator.

        Parameters
        ----------
        train_fraction : float
            Number of points to use in the train and test set.
        """
        dataset = self._download_data()

        train_ds = dataset.sample(frac=train_fraction, random_state=0)
        train_labels = train_ds.pop("MPG")

        test_ds = dataset.drop(train_ds.index)
        test_labels = test_ds.pop("MPG")

        self.train_ds = {
            "inputs": train_ds.to_numpy(),
            "targets": train_labels.to_numpy().reshape(-1, 1),
        }
        self.test_ds = {
            "inputs": test_ds.to_numpy(),
            "targets": test_labels.to_numpy().reshape(-1, 1),
        }

        self.data_pool = self.train_ds["inputs"]

    def _download_data(self):
        """
        Download the data from the UCI repository.

        This method will also normalize the data for use in the
        neural network.
        """
        base_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/"
        dataset = "auto-mpg/auto-mpg.data"
        url = f"{base_url}{dataset}"
        column_names = [
            "MPG",
            "Cylinders",
            "Displacement",
            "Horsepower",
            "Weight",
            "Acceleration",
            "Model Year",
            "Origin",
        ]

        raw_dataset = pd.read_csv(
            url,
            names=column_names,
            na_values="?",
            comment="\t",
            sep=" ",
            skipinitialspace=True,
        )

        dataset = raw_dataset.copy()
        dataset = dataset.dropna()
        dataset["Origin"] = dataset["Origin"].map({1: "USA", 2: "Europe", 3: "Japan"})
        dataset = pd.get_dummies(dataset, columns=["Origin"], prefix="", prefix_sep="")

        return (dataset - dataset.mean()) / dataset.std()
