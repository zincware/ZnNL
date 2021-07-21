"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincware Project.

Description: Module for a standard feed forward neural network.
"""
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense


class DenseModel:
    """
    Class for the feed forward network implementation.
    """

    def __init__(self,
                 units: int = 12,
                 layers: int = 4,
                 in_d: int = 2,
                 out_d: int = 1,
                 activation: str = "relu",
                 learning_rate: float = 1e-2):
        """
        Constructor for the Feed forward network module.

        Parameters
        ----------
        units : int
                Number of units to have in the NN.
        layers : int
                Number of hidden layers to have of size units.
        in_d : int
                Input dimension of the data.
        out_d : int
                Output dimension of the representation.
        activation : str
                Activation function to use in the training.
        learning_rate : float
                Learning rate for the network.
        """
        # User arguments
        self.units = units
        self.layers = layers
        self.in_d = in_d
        self.out_d = out_d
        self.activation = activation
        self.learning_rate = learning_rate

        # Model parameters
        self.model = tf.keras.Sequential()
        self._build_model()  # build the model immediately.

    def _build_model(self):
        """
        Build the NN model such that it can be trained.

        Returns
        -------

        """
        # Add the input layer
        input_layer = Input(shape=[self.in_d])
        self.model.add(input_layer)

        # Add the hidden layers
        for i in range(self.layers):
            self.model.add(Dense(self.units, self.activation))

        # Add the output layer
        self.model.add(Dense(self.out_d, self.activation))

    def summary(self):
        """
        Print a model summary.

        Returns
        -------
        Print the model to the screen.
        """
        print(self.model.summary())

    def _compile_model(self):
        """
        Compile the neural network model

        Returns
        -------

        Notes
        -----
        TODO: Add options for the loss function. Make some nice classes.

        """
        opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, decay=0.0)
        self.model.compile(optimizer=opt, loss="cosine_similarity")

    def _evaluate_model(self, dataset: tf.data.Dataset):
        """
        Evaluate the model.

        Parameters
        ----------
        dataset : tf.data.Dataset
                Dataset on which to assess the model.

        Returns
        -------

        """



    def train_model(self,
                    dataset: tf.data.Dataset,
                    re_initialize: bool = False,
                    epochs: int = 10):
        """
        Train the model on data.

        Parameters
        ----------
        dataset : tf.data.Dataset
                Dataset on which to train the model.
        re_initialize : bool
                If true, the network should be re-built and compiled.
        epochs : int
                Number epochs to train with on each batch.
        Returns
        -------
        Trains the model.

        Notes
        -----
        TODO: Adjust batch size depending on data availability.
        """
        if re_initialize:
            self.model = tf.keras.Sequential()
            self._build_model()

        self._compile_model()  # compile the network.
        converged = False  # set the converged flag.

        while converged is False:
            self.model.fit(dataset,
                           batch_size=5,
                           epochs=epochs,
                           shuffle=True,
                           verbose=0)



