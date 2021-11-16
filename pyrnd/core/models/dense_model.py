"""
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincware Project.

Description: Module for a standard feed forward neural network.
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pyrnd.core.models.model import Model
from pyrnd.core.similarity_measures import SimilarityMeasures
from pyrnd.core.similarity_measures import CosineSim


class DenseModel(Model):
    """
    Class for the feed forward network implementation.

    Attributes
    ----------
    units : tuple
            Number of units to have per layer in the NN.
    in_d : int
            Input dimension of the data.
    out_d : int
            Output dimension of the representation.
    activation : str
            Activation function to use in the training.
    learning_rate : float
            Learning rate for the network.
    tolerance : float
            Minimum value of the loss before the model is considered trained.
    loss : str
            Loss to use during the training.
    model : tf.Model
            Machine learning model to be trained.
    """

    def __init__(
        self,
        units: tuple = (12, 12, 12),
        in_d: int = 2,
        out_d: int = 1,
        activation: str = "relu",
        learning_rate: float = 1e-2,
        tolerance: float = 1e-5,
        loss: SimilarityMeasures = CosineSim(),
    ):
        """
        Constructor for the Feed forward network module.

        Parameters
        ----------
        units : tuple
                Number of units to have per layer in the NN.
        in_d : int
                Input dimension of the data.
        out_d : int
                Output dimension of the representation.
        activation : str
                Activation function to use in the training.
        learning_rate : float
                Learning rate for the network.
        tolerance : float
                Minimum value of the loss before the model is considered
                trained.
        loss : SimilarityMeasures
                Loss to use during the training.
        """
        super().__init__()  # update parent.
        # User arguments
        self.units = units
        self.in_d = in_d
        self.out_d = out_d
        self.activation = activation
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        self.loss = loss

        # Model parameters
        self._build_model()  # build the model immediately

    def _build_model(self):
        """
        Build the NN model such that it can be trained.

        Returns
        -------

        """

        # Add the input layer
        input_layer = keras.Input(shape=(self.in_d,))
        x = layers.Dense(self.units[0], activation=self.activation)(input_layer)

        # Add the hidden layers
        for unit in self.units[1:]:
            x = layers.Dense(unit, self.activation)(x)

        # Add the output layer
        output_layer = layers.Dense(self.out_d, self.activation)(x)

        self.model = keras.Model(input_layer, output_layer)

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

        self.model.compile(optimizer=opt, loss=self.loss)

    def _evaluate_model(self, x, y):
        """
        Evaluate the model.

        Parameters
        ----------
        dataset : tf.data.Dataset
                Dataset on which to assess the model.

        Returns
        -------

        """
        loss = self.model.evaluate(x, y)

        return loss <= self.tolerance

    def _lr_reduction(self, counter: int):
        """
        Perform learning rate reduction.

        Parameters
        ----------
        counter : int
                Counter determining whether or not to perform the reduction.

        Returns
        -------
        Changes the learning rate and re-compiles the model.
        """
        if counter % 10 == 0:
            self.learning_rate = 0.8 * self.learning_rate
            self._compile_model()

    def _model_rebuild(self, counter: int):
        """
        Perform learning rate reduction.

        Parameters
        ----------
        counter : int
                Counter determining whether or not to perform rebuild.

        Returns
        -------
        Rebuilds and re-compiles the model.
        """
        if counter % 10 == 0:
            print("Model re-build triggered.")
            self._build_model()
            self._compile_model()

    def predict(self, point: tf.Tensor):
        """
        Make a prediction on a point.

        Parameters
        ----------
        point : tf.Tensor
                Point(s) on which to perform a prediction.

        Returns
        -------
        prediction : tf.Tensor
                Model prediction on the point.
        """
        return self.model.predict(point)

    def train_model(
        self,
        x: tf.Tensor,
        y: tf.Tensor,
        re_initialize: bool = False,
        epochs: int = 30,
    ):
        """
        Train the model on data.

        Parameters
        ----------
        y
        x
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
            self._build_model()

        self._compile_model()  # compile the network.
        converged = False  # set the converged flag.

        counter = 1
        while converged is False:
            # verbose=0 only shows the final result, not single epochs
            self.model.fit(
                x=x, y=y, epochs=epochs, shuffle=True, verbose=0, batch_size=32
            )
            converged = self._evaluate_model(x, y)

            self._lr_reduction(counter)
            self._model_rebuild(counter)
            counter += 1
