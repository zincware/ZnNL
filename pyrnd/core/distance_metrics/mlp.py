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
Module to train an MLP to act as a distance metric.
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Callable

import pyrnd
from pyrnd import DataGenerator
from pyrnd import DenseModel
from typing import Union
import datetime
import shutil


class MLPMetricModel(keras.Model):
    """
    Custom model train method for the MLP metric.
    """
    def train_step(self, data):
        """
        Custom train step for the MLPMetric.

        Returns
        -------

        """
        x, y = data  # split data into tuples.
        split = int(x.shape[-1]/2)

        point_a, point_b = x[:, :split], x[:, split:]
        x_permuted = tf.concat([point_b, point_a], 1)

        c = point_a + point_b

        # Compute loss
        with tf.GradientTape() as tape:
            # Get various forward passes.
            y_pred = self(x, training=True)
            y_pred_permuted = self(x_permuted, training=True)
            z_prime = self(tf.concat([point_a, c], 1), training=True)
            z_prime_prime = self(tf.concat([c, point_b], 1), training=True)

            loss_vector = tf.convert_to_tensor(
                [y_pred, y_pred_permuted, z_prime, z_prime_prime],
                dtype=tf.float64
            )

            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, loss_vector, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


class MLPMetric:
    """
    Class for the MLP distance metric.

    Attributes
    ----------
    model : keras.Model
            Keras model to be called by the class.
    """

    def __init__(
            self,
            data_generator: DataGenerator,
            distance_metric: Callable,
            embedding_operator: DenseModel,
            training_points: int = 100,
            validation_points: int = 100,
            test_points: int = 100,
            name: str = "mlp_metric"
    ):
        """
        Constructor for the MLP metric.

        Parameters
        ----------
        data_generator : DataGenerator
                Generator from which data can be selected.
        distance_metric : Callable
                Distance metric to use on the generated data in the non-embedded space.
        embedding_operator : DenseModel
                Neural network used in the embedding of RND, i.e., the random neural
                network.
        training_points : int
                Number of training points to use in the metric development.
        validation_points : int
                Number of validation points to use in the metric development.
        test_points : int
                Number of test points to use in the metric development.
        name : str
                A name for the metric.
        """
        self.data_generator = data_generator
        self.distance_metric = distance_metric
        self.embedding_operator = embedding_operator
        self.training_points = training_points
        self.validation_points = validation_points
        self.test_points = test_points
        self.name = name

        # Built during the run.
        self.model: keras.Model = None

    def __call__(self, point_1: tf.Tensor, point_2: tf.Tensor):
        """
        Call method for the MLPMetric.

        This method allows the MLP metric to be used as a simple distance metric
        through the call method.

        Parameters
        ----------
        point_1 : tf.Tensor
                Point 1 in the distance metric.
        point_2 : tf.Tensor
                Point 2 in the distance metric.

        Returns
        -------

        """
        inputs = tf.concat([point_1, point_2], 1)

        return self.model(inputs)

    def build_model(
            self, units: tuple, activation: str, normalization: Union[None, tf.Tensor]
    ) -> tf.keras.Model:
        """
        Build a neural network model.

        Parameters
        ----------
        units : tuple
                Number of units to have in each layer.
        activation : str
                TF activate function to use in each layer.
        normalization : tf.Tensor
                If None, nothing is changed, if a Tensor is passed, a normalization
                layer is built and added to the model.

        Returns
        -------

        """
        test_point = self.embedding_operator.predict(self.data_generator.get_points(1))
        input_layer = keras.Input(shape=(test_point[0].shape[0] * 2))

        # Handle normalization.
        if normalization is None:  # no normalization layer
            x = layers.Dense(units[0], activation)(input_layer)
        else:  # add batch normalization layer.
            normalization_layer = layers.BatchNormalization(axis=-1)
            x = normalization_layer(input_layer)
            x = layers.Dense(units[0], activation)(x)

        for layer in units[0:]:
            x = layers.Dense(layer, activation)(x)

        x = layers.Dense(1, activation)(x)

        return MLPMetricModel(input_layer, x)

    def prepare_dataset(self, points: int):
        """
        Prepare a singular dataset e.g. training or validation.

        Parameters
        ----------
        points : int
                Number of points to collect.
        Returns
        -------

        """
        lhs_points = self.data_generator.get_points(points)
        rhs_points = self.data_generator.get_points(points)

        distances = self.distance_metric(point_1=lhs_points, point_2=rhs_points)

        embedded_lhs = self.embedding_operator.predict(lhs_points)
        embedded_rhs = self.embedding_operator.predict(rhs_points)

        return tf.concat([embedded_lhs, embedded_rhs], 1), distances

    def prepare_data(self):
        """
        Prepare the data for training and validation.

        Returns
        -------

        """
        training = self.prepare_dataset(self.training_points)
        validation = self.prepare_dataset(self.validation_points)
        test = self.prepare_dataset(self.test_points)

        return training, validation, test

    @staticmethod
    def clean_workspace():
        """
        Clean the current workspace.

        Remove past logs and models.

        Returns
        -------

        """
        # Handle previous log files.
        try:
            shutil.rmtree('./logs')
        except FileNotFoundError:
            pass
        except OSError:
            raise OSError("Please close the previous Tensorboard url.")

    def train_model(
            self,
            units: tuple = (12, 12, 12),
            activation: str = "relu",
            normalize: bool = False,
            epochs: int = 100
    ) -> keras.Model:
        """
        Train the distance metric.

        Parameters
        ----------
        units : tuple (default = (12, 12, 12))
                Number of units to have in each layer.
        activation : str (default = 'relu')
                Activate function to use in the layers.
        normalize : bool (default = False)
                If true, data should be normalized with a normalization layer before
                training commences.
        epochs : int
                Number of epochs to train on.

        Returns
        -------
        Saves a tf model to be used as an activate function.

        model : keras.Model
                Also returns the model in code for direct use.
        """
        self.clean_workspace()  # delete unnecessary files.

        # training[0] -> input features, training[1] -> input targets
        training, validation, test = self.prepare_data()  # collect training data.

        # Handle normalization.
        if normalize:
            normalization = training[0]
        else:
            normalization = None

        # Build and compile the model
        model = self.build_model(units, activation, normalization)
        opt = tf.keras.optimizers.Adam(learning_rate=0.01, decay=0.0)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.1, patience=15, min_lr=0.00001
        )
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50)
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                              histogram_freq=1)
        loss_fn = pyrnd.DistanceMetricLoss(alpha=1, beta=5, gamma=0.1)
        model.compile(
            optimizer=opt,
            loss=loss_fn,
            metrics=['accuracy']
        )

        # Train and evaluate the model.
        model.fit(
            x=training[0],
            y=training[1],
            validation_data=(validation[0], validation[1]),
            epochs=epochs,
            shuffle=True,
            verbose=1,
            batch_size=32,
            callbacks=[reduce_lr, early_stopping, tensorboard_callback]
        )
        evaluation = model.evaluate(x=test[0], y=test[1])
        print(evaluation)
        model.save(self.name)  # save the metric

        self.model = model
