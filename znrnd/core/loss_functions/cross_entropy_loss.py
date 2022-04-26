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
Implement a cross entropy loss function.
"""
import jax
import jax.numpy as np

from znrnd.core.loss_functions.simple_loss import SimpleLoss


class CrossEntropyDistance:
    """
    Class for the cross entropy distance
    """

    def __init__(self, classes: int):
        """
        Constructor for the distance

        Parameters
        ----------
        classes : int
                Number of classes in the one-hot encoding.
        """
        self.classes = classes

    def __call__(self, prediction, target):
        """

        Parameters
        ----------
        prediction
        target

        Returns
        -------

        """
        one_hot_labels = jax.nn.one_hot(target, num_classes=self.classes)

        return -1 * np.sum(prediction * one_hot_labels, axis=-1)


class CrossEntropyLoss(SimpleLoss):
    """
    Class for the cross entropy loss
    """

    def __init__(self, classes: int = 10):
        """
        Constructor for the mean power loss class.

        Parameters
        ----------
        classes : int (default=10)
                Number of classes in the loss.
        """
        super(CrossEntropyLoss, self).__init__()
        self.metric = CrossEntropyDistance(classes=classes)
