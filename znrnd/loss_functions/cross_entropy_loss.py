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
import optax

from znrnd.loss_functions.simple_loss import SimpleLoss


class CrossEntropyDistance:
    """
    Class for the cross entropy distance
    """

    def __init__(self, apply_softmax: bool = False):
        """
        Constructor for the distance

        Parameters
        ----------
        apply_softmax : bool (default = False)
                If true, softmax is applied to the prediction before computing the loss.
        """
        self.apply_softmax = apply_softmax

    def __call__(self, prediction, target):
        """

        Parameters
        ----------
        prediction (batch_size, n_classes)
        target

        Returns
        -------

        """
        if self.apply_softmax:
            prediction = jax.nn.softmax(prediction)
        return optax.softmax_cross_entropy(logits=prediction, labels=target)


class CrossEntropyLoss(SimpleLoss):
    """
    Class for the cross entropy loss
    """

    def __init__(self, apply_softmax: bool = False):
        """
        Constructor for the mean power loss class.

        Parameters
        ----------
        apply_softmax : bool (default = False)
                If true, softmax is applied to the prediction before computing the loss.
        """
        super(CrossEntropyLoss, self).__init__()
        self.metric = CrossEntropyDistance(apply_softmax=apply_softmax)
