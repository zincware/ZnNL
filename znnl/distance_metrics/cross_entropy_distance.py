"""
ZnNL: A Zincwarecode package.

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
"""
import optax

from znnl.distance_metrics.distance_metric import DistanceMetric


class CrossEntropyDistance(DistanceMetric):
    """
    Class for the cross entropy distance
    """

    def __call__(self, prediction, target):
        """

        Parameters
        ----------
        prediction (batch_size, n_classes)
        target

        Returns
        -------
        Softmax cross entropy of the batch.

        """
        return optax.softmax_cross_entropy(logits=prediction, labels=target)
