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
Module for using a Flax model from Hugging Face in ZnNL.
"""

import logging
from typing import Any, Callable

import jax.numpy as np
from transformers import FlaxPreTrainedModel

from znnl.models.jax_model import JaxModel

logger = logging.getLogger(__name__)


class HuggingFaceFlaxModel(JaxModel):
    """
    Class for a Hugging Face (HF) flax model.
    """

    def __init__(
        self,
        pre_built_model: FlaxPreTrainedModel,
        optimizer: Callable,
    ):
        """
        Constructor of a HF flax model.

        Parameters
        ----------
        pre_built_model : FlaxPreTrainedModel or subclass
                Pre-built model from Hugging Face.
        optimizer : Callable
                optimizer to use in the training. OpTax is used by default and
                cross-compatibility is not assured.
        input_shape : tuple
                Shape of the NN input.
        """
        logger.info(
            "Flax models have occasionally experienced memory allocation issues on "
            "GPU. This is an ongoing bug that we are striving to fix soon."
        )

        self.apply_fn = pre_built_model.__call__
        self.module = pre_built_model.module
        # self.config = pre_built_model.config

        # Save input parameters, call self.init_model
        super().__init__(
            pre_built_model=pre_built_model,
            optimizer=optimizer,
        )

    def ntk_apply_fn(self, params: dict, inputs: np.ndarray):
        """
        Return an NTK capable apply function.

        Parameters
        ----------
        params : dict
                Network parameters to use in the calculation.
        x : np.ndarray
                Data on which to apply the network

        Returns
        -------
        Acts on the data with the model architecture and parameter set.
        """
        pixel_values = np.transpose(inputs, (0, 2, 3, 1))

        # Handle any PRNG if needed
        rngs = {}

        out = self.module.apply(
            params,
            np.array(pixel_values, dtype=np.float32),
            True,
            output_hidden_states=False,
            return_dict=True,
            rngs=rngs,
            mutable=["batch_stats"],
        )
        return out[0].logits

    def _init_params(self, kernel_init: Callable = None, bias_init: Callable = None):
        """
        Initialize a state for the model parameters.

        Not implemented for HF models, as the model parameters are inititialized in
        advance. The pre-built model is passed to the constructor.
        """
        raise NotImplementedError(
            "HF models are passed pre-built. "
            "If you wish to re-initialize the parameters, "
            "please pass a newly constructed model to the constructor of the HFModel "
            "class."
        )

    def train_apply_fn(self, params: dict, inputs: np.ndarray):
        """
        Apply function used for training the model.

        This is the function that is used to apply the model to the data in the training
        loop. It is defined for each child class indivudally and is used to create the
        train state.

        Parameters
        ----------
        params: dict
                Contains the model parameters to use for the model computation.
        inputs : np.ndarray
                Feature vector on which to apply the model.

        Returns
        -------
        Output of the model.
        """
        out, batch_stats = self.apply_fn(
            pixel_values=inputs, params=params, train=True, return_dict=True
        )
        return out.logits, batch_stats

    def apply(self, params: dict, inputs: np.ndarray):
        """
        Apply the model to a feature vector.

        The apply method is used to apply the model to evaluate the model on data,
        outside of the training loop.

        Parameters
        ----------
        params: dict
                Contains the model parameters to use for the model computation.
        inputs : np.ndarray
                Feature vector on which to apply the model.

        Returns
        -------
        Output of the model.
        """
        return self.apply_fn(
            pixel_values=inputs, params=params, train=False, return_dict=True
        ).logits
