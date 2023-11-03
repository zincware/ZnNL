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
from typing import Callable, List, Sequence, Union

import jax
import jax.numpy as np
from flax import linen as nn
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
        batch_size: int = 10,
        trace_axes: Union[int, Sequence[int]] = (-1,),
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
        batch_size : int
                Size of batch to use in the NTk calculation.
        trace_axes : Union[int, Sequence[int]]
                Tracing over axes of the NTK.
                The default value is trace_axes(-1,), which reduces the NTK to a tensor
                of rank 2.
                For a full NTK set trace_axes=().
        """
        logger.info(
            "Flax models have occasionally experienced memory allocation issues on "
            "GPU. This is an ongoing bug that we are striving to fix soon."
        )

        self.apply_fn = jax.jit(pre_built_model.__call__)

        # Save input parameters, call self.init_model
        super().__init__(
            pre_built_model=pre_built_model,
            optimizer=optimizer,
            trace_axes=trace_axes,
            ntk_batch_size=batch_size,
        )

    def _ntk_apply_fn(self, params, inputs: np.ndarray):
        """
        Return an NTK capable apply function.

        Parameters
        ----------
        params : dict
                Network parameters to use in the calculation.
        inputs : np.ndarray
                Data on which to apply the network

        Returns
        -------
        Acts on the data with the model architecture and parameter set.
        """
        return self.model_state.apply_fn(inputs, params=params).logits

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

    def apply(self, params: dict, inputs: np.ndarray):
        """
        Apply the model to a feature vector.

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
        return self.model_state.apply_fn(inputs, params=params).logits
