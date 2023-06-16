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
import functools
import inspect
import logging
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from znnl.training_strategies.simple_training import SimpleTraining

import numpy as onp

logger = logging.getLogger(__name__)


def train_func(train_fn: Callable):
    """
    Decorator enabling flexible argument checking and recursive mode when training a
    model.

    The model training of given strategy is executed by calling the train_model method.
    The decorator is used for these methods. It enables:
        1. A flexible checking of arguments before starting the training.
            Different training strategies can demand different input arguments.
            When starting the training this decorator enables to set custom checks and
            default settings in each training strategy.
            This is done in the update_training_kwargs method.
        2. A recursive option of training a model.
            The decorator enables the use of the train_fn in a recursive way.
            In this mode, training is repeated until a condition is satisfied or the
            training is considered to be stuck in a local minimum.
            The definition of the recursive mode is handled by the RecursiveMode class.

    Parameters
    ----------
    train_fn : Callable
            Decorated train function.
            Function that trains a model with a given training strategy.

    Returns
    -------
    Wrapped training function.
    """

    @functools.wraps(train_fn)
    def wrapper(trainer: "SimpleTraining", *args, **kwargs):
        """
        Wrapper function of the train_func decorator.

        A more detailed documentation can be found in train_func.

        Parameters
        ----------
        trainer : SimpleTraining
                Training strategy in which the decorator is used for the train_model
                method.
        args : tuple
                Arguments put into train_fn.
                For more details see the docstring of the wrapped function.
        kwargs : dict
                Keyword arguments put into the train_fn.
                For more details see the docstring of the wrapped function.

        Returns
        -------
        in_training_metrics : dict
        Output of the train_fn. A more detailed documentation can be found in the used
        training strategy.
        """
        # Make args to kwargs to enable easy access
        # Get all possible args
        all_args = list(inspect.signature(train_fn).parameters)
        all_args.remove("self")
        # Check which args were given and put them into a dict
        new_kwargs = {k: v for k, v in zip(all_args, args)}
        # Merge given kwargs and the arg dict
        kwargs.update(new_kwargs)
        # Add the remaining keys to the dict and set them to None
        remaining_kwargs = {k: None for k in all_args if k not in kwargs.keys()}
        kwargs.update(remaining_kwargs)

        # check parameters and model
        kwargs = trainer.update_training_kwargs(**kwargs)

        # Set some defaults
        recursive_mode = trainer.recursive_mode
        kwargs["epochs"] = onp.array(kwargs["epochs"])
        initial_epochs = kwargs["epochs"]

        # Recursive use of train_fn
        if recursive_mode and recursive_mode.use_recursive_mode:
            # Set the default for perturbing the model in recursive training.
            if recursive_mode.perturb_fn is None:
                recursive_mode.perturb_fn = trainer.model.init_model

            condition = False
            counter = 0
            batch_wise_loss = {"train_losses": [], "train_accuracy": []}

            recursive_mode.early_stop = recursive_mode.early_stop.reset()

            while not condition:
                new_batch_wise_loss = train_fn(trainer, **kwargs)
                for key, val in new_batch_wise_loss.items():
                    batch_wise_loss[key].extend(val)

                # Check the condition and update epochs
                counter += 1
                kwargs["epochs"] = (
                    recursive_mode.scale_factor * kwargs["epochs"]
                ).astype(int)
                condition = recursive_mode.update_recursive_condition(
                    trainer.review_metric["loss"]
                )

                # Re-initialize the network if it is simply not converging.
                if counter % recursive_mode.break_counter == 0:
                    recursive_mode.perturb_training()
                    # Reset local variables
                    counter = 0
                    kwargs["epochs"] = initial_epochs

            return batch_wise_loss

        # Non-recursive use of train_fn
        else:
            return train_fn(trainer, **kwargs)

    return wrapper
