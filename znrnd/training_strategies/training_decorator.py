"""
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincware Project.

Description: Decorator used for training a model.
"""
import functools
import inspect
import logging
from typing import Callable

import numpy as onp

logger = logging.getLogger(__name__)


def train_func(train_fn: Callable):
    """
    Decorator enabling a recursive mode of a train function.

    The training of a model (with a given strategy) is executed by calling the
    train_model function.
    The decorator enables the use of this function in a recursive way. The training is
    repeated until a condition is satisfied or the training is considered to be stuck in
    a local minimum.
    In the letter case, the model is perturbed and the training starts again. The
    default perturbation method is the re-initialization of the model. The perturbation
    function is an optional parameter when constructing the recursive mode.

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
    def wrapper(trainer, *args, **kwargs):
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
