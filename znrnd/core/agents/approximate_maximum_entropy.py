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
Module for the approximate maximum entropy agent.
"""
from znrnd.core.agents.agent import Agent
from znrnd.core.models.model import Model


class ApproximateMaximumEntropy(Agent):
    """
    Class for the approximate maximum entropy data selection agent.
    """

    def __init__(self, target_network: Model):
        """
        Constructor for the Approximate maximum entropy agent.

        Parameters
        ----------
        target_network : Model
                Model of the target network.
        """
        self.target_network = target_network

    def build_dataset(
        self, target_size: int = None, visualize: bool = False, report: bool = True
    ):
        """
        Run the random network distillation methods and build the target set.

        Parameters
        ----------
        target_size : int
                Target size of the operation.
        visualize : bool (default=False)
                If true, a t-SNE visualization will be performed on the final models.
        report : bool (default=True)
                If true, print a report about the RND performance.

        Returns
        -------
        target_set : list
                Returns the newly constructed target set.
        """
