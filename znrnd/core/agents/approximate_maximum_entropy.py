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


class ApproximateMaximumEntropy(Agent):
    """
    Class for the approximate maximum entropy data selection agent.
    """

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
        # Allow for optional target_sizes.
        self.target_size = target_size
        start = time.time()
        self._seed_process()
        criteria = False
        self.update_visualization(reference=True)
        self.update_visualization(reference=False)

        while not criteria:
            self._choose_points()
            self._store_metrics()
            self._retrain_network()

            criteria = self._evaluate_agent()
            self.update_visualization(reference=False)
            self.iterations += 1

        stop = time.time()
        if visualize:
            self.visualizer.run_visualization()

        if report:
            self._report_performance(stop - start)

        return self.target_set
