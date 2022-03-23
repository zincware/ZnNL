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
TSNE visualizer for the RND.
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.manifold import TSNE


class TSNEVisualizer:
    """
    Perform visualization on network representations with TSNE.
    """

    def __init__(self, components: int = 2):
        """
        Constructor for the TSNE visualizer.

        Parameters
        ----------
        components : int
                Number of components to use in the representation. Either 2 or 3.
        """
        self.model = TSNE(n_components=components, init="random", learning_rate=200)

        self.reference = []
        self.dynamic = []

    def build_representation(self, data: np.ndarray, reference: bool = False):
        """
        Construct a TSNE representation.

        Parameters
        ----------
        data : np.ndarray (n_coordinates, dimension)
                Data on which the representation should be constructed.
        reference : bool
                If true, populate the reference attribute.

        Returns
        -------
        stores a plot
        """
        if reference:
            self.reference.append(self.model.fit_transform(data))
        else:
            self.dynamic.append(self.model.fit_transform(data))

    def data_gen(self):
        """
        Generate next data point for the animation.

        Returns
        -------

        """
        for count in self.dynamic:
            yield count[:, 0], count[:, 1]

    def run_visualization(self):
        """
        Run the visualization.
        """
        fig_dict = {"data": [], "layout": {}, "frames": []}
        # fill in most of layout
        fig_dict["layout"]["xaxis"] = {
            "range": [-15, 15],
            "title": "x",
            "domain": [0.0, 0.7],
        }
        fig_dict["layout"]["yaxis"] = {"title": "y", "range": [-15, 15]}
        fig_dict["layout"]["xaxis2"] = {"domain": [0.8, 1.0]}
        fig_dict["layout"]["yaxis2"] = {"anchor": "x2"}
        fig_dict["layout"]["hovermode"] = "closest"
        fig_dict["layout"]["updatemenus"] = [
            {
                "buttons": [
                    {
                        "args": [
                            None,
                            {
                                "frame": {"duration": 500, "redraw": False},
                                "fromcurrent": True,
                                "transition": {
                                    "duration": 300,
                                    "easing": "quadratic-in-out",
                                },
                            },
                        ],
                        "label": "Play",
                        "method": "animate",
                    },
                    {
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                        "label": "Pause",
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top",
            }
        ]

        sliders_dict = {
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "font": {"size": 20},
                "prefix": "Step:",
                "visible": True,
                "xanchor": "right",
            },
            "transition": {"duration": 300, "easing": "cubic-in-out"},
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [],
        }

        # Add initial data
        fig_dict["data"].append(
            {
                "x": self.dynamic[0][:, 0],
                "y": self.dynamic[0][:, 1],
                "mode": "markers",
                "name": "Predictor"
            }
        )
        fig_dict["data"].append(
            {
                "x": self.reference[0][:, 0],
                "y": self.reference[0][:, 1],
                "mode": "markers",
                "xaxis": "x2",
                "yaxis": "y2",
                "name": "Target"
            }
        )

        # Make the figure frames.
        for i, item in enumerate(self.dynamic):
            frame = {
                "data": [
                    {
                        "x": item[:, 0],
                        "y": item[:, 1],
                        "mode": "markers",
                        "name": "Predictor"
                    },
                    {
                        "x": self.reference[0][:, 0],
                        "y": self.reference[0][:, 1],
                        "mode": "markers",
                        "xaxis": "x2",
                        "yaxis": "y2",
                        "name": "Target"
                    }
                ]
            }

            fig_dict["frames"].append(frame)

            slider_step = {
                "args": [
                    [i],
                    {
                        "frame": {"duration": 300, "redraw": False},
                        "mode": "immediate",
                        "transition": {"duration": 300},
                    },
                ],
                "label": i,
                "method": "animate",
            }
            sliders_dict["steps"].append(slider_step)

        fig_dict["layout"]["sliders"] = [sliders_dict]

        figure = go.Figure(fig_dict)
        figure.show()
