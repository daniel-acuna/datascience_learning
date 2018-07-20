from __future__ import print_function
import numpy as np
from bqplot import (
    Axis, ColorAxis, LinearScale, DateScale, DateColorScale, OrdinalScale,
    OrdinalColorScale, ColorScale, Scatter, Lines, Figure, Tooltip
)
from ipywidgets import Label
from ipywidgets import HBox, VBox

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

import ipywidgets as widgets
from .datasets import fake_datasets

def polynomial_regression():
    """Polynomial regression example"""
    regression_config = {'degree': 1}
    sc_x = LinearScale(min=-100, max=100)
    sc_y = LinearScale(min=-100, max=100)
    scat = Scatter(x=[],
                   y=[],
                   scales={'x': sc_x, 'y': sc_y},
                   colors=['orange'],
                   enable_move=True)
    lin = Lines(x=[],
                y=[],
                scales={'x': sc_x, 'y': sc_y},
                line_style='dotted',
                colors=['orange'])

    def update_line(change=None):
        if len(scat.x) == 0:
            lin.x = []
            lin.y = []
            return
        pipe = make_pipeline(PolynomialFeatures(degree=regression_config['degree']),
                             LinearRegression())
        pipe.fit(scat.x.reshape(-1, 1), scat.y)
        with lin.hold_sync():
            lin.x = np.linspace(sc_x.min, sc_x.max)
            lin.y = pipe.predict(
                np.linspace(sc_x.min, sc_x.max).reshape(-1, 1)
            )

    update_line()
    # update line on change of x or y of scatter
    scat.observe(update_line, names=['x'])
    scat.observe(update_line, names=['y'])
    with scat.hold_sync():
        scat.enable_move = False
        scat.interactions = {'click': 'add'}
    ax_x = Axis(scale=sc_x, tick_format='0.0f')
    ax_y = Axis(scale=sc_y,
                tick_format='0.0f',
                orientation='vertical')

    fig = Figure(marks=[scat, lin],
                 axes=[ax_x, ax_y],
                 fig_margin={'top': 0,
                             'bottom': 30,
                             'left': 40,
                             'right': 0})
    button = widgets.Button(description="Reset")

    def degree_change(change):
        regression_config['degree'] = change['new']
        update_line()

    degree = widgets.IntSlider(value=regression_config['degree'],
                               min=1, max=5, step=1,
                               description='Degree')
    degree.observe(degree_change, names='value')

    def on_button_clicked(change=None):
        with scat.hold_sync():
            scat.x = []
            scat.y = []

    dropdown_w = widgets.Dropdown(
        options=fake_datasets(ndim=2),
        value=None,
        description='Dataset:',
    )

    def dropdown_on_change(change):
        if change['type'] == 'change' and change['name'] == 'value':
            print("changed to %s" % change['new'])
            x, y = fake_datasets(name=change['new'])
            with scat.hold_sync():
                scat.x = x.flatten()
                scat.y = y.flatten()

    dropdown_w.observe(dropdown_on_change)

    button.on_click(on_button_clicked)
    return VBox((Label(value="test"),
                 dropdown_w,
                 HBox((degree, button)),
                 fig))
