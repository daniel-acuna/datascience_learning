from __future__ import print_function
import numpy as np
from bqplot import Axis, LinearScale, Scatter, Lines, Figure
from ipywidgets import Label
from ipywidgets import HBox, VBox

import ipywidgets as widgets
from .datasets import fake_datasets


def gradient_descent():
    line_params = {"b": 0, "m": 0, "iter": 1}

    sc_x = LinearScale(min=-100, max=100)
    sc_y = LinearScale(min=-100, max=100)
    scat = Scatter(
        x=[], y=[], scales={"x": sc_x, "y": sc_y}, colors=["orange"], enable_move=True
    )
    lin = Lines(x=[], y=[], scales={"x": sc_x, "y": sc_y}, colors=["blue"])

    ax_x = Axis(scale=sc_x, tick_format="0.0f", label="x")
    ax_y = Axis(scale=sc_y, tick_format="0.0f", orientation="vertical", label="y")

    fig_function = Figure(marks=[scat, lin], axes=[ax_x, ax_y])

    sc_x_cost = LinearScale(min=0, max=100)
    sc_y_cost = LinearScale(min=0, max=100)
    lin_cost = Lines(x=[], y=[], scales={"x": sc_x_cost, "y": sc_y_cost})
    ax_x_cost = Axis(scale=sc_x_cost, tick_format="0.0f", label="iteration")
    ax_y_cost = Axis(
        scale=sc_y_cost,
        tick_format="0.0f",
        orientation="vertical",
        label="Mean Squared Error",
    )

    fig_cost = Figure(marks=[lin_cost], axes=[ax_x_cost, ax_y_cost])

    def draw_line():
        x = np.linspace(-100, 100)
        y = line_params["b"] + line_params["m"] * x
        with lin.hold_sync():
            lin.x = x
            lin.y = y

    play_button = widgets.Play(
        interval=100,
        value=0,
        min=0,
        max=100,
        step=1,
        repeat=True,
        description="Run gradient descent",
        disabled=False,
    )

    year_slider = widgets.IntSlider(
        min=0, max=100, step=1, description="Step", value=0, disabled=True
    )

    def mse():
        b = line_params["b"]
        m = line_params["m"]
        return (((scat.x * m + b) - scat.y) ** 2).mean()

    def play_change(change):
        b = line_params["b"]
        m = line_params["m"]
        b_gradient = 0
        m_gradient = 0
        n = len(scat.x)
        learning_rate = 0.0001
        for i in range(0, len(scat.x)):
            b_gradient += -(2 / n) * (scat.y[i] - ((m * scat.x[i]) + b))
            m_gradient += -(2 / n) * scat.x[i] * (scat.y[i] - ((m * scat.x[i]) + m))
        b = b - (learning_rate * 500 * b_gradient)
        m = m - (learning_rate * m_gradient)

        line_params["b"] = b
        line_params["m"] = m
        lin_cost.x = np.append(np.array(lin_cost.x), np.array([line_params["iter"]]))
        lin_cost.y = np.append(np.array(lin_cost.y), mse())
        sc_x_cost.min = np.min(lin_cost.x)
        sc_x_cost.max = np.max(lin_cost.x)
        sc_y_cost.min = 0
        sc_y_cost.max = np.max(lin_cost.y)

        line_params["iter"] = line_params["iter"] + 1

        draw_line()

    play_button.observe(play_change, "value")
    widgets.jslink((play_button, "value"), (year_slider, "value"))

    # reset reset_button
    reset_button = widgets.Button(description="Reset")

    def on_button_clicked(change=None):
        x, y = fake_datasets("Linear")
        with scat.hold_sync():
            scat.x = x.flatten()
            scat.y = y.flatten()
        with lin_cost.hold_sync():
            lin_cost.x = []
            lin_cost.y = []

        line_params["b"] = (np.random.random() - 0.5) * 100
        line_params["m"] = np.random.random() - 0.5
        line_params["iter"] = 1
        draw_line()

    on_button_clicked()

    reset_button.on_click(on_button_clicked)

    return VBox(
        (
            widgets.HTML("<h1>Gradient Descent</h1>"),
            reset_button,
            HBox((Label("Run gradient descent"), play_button, year_slider)),
            HBox((fig_function, fig_cost)),
        )
    )
