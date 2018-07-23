import datascience_learning as dsl
import ipywidgets as ipw


def test_gradient_descent():
    fig = dsl.gradient_descent()
    assert isinstance(fig, ipw.Widget)
