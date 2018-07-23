import datascience_learning as dsl
import ipywidgets as ipw


def test_polynomial_regression():
    fig = dsl.polynomial_regression()
    assert isinstance(fig, ipw.Widget)
