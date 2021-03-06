import datascience_learning as dsl
import numpy as np
import pytest


def test_stochaticfunction():
    with pytest.raises(TypeError):
        _ = dsl.StochasticFunction()


def test_polynomial():
    f2 = dsl.Polynomial({0: 10}, sd=2)
    x1 = np.array([0, 1, 2, 3])
    y1 = np.array([10, 10, 10, 10])
    # predictions
    np.testing.assert_equal(f2(x1), y1)
    # noise
    sample1 = f2.sample(np.array([0]), n=10000)
    np.testing.assert_almost_equal(sample1.std(), 2, decimal=1)

    f2 = dsl.Polynomial({0: 0, 1: 1}, sd=1)
    x2 = np.linspace(0, 10)
    y2 = x2
    np.testing.assert_equal(f2(x2), y2)

    # noise
    sample2 = f2.sample(np.array([0]), n=10000)
    np.testing.assert_almost_equal(sample2.std(), 1, decimal=1)

    # sample both x and y
    x, y = f2.sample(n=100)
    assert x.shape == y.shape


def test_mse():
    f_poly = dsl.Polynomial({0: -10, 2: 1, 3: -0.5}, sd=1.)
    f_poly2 = dsl.Polynomial({0: -10, 2: 1, 3: -0.5}, sd=0.)
    f_linear = dsl.Polynomial({1: 1}, sd=1.)
    bias_sq1, var1, _ = f_poly.mse(x=0, fh=f_linear)
    assert bias_sq1 > 0 and var1 > 0
    bias_sq2, _, _ = f_poly.mse(f_poly)
    np.testing.assert_almost_equal(bias_sq2, 0)

    bias_sq3, var3, _ = f_poly.mse(x=0, fh=f_poly2)
    np.testing.assert_almost_equal([bias_sq3, var3], [0, 0])
