import datascience_learning as dsl
import numpy as np


def test_polynomial():
    f = dsl.Polynomial({0: 10}, sd=2)
    np.testing.assert_equal(f(np.array([0, 1, 2, 3])), np.array([10, 10, 10, 10]))
