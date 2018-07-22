import numpy as np

from sklearn.datasets import make_regression


def fake_datasets(name=None, ndim=2):
    datasets = {2: ["Exponential decay", "Linear", "Linear with outliers"]}
    if name is None:
        return datasets[ndim]

    if name == "Linear":
        x, y = make_regression(
            n_samples=20, n_features=1, n_informative=1, n_targets=1, bias=0, noise=20
        )
        return x * 30, y
    elif name == "Linear with outliers":
        x, y = make_regression(
            n_samples=20, n_features=1, n_informative=1, n_targets=1, bias=0, noise=20
        )
        y = np.append(y, np.array([100, 90]))
        x = np.append(x, np.array([[-2], [-1]]), axis=0)
        return x * 30, y
    elif name == "Exponential decay":
        a0 = 20
        a1 = -0.1
        # x = np.random.random(40) * 100
        x = np.linspace(0, 100, num=30)  # + np.random.randn(20) * 5
        y = a0 * np.exp(a1 * x)
        x = (x - 50) * 1.8
        y = (y - 10) * 4
        y += y + np.random.randn(*y.shape) * 10

        return x, y
