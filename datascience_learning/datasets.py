
from sklearn.datasets import make_regression

def fake_datasets(name=None, ndim=2):
    datasets = \
        {2: [
            "Exponential decay",
            "Linear",
            "Linear with outlier"
        ],
    }
    if name is None:
        return datasets[ndim]

    if name == "Exponential decay":
        X, y = make_regression(n_samples=100,
                            n_features=1,
                            n_informative=1,
                            n_targets=1,
                            bias=10,
                            noise=100,
                            random_state=0)
        return X, y