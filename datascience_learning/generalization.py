from typing import Dict

import numpy as np


class StochasticFunction(object):
    """Abstract class for Function object"""

    def __init__(self) -> None:
        raise NotImplementedError

    def __call__(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def sample(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Polynomial(StochasticFunction):
    def __init__(self, terms: Dict[int, int], sd: float) -> None:
        self.terms = terms
        self.sd = sd

    def __call__(self, x: np.ndarray):
        """Real value"""
        mean = np.zeros(x.shape)
        for exponent, weight in self.terms.items():
            mean += weight * x ** exponent
        return mean

    def sample(self, x: np.ndarray, n=1) -> np.ndarray:
        return np.repeat(self(x), n, 0) + np.random.randn(n, len(x)) * self.sd
