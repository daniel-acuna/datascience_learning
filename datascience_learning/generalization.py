from typing import Dict, Union, List

import numpy as np
from abc import ABC, abstractmethod


class StochasticFunction(ABC):
    """Abstract class for Function object"""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def sample(self, x: np.ndarray = None, n: int) -> Union[np.ndarray, List[np.ndarray]]:
        pass


class Polynomial(StochasticFunction):
    def __init__(self, terms: Dict[int, int], sd: float) -> None:
        super().__init__()
        self.terms = terms
        self.sd = sd

    def __call__(self, x: np.ndarray):
        """Real value"""
        mean = np.zeros(x.shape)
        for exponent, weight in self.terms.items():
            mean += weight * x ** exponent
        return mean

    def sample(self, x: np.ndarray = None, n=1):
        if x is None:
            x = np.random.randn(n)
            y = self.sample(x).flatten()
            return x, y
        else:
            return np.repeat(self(x), n, 0) + np.random.randn(n, len(x)) * self.sd

    def mse(self, x: float, fh: StochasticFunction) -> List[float]:
        """Decompose mean squared error into bias^2, variance, and noise"""
        n = 10000
        x = np.ndarray([x])
        y = np.repeat(self(x), n)
        yp = fh.sample(x=x, n=n)

        bias_sq = np.mean((y - yp) ** 2)
        variance = np.var(yp)
        error = self.sd ** 2
        return [bias_sq, variance, error]
