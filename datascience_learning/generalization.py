from abc import ABC, abstractmethod
from typing import Dict, Union, List
from sklearn.linear_model import LinearRegression

import numpy as np


class StochasticFunction(ABC):
    """Abstract class for Function object"""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def sample(self, x: np.ndarray, n: int) -> Union[np.ndarray, List[np.ndarray]]:
        pass

    # @abstractmethod
    # def fit(self, x: np.ndarray, y: np.ndarray):
    #     pass


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
            # TODO: Fix sampling
            y = self(x)
            return y + np.random.randn(len(y)) * self.sd

    def mse(self, x: float, fh: StochasticFunction) -> List[float]:
        """Decompose mean squared error into bias^2, variance, and noise"""
        n = 10000
        x = np.array([x])
        y = np.repeat(self(x), n)
        yp = fh.sample(x=x, n=n)

        bias_sq = np.mean((y - yp) ** 2)
        variance = np.var(yp)
        error = self.sd ** 2
        return [bias_sq, variance, error]

    def design_matrix(self, x: np.ndarray) -> np.ndarray:
        """Create the design matrix for the polynomial based on x"""
        x = np.atleast_2d(x).T
        exponents = np.atleast_2d(np.array(list(exponent for exponent in self.terms.keys())))
        weight = np.atleast_2d(np.array(list(weight for weight in self.terms.values())))
        return weight * (x ** exponents)

    def fit(self, x: np.ndarray, y: np.ndarray) -> StochasticFunction:
        design_matrix = self.design_matrix(x)
        new_weights = LinearRegression().fit(design_matrix, y).coef_

        for idx, k in enumerate(self.terms.keys()):
            self.terms[k] = new_weights[idx]
