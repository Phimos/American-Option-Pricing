from typing import Callable

import numpy as np
from sklearn.linear_model import LinearRegression


def power_polynomial(order: int = 2) -> Callable:
    """Generate power polynomial basis

    Args:
        order (int, optional): order of polynomial. Defaults to 2.

    Returns:
        Callable: power polynomial basis
    """
    def func(x):
        result = np.zeros((x.shape[0], order + 1))
        result[:, 0] = 1
        for i in range(order):
            result[:, i+1] = result[:, i] * x
        return result
    return func


def laguerre_polynomial(order: int = 2) -> Callable:
    """Generate Laguerre polynomial basis

    Details: https://en.wikipedia.org/wiki/Laguerre_polynomials

    Recurrence relation:
        (n+1)L_{n+1}(x) = (2n+1-x)L_n(x) - nL_{n-1}(x)

    Args:
        order (int, optional): order of polynomial. Defaults to 2.

    Returns:
        Callable: Laguerre polynomial basis
    """
    def func(x):
        if order == 0:
            return np.ones((x.shape[0], 1))
        result = np.zeros((x.shape[0], order + 1))
        result[:, 0] = 1
        result[:, 1] = 1 - x
        for i in range(1, order):
            result[:, i+1] = ((2 * i + 1 - x) * result[:, i] -
                              i * result[:, i-1]) / (i + 1)
        return result
    return func


def legendre_polynomial(order: int = 2) -> Callable:
    """Generate Legendre polynomial basis

    Details: https://en.wikipedia.org/wiki/Legendre_polynomials

    Recurrence relation:
        (n+1)P_{n+1}(x) = (2n+1)xP_n(x) - nP_{n-1}(x)

    Args:
        order (int, optional): order of polynomial. Defaults to 2.

    Returns:
        Callable: Legendre polynomial basis
    """
    def func(x):
        if order == 0:
            return np.ones((x.shape[0], 1))
        result = np.zeros((x.shape[0], order + 1))
        result[:, 0] = 1
        result[:, 1] = x
        for i in range(1, order):
            result[:, i+1] = ((2 * i + 1) * x * result[:, i] -
                              i * result[:, i-1]) / (i + 1)
        return result
    return func


def hermite_polynomial(order: int = 2) -> Callable:
    """Generate Hermite polynomial basis

    Details: https://en.wikipedia.org/wiki/Hermite_polynomials

    Recurrence relation:
        H_{n+1}(x) = 2xH_n(x) - 2nH_{n-1}(x)

    Args:
        order (int, optional): order of polynomial. Defaults to 2.

    Returns:
        Callable: Hermite polynomial basis
    """
    def func(x):
        if order == 0:
            return np.ones((x.shape[0], 1))
        result = np.zeros((x.shape[0], order + 1))
        result[:, 0] = 1
        result[:, 1] = x
        for i in range(1, order):
            result[:, i+1] = 2 * x * result[:, i] - 2 * i * result[:, i-1]
        return result
    return func


def LeastSquaresMonteCarlo(S0: float, K: float, T: float, r: float,
                           sigma: float, payoff: str, n_step=10000,
                           n_path=10000, basis=power_polynomial, order=8) -> float:
    """Least squares Monte Carlo method for American option pricing

    Args:
        S0 (float): initial stock price
        K (float): strike price
        T (float): time to maturity
        r (float): risk-free rate
        sigma (float): volatility
        payoff (str): payoff function, either 'call' or 'put'
        n_step (int, optional): num of simulation steps. Defaults to 10000.
        n_path (int, optional): num of simulation paths. Defaults to 10000.
        basis ([type], optional): basis function. Defaults to power_polynomial.
        order (int, optional): parameter of basis function. Defaults to 8.

    Raises:
        ValueError: if payoff is not 'call' or 'put'

    Returns:
        float: option price
    """
    if payoff.lower() == 'call':
        def payoff(x): return np.maximum(x - K, 0)
    elif payoff.lower() == 'put':
        def payoff(x): return np.maximum(K - x, 0)
    else:
        raise ValueError('payoff must be either call or put')

    dt = T / n_step
    df = np.exp(-r * dt)

    # Simulate n_path independent paths
    increments = np.random.normal(size=(n_path, n_step + 1))
    increments = (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * increments
    increments[:, 0] = 0
    increments = np.cumsum(increments, axis=1)
    S = S0 * np.exp(increments)

    # set payoff at terminal nodes
    V = np.zeros((n_path, n_step + 1))
    H = payoff(S)
    V[:, -1] = H[:, -1]

    regression = LinearRegression(fit_intercept=False)

    for t in range(n_step - 1, 0, -1):
        # select good paths, only apply regression to those paths
        good_paths = H[:, t] > 0

        # linear regression
        transformed = basis(order)(S[good_paths, t])
        regression.fit(transformed, V[good_paths, t + 1])
        C = regression.predict(transformed) * df

        # update value function
        exercise = np.zeros(n_path, dtype=bool)
        exercise[good_paths] = H[good_paths, t] > C
        V[exercise, t] = H[exercise, t]
        V[~exercise, t] = V[~exercise, t+1] * df

    return np.mean(V[:, 1]) * df


def CoxRossRubinstein(S0: float, K: float, T: float, r: float, sigma: float,
                      payoff: str, n_step: int = 10000) -> float:
    """Cox-Ross-Rubinstein method for American option pricing

    Args:
        S0 (float): initial stock price
        K (float): strike price
        T (float): time to maturity
        r (float): risk-free rate
        sigma (float): volatility
        payoff (str): payoff function, either 'call' or 'put'
        n_step (int, optional): num of steps. Defaults to 10000.

    Raises:
        ValueError: if payoff is not 'call' or 'put'

    Returns:
        float: option price
    """
    if payoff.lower() == 'call':
        def payoff(x): return np.maximum(x - K, 0)
    elif payoff.lower() == 'put':
        def payoff(x): return np.maximum(K - x, 0)
    else:
        raise ValueError('payoff must be either call or put')

    dt = T / n_step
    df = np.exp(-r * dt)
    u, d = np.exp(sigma * np.sqrt(dt)), np.exp(-sigma * np.sqrt(dt))
    p = (np.exp(r * dt) - d) / (u - d)
    q = 1 - p

    mu = np.arange(n_step + 1)
    mu = np.tile(mu, (n_step + 1, 1))
    md = np.transpose(mu)

    S = S0 * (u ** (mu - md)) * (d ** md)

    V = np.zeros((n_step+1, n_step+1))
    V[:, -1] = payoff(S[:, -1])

    for t in range(n_step, 0, -1):
        V[:t, t-1] = np.maximum((p * V[:t, t] + q *
                                 V[1: t+1, t]) * df, payoff(S[:t, t-1]))
    return V[0, 0]


if __name__ == '__main__':
    print(CoxRossRubinstein(100, 100, 1, 0.1, 0.2, 'call'))
    print(LeastSquaresMonteCarlo(100, 100, 1,
                                 0.1, 0.2, 'call', basis=power_polynomial))
    print(LeastSquaresMonteCarlo(100, 100, 1,
                                 0.1, 0.2, 'call', basis=laguerre_polynomial))
    print(LeastSquaresMonteCarlo(100, 100, 1,
                                 0.1, 0.2, 'call', basis=legendre_polynomial))
    print(LeastSquaresMonteCarlo(100, 100, 1,
                                 0.1, 0.2, 'call', basis=hermite_polynomial))
