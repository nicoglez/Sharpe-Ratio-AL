import numpy as np
import pandas as pd
from scipy.optimize import minimize
from backtesting import backtesting
import optuna
optuna.logging.disable_default_handler()

# Optimize Weights Class
class OptimizePortfolioWeights:

    def __init__(self, returns: pd.DataFrame, kernel: np.array, risk_free: float):

        self.rets = returns
        self.k = kernel
        self.rf = risk_free / 252
        self.n_stocks = len(returns.columns)

    # Normal Sharpe Ratio using minimize optimization
    def sharpe_scipy(self):
        rets = self.rets
        rend, k, rf = self.rets.mean(), self.k, self.rf

        sr = lambda w: -((np.dot(rend, w) - rf) / ((w.reshape(-1, 1).T @ k @ w) ** 0.5))

        result = minimize(sr, np.ones(len(rets.T)), bounds=[(0, None)] * len(rets.T),
                          constraints={'fun': lambda w: sum(w) - 1, 'type': 'eq'},
                          options={'xatol': 1e-8})

        return result.x, -result.fun

    # Normal Sharpe Ratio using Montecarlo Optimization
    def sharpe_montecarlo(self, n_sims: int):

        rend, k, rf = self.rets.mean(), self.k, self.rf

        n_stocks = len(self.rets.columns)

        w_s, ratios = [], []

        for i in range(n_sims):
            w_temp = np.random.uniform(0, 1, n_stocks)
            w_temp = w_temp / np.sum(w_temp)
            ret_temp = np.dot(w_temp, rend)
            std_temp = (w_temp.reshape(1, -1) @ k @ w_temp) ** 0.5
            rs = (ret_temp - rf) / std_temp

            w_s.append(w_temp)
            ratios.append(rs)

        return w_s[np.argmax(ratios)], np.max(ratios)

    # Liquidity-Adjusted Sharpe using Scipy
    def sharpe_AL_scipy(self, last_prices: np.array, volume: np.array, capital: int, lmda: float):
        rets = self.rets
        rend, k, rf = self.rets.mean(), self.k, self.rf

        def sr(w):
            expected_return = np.dot(rend, w) - rf
            std = (w.reshape(-1, 1).T @ k @ w) ** 0.5
            sharpe = expected_return / std
            liquidity = sum((capital * w * 1 / last_prices) / volume)

            return -sharpe + lmda * liquidity

        result = minimize(sr, np.ones(len(rets.T)), bounds=[(0, None)] * len(rets.T),
                          constraints={'fun': lambda w: sum(w) - 1, 'type': 'eq'},
                          options={'xatol': 1e-8})

        return result.x, -result.fun

    # Liquidity-Adjusted Sharpe using Montecarlo
    def sharpe_AL_montecarlo(self, n_sims: int, last_prices: np.array, volume: np.array,
                             capital: int, lmda: float, lmbda_2: float = 0):

        rend, k, rf = self.rets.mean(), self.k, self.rf

        w_s, ratios = [], []

        for i in range(n_sims):
            w_temp = np.random.uniform(0, 1, self.n_stocks)
            w_temp = w_temp / np.sum(w_temp)

            expected_return = np.dot(rend, w_temp) - rf

            std = (w_temp.reshape(-1, 1).T @ k @ w_temp) ** 0.5
            sharpe = expected_return / std
            liquidity = sum((capital * w_temp * 1 / last_prices) / volume)
            w_s.append(w_temp)
            ratios.append(sharpe - lmda * liquidity)

        return w_s[np.argmax(ratios)], np.max(ratios)


def optimize_lmda(trial, prices: pd.DataFrame, benchmark: pd.DataFrame, volume: np.array,
                  kernel: np.array, capital: int, rf: float):
    lam = trial.suggest_float("lam", 0.01, 10)

    rets = prices.pct_change().dropna()
    last_prices = prices.iloc[-1, :]
    optimizer = OptimizePortfolioWeights(returns=rets, kernel=kernel, risk_free=rf)

    weights, _ = optimizer.sharpe_AL_scipy(last_prices, volume, capital, lmda=lam)

    bt = backtesting(weights_summary=[weights],
                     names=['OptimalWeights'],
                     data_stocks=prices, data_benchmark=benchmark, cap_inicial=capital)

    history = bt.history()

    trial.set_user_attr("weights", weights)

    return -history["OptimalWeights"].values[-1]


def gaussian_kernel_matrix(X, sigma):
    return np.exp(-np.sum((X[:, np.newaxis] - X) ** 2, axis=-1) / ((2 * sigma ** 2) * len(X.T)))


def squared_exp_kernel(X, sigma, l):
    return np.pi * l * sigma ** 2 * np.exp(-np.sum((X[:, np.newaxis] - X) ** 2, axis=-1) / (2 * (np.sqrt(2) * l) ** 2))


def polynomial_kernel(X, alpha, beta):
    return (alpha ** 2 * np.identity(len(X))) @ X @ X.T + beta ** 2


def periodic_kernel(X, l):
    return np.exp(-(2 * np.sin(np.sum((X[:, np.newaxis] - X), axis=-1) / 2) ** 2) / l ** 2)


def ornstein_kernel(X, l):
    return np.exp(-np.sum(abs(X[:, np.newaxis] - X), axis=-1) / l)


def optimize_kernel_params(trial, prices: pd.DataFrame, volume: np.array, capital: int,
                           n_sims: int, rf: float, k, params, X_kernel, benchmark):
    if len(params) == 1:
        param_0 = trial.suggest_float(f"{params[0]}", 1e-3, 10000)
        kernel = k(X_kernel, param_0)

    else:
        param_0 = trial.suggest_float(f"{params[0]}", 1e-3, 10000)
        param_1 = trial.suggest_float(f"{params[1]}", 1e-3, 10000)
        kernel = k(X_kernel, param_0, param_1)

    lam = trial.suggest_float("lam", 1e-3, 50)

    rets = prices.pct_change().dropna()

    optimizer = OptimizePortfolioWeights(returns=rets, kernel=kernel, risk_free=rf)
    weights, _ = optimizer.sharpe_AL_scipy(last_prices=prices.iloc[-1, :],
                                           volume=volume,
                                           capital=capital,
                                           lmda=lam)
    bt = backtesting(weights_summary=[weights],
                     names=['OptimalWeights'],
                     data_stocks=prices, data_benchmark=benchmark, cap_inicial=capital)

    history = bt.history()

    trial.set_user_attr("weights", weights)

    return -history["OptimalWeights"].values[-1]

def try_kernels(prices: pd.DataFrame, benchmark: pd.DataFrame, volume: pd.DataFrame, capital: float, rf: float):

    rets=prices.pct_change().dropna()

    kernels = [gaussian_kernel_matrix, squared_exp_kernel, polynomial_kernel, periodic_kernel, ornstein_kernel]

    results = {}

    for ke in kernels:

        params = list(ke.__code__.co_varnames)[1:]

        for method in ['returns', 'prices']:
            typ = rets if method == 'returns' else prices
            kernel_study = optuna.create_study(direction="minimize")
            kernel_study.optimize(lambda trial: optimize_kernel_params(trial, prices, volume.mean(), capital, 1, rf,
                                                                       ke, params, typ.T.values, benchmark), n_trials=100)

            p, value, weights = kernel_study.best_params, kernel_study.best_value, kernel_study.best_trial.user_attrs[
                "weights"]

            temp = {f'{ke.__name__}_{method}': [p, value, weights]}
            results.update(temp)

    for result in results:
        best_value = np.inf
        best_method = None
        backtest = results.get(result)[1]

        if best_value > backtest:
            best_method = result

    return best_method, results.get(best_method)


def random_portfolio_testing(capital, rf, n_sims, prices, rets, volume, benchmark):
    # Crear listas vacias
    result_normal_sharpe, result_al_sharpe = [], []

    for i in range(n_sims):
        random = np.random.choice(prices.columns, 5)

        # Cortar activos
        temp_prices = prices[random]
        temp_rets = rets[random]
        temp_volume = volume[random]

        # Realizar optimizacion
        study_cov = optuna.create_study(direction="minimize")
        study_cov.optimize(lambda trial: optimize_lmda(trial, temp_prices, benchmark,
                                                       temp_volume.iloc[-90:, :].mean(), temp_rets.cov(),
                                                       capital, rf), n_trials=100)

        optimizer_1 = OptimizePortfolioWeights(returns=temp_rets, kernel=temp_rets.cov(), risk_free=rf)
        w_1_covariance, _ = optimizer_1.sharpe_scipy()
        w_2_covariance = study_cov.best_trial.user_attrs["weights"]

        bt = backtesting(weights_summary=[w_1_covariance, w_2_covariance],
                         names=['Normal Sharpe', 'AL Sharpe'],
                         data_stocks=temp_prices, data_benchmark=benchmark, cap_inicial=capital)

        history = bt.history()

        result_normal_sharpe.append(history['Normal Sharpe'])
        result_al_sharpe.append(history['AL Sharpe'])

    return result_normal_sharpe, result_al_sharpe