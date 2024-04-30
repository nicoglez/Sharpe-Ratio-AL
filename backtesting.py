import pandas as pd
import numpy as np

# Backtesting Class
class backtesting:

    def __init__(self, weights_summary: list, names: list, data_stocks: pd.DataFrame,
                 data_benchmark: pd.DataFrame, cap_inicial: int):
        self.weights = weights_summary
        self.returns = data_stocks.pct_change().dropna()
        self.bench = data_benchmark.pct_change().dropna()
        self.capital = cap_inicial
        self.names = names

    def history(self):
        # Copy returns to dont mess with the initial variables
        returns = self.returns.copy()
        weights = self.weights.copy()

        # Change pandas display format
        pd.options.display.float_format = '{:,.4f}'.format
        # Empty DF
        h = pd.DataFrame()
        # Make backtesting for the different weights of strategies
        for i in range(len(weights)):
            # get weights
            temp = weights[i]
            # get returns of strategy
            port_returns = 1 + (returns * temp).sum(axis=1)
            # make cumprod
            port_returns.iloc[0] = self.capital
            port_returns = port_returns.cumprod()
            # fill DF
            h[self.names[i]] = port_returns
        # Backtesting for benchmark
        rb = self.bench.copy()
        benchmark = 1 + rb
        # make cumprod
        benchmark.iloc[0] = self.capital
        benchmark = benchmark.cumprod()
        # fill DF
        h["Benchmark"] = benchmark

        return h

    def metrics(self, rf: float):
        # get portfolio evolution
        evol = self.history()
        evol.drop('Benchmark', axis=1, inplace=True)
        # get metrics
        # get metrics
        rets = evol.pct_change().dropna()
        returns = evol.pct_change().dropna().mean() * 252
        std = evol.pct_change().dropna().std() * 252 ** 0.5
        sharpe = (returns - rf) / std
        downside = rets[rets < 0].fillna(0).std()
        upside = rets[rets > 0].fillna(0).std()
        omega = upside / downside

        # create df
        m = pd.DataFrame()
        m["Expected Return"] = returns * 100
        m["Volatility"] = std * 100
        m["Sharpe"] = sharpe
        m["Downside"] = downside * 100
        m["Upside"] = upside * 100
        m["Omega"] = omega
        m['VaR 97.5%'] = np.percentile(rets, 100 - 97.5, axis=0) * 100

        return m

