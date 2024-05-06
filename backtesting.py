import pandas as pd
import numpy as np
from typing import Optional

# Backtesting Class
class backtesting:

    def __init__(self, weights_summary: list, names: list, data_stocks: pd.DataFrame,
                 data_benchmark: pd.DataFrame, cap_inicial: int, entrance_fee: Optional[float] = 0):
        self.weights = weights_summary
        self.returns = data_stocks.pct_change().dropna()
        self.bench = data_benchmark.pct_change().dropna()
        self.capital = cap_inicial
        self.entrance_fee = entrance_fee
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
            port_returns.iloc[0] = self.capital * (1 - 1.16 * .00125) - self.entrance_fee
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


def multiple_backtesting(w: list, prices_SIC: pd.DataFrame, benchmark: pd.DataFrame,
                         capital: float, rf: float, fee: float):
    start, end = prices_SIC.index[0], prices_SIC.index[-1]
    ticks = list(prices_SIC.columns.values)

    to_date = lambda x: pd.to_datetime(x, format='%Y-%m-%d')

    data_NYSE = download_data_NYSE(start_date=start, end_date=end, tickers_USA=ticks).download()

    data_NYSE.reset_index(inplace=True)
    data_NYSE['Date'] = [to_date(i) for i in data_NYSE['Date']]
    data_NYSE.set_index('Date', inplace=True)

    prices_SIC.reset_index(inplace=True)
    prices_SIC['index'] = [to_date(i) for i in prices_SIC['index']]
    prices_SIC.set_index('index', inplace=True)

    rets_NYSE = data_NYSE.pct_change().dropna()

    data_NYSE = data_NYSE[[i in prices_SIC.index for i in data_NYSE.index]]
    prices_SIC = prices_SIC[[i in data_NYSE.index for i in prices_SIC.index]]

    optimizer_nyse = OptimizePortfolioWeights(returns=rets_NYSE, kernel=rets_NYSE.cov(), risk_free=rf)
    w_nyse, _ = optimizer_nyse.sharpe_scipy()

    fee = fee * len(w_nyse)

    temp_bt = backtesting(w, ['Adjusted Liquidity SIC'], prices_SIC, benchmark, capital)
    temp_history = temp_bt.history()
    # temp_metrics=temp_bt.metrics(rf=rf)

    temp_bt_2 = backtesting([w_nyse], ['Sharpe NYSE'], data_NYSE, benchmark, capital, fee)

    temp_history_2 = temp_bt_2.history()
    # temp_metrics_2=temp_bt_2.metrics(rf=rf)
    history = temp_history.join(temp_history_2.iloc[:, 0]).dropna()

    # metrics=temp_metrics.T.join(temp_metrics_2.T).T

    return history  # , metrics

