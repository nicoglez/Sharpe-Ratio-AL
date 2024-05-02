import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import warnings

sns.set_style()
warnings.filterwarnings("ignore")

def plot_volume(volume: pd.Series):

    sorted_data = sorted(zip(volume.values, volume.index), key=lambda tup: tup[0])
    sorted_x = [tup[0] for tup in sorted_data][::-1]
    sorted_y = [tup[1] for tup in sorted_data][::-1]

    plt.figure(figsize=(12, 4), facecolor='none')
    bars = plt.bar(sorted_y, sorted_x)

    plt.title('Volume of Stocks in NYSE')
    plt.ylabel('Volume')
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:,.0f}'.format(y)))
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 500, f'{height:,.0f}', ha='center')
    plt.xlabel('Stock')
    plt.show()


def plot_prices(prices):
    plt.figure(figsize=(12, 6))
    plt.plot(prices, label=prices.columns)
    plt.legend()
    plt.title('Stock Prices Plot')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.show()


def plot_backtesting(history: pd.DataFrame):
    plt.figure(figsize=(12, 6), facecolor='none')
    plt.plot(history, label=history.columns)
    plt.title("Backtesting of Strategies")
    plt.xlabel("Date")
    plt.ylabel("Capital")
    plt.legend()
    plt.show();

def plot_pie(w_1: np.array, w_2: np.array, labels: list):
    # Define the minimum percentage threshold
    min_percentage_threshold = 1

    # Your existing code for creating subplots and pie charts
    fig, axs = plt.subplots(1, 2, figsize=(12, 8), facecolor='none')

    # Pie chart for the first subplot
    wedges1, labels1, autotexts1 = axs[0].pie(w_1, autopct=lambda
        pct: f'{pct:.1f}%' if pct > min_percentage_threshold else '', colors=plt.cm.Paired.colors)
    axs[0].set_title('Weights Sharpe Ratio')

    # Pie chart for the second subplot
    wedges2, labels2, autotexts2 = axs[1].pie(w_2, autopct=lambda
        pct: f'{pct:.1f}%' if pct > min_percentage_threshold else '', colors=plt.cm.Paired.colors)
    axs[1].set_title('Weights Liquidity-Adjusted Sharpe Ratio')

    plt.tight_layout()
    plt.legend(labels, loc='upper right')
    plt.show()

def plot_simulation_hist(data):
    plt.figure(figsize=(12, 6))
    plt.hist(data['Adjusted by Liquidity'], bins=1500, label='Adjusted-Liquidity Sharpe', alpha=0.5)
    plt.hist(data['Sharpe Ratio'], bins=1500, label='Sharpe', alpha=0.5)
    plt.xlim(-50, 400)
    plt.legend()
    plt.title('Strategies Total Return Distribution')
    plt.xlabel('Expected Return')
    plt.ylabel('Count')
    plt.show();


def plot_boxplot(simulation_data):
    medians = np.median(simulation_data, axis=0)
    means = np.mean(simulation_data, axis=0)
    q1s = np.percentile(simulation_data, 25, axis=0)
    q3s = np.percentile(simulation_data, 75, axis=0)

    plt.figure(figsize=(12, 6))
    plt.boxplot(simulation_data, labels=simulation_data.columns, showfliers=False)

    column_names = list(simulation_data.columns)

    for i in range(len(column_names)):
        plt.text(i + 1.1, means.iloc[i], f'$\mu$: {means.iloc[i]:,.2f}%', verticalalignment='bottom')
        plt.text(i + 1.1, medians[i], f'Q2: {medians[i]:,.2f}%', verticalalignment='bottom')
        plt.text(i + 1.1, q1s[i], f'Q1: {q1s[i]:,.2f}%', verticalalignment='bottom')
        plt.text(i + 1.1, q3s[i], f'Q3: {q3s[i]:,.2f}%', verticalalignment='bottom')

    plt.title('Expected Returns Boxplot')
    plt.ylabel('Return')
    plt.show()


def plot_efficient_frontier(data, rf_annualized, adjusted_weights=None):
    expected_return = lambda w, mean: w.dot(mean)
    var = lambda w, sigma: w.reshape(-1, 1).T @ sigma @ w

    def sharpe(w0, mean, rf, cov):
        E = w0.dot(mean)
        return -((E - rf) / var(w0, cov))

    names = data.columns
    nport = 100000

    mean = data.mean()
    std = data.std()

    cov = data.cov()
    corr = data.corr()

    pesos = np.random.dirichlet(0.1 * np.ones(len(names)), nport)

    rf = rf_annualized / 252
    rend_esperado = expected_return(pesos, mean)
    Var = np.array([var(i, cov) for i in pesos])
    sharpe_ = (rend_esperado - rf / 252) / (Var.T)

    # Portafolio de mínima varianza
    n_assets = len(names)
    minvar = minimize(fun=var,
                      x0=np.ones((n_assets,)) / n_assets,
                      args=(cov,),
                      bounds=((0, 1),) * n_assets,
                      constraints={'type': 'eq', 'fun': lambda pesos: pesos.sum() - 1},
                      tol=1e-16)

    # Portafolio de maximo de sharpe
    maxsharpe = minimize(fun=sharpe,
                         x0=np.ones((n_assets,)) / n_assets,
                         args=(mean, rf / 252, cov,),
                         bounds=((0, 1),) * n_assets,
                         constraints={'type': 'eq', 'fun': lambda pesos: pesos.sum() - 1})

    # Get Metrics
    minvar_x = np.sqrt(var(minvar.x, cov))
    minvar_y = minvar.x.dot(mean)

    maxsharpe_x = np.sqrt(var(maxsharpe.x, cov))
    maxsharpe_y = maxsharpe.x.dot(mean)

    if adjusted_weights is not None:
        adjusted_liquidity_x = np.sqrt(var(adjusted_weights, cov))
        adjusted_liquidity_y = adjusted_weights.T @ data.mean()

    # Plot
    fig = plt.figure(figsize=(12, 6))
    X = np.sqrt(Var)
    Y = rend_esperado
    # Frontera Eficiente
    plt.scatter(X, Y, c=sharpe_[0, :], cmap='Purples')

    # Minima Varianza
    plt.scatter(minvar_x, minvar_y, color='red', s=40)
    plt.text(minvar_x, minvar_y, 'MinVar', fontsize=11)

    # Maximo de Sharpe
    plt.scatter(maxsharpe_x, maxsharpe_y, color='red', s=40)
    plt.text(maxsharpe_x, maxsharpe_y, 'MaxSharpe', fontsize=11)

    # Ajustado por Liquidez
    if adjusted_weights is not None:
        plt.scatter(adjusted_liquidity_x, adjusted_liquidity_y, c='k', s=40)
        plt.text(adjusted_liquidity_x, adjusted_liquidity_y, 'Adjusted Liquidity', fontsize=11)

    plt.xlabel(f'$\sigma$')
    plt.ylabel(f'$\mu$')
    plt.title('Frontera Eficiente en Media Varianza')

    plt.scatter(std, mean, color='blue')
    for i in range(n_assets):
        plt.text(std.values[i], mean.values[i], data.columns[i])
