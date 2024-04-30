import yfinance as yf
import pandas as pd

tickers_NYSE=['AAPL', 'MSFT', 'AMD', 'HD', 'HPQ', 'HSBC']

prices_NYSE=yf.download(tickers_NYSE, start='2020-01-01', progress=False)['Adj Close']
volume_NYSE=yf.download(tickers_NYSE, start='2020-01-01', progress=False)['Volume']
benchmark_NYSE=yf.download('SPY', start='2020-01-01', progress=False)['Adj Close']
returns_NYSE=prices_NYSE.pct_change().dropna()


to_date = lambda x: pd.to_datetime(x, format='%Y-%m-%d')
prices_SIC=pd.read_excel('SIC_data/prices.xlsx')
prices_SIC['Date']=[to_date(i) for i in prices_SIC['Date']]
prices_SIC.set_index('Date', inplace=True)
volume_SIC=pd.read_excel('SIC_data/volume.xlsx', index_col=0)
benchmark_SIC=yf.download('SPY.MX', start='2020-01-01', progress=False)['Adj Close']
returns_SIC=prices_SIC.pct_change().dropna()

simulation_data=pd.read_excel('SIC_data/results.xlsx', index_col=0)

sharpe_results=pd.read_excel('SIC_data/sharpe_results.xlsx', index_col=0).T
adjusted_sharpe=pd.read_excel('SIC_data/AbL_results.xlsx', index_col=0).T

sharpe_mean=sharpe_results.pct_change().dropna().mean(axis=0)*25200
adjusted_sharpe_mean=adjusted_sharpe.pct_change().dropna().mean(axis=0)*25200

expected_return=pd.DataFrame()
expected_return['Sharpe Ratio']= sharpe_mean.values
expected_return['Adjusted by Liquidity']=adjusted_sharpe_mean.values

sharpe_mean=(sharpe_results.iloc[-1, :]/1000000 - 1)/len(sharpe_results)*25200
adjusted_sharpe_mean=(adjusted_sharpe.iloc[-1, :]/1000000 - 1)/len(sharpe_results)*25200

total_return=pd.DataFrame()
total_return['Sharpe Ratio']= sharpe_mean.values
total_return['Adjusted by Liquidity']=adjusted_sharpe_mean.values

daily_return=pd.DataFrame()
daily_return['Sharpe Ratio']=sharpe_results.pct_change().dropna().values.flatten()*25200
daily_return['Adjusted by Liquidity']=adjusted_sharpe.pct_change().dropna().values.flatten()*25200


