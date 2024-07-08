import pandas as pd   
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

start = pd.to_datetime('2018-01-01')
end = pd.to_datetime('today')

# Adjusted closing price using yfinance
aapl = yf.download('AAPL', start=start, end=end)['Adj Close']
cisco = yf.download('CSCO', start=start, end=end)['Adj Close']
ibm = yf.download('IBM', start=start, end=end)['Adj Close']
amzn = yf.download('AMZN', start=start, end=end)['Adj Close']

# Print the head of each dataframe to verify
print(aapl.head())
print(cisco.head())
print(ibm.head())
print(amzn.head())


stocks = pd.concat([aapl, cisco, ibm, amzn], axis = 1)

stocks.columns = ['aapl', 'cisco', 'ibm', 'amzn']

print(stocks.pct_change(1).mean())


#correlation

print('correlation')
print(stocks.pct_change(1).corr())

print('stocks')
print(stocks)

print('stocks shifted one')
print(stocks.shift(1))

# print('divided')

# print(stocks/stocks.shift(1))

log_returns = np.log(stocks/stocks.shift(1))

print(log_returns.head())

weights = np.random.random(4)

# rebalance the weights
weights /= np.sum(weights)


# expected portfolio returns

exp_ret = np.sum((log_returns.mean() * weights) * 252)

exp_vola = np.sqrt(np.dot(weights.T, np.dot(log_returns.cov()*252, weights)))

# sharpe ratio

sharpe = exp_ret/exp_vola

print(sharpe)

