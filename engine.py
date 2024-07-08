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


np.random.seed(42)

num_ports = 100
all_weights = np.zeros((num_ports, len(stocks.columns)))
ret_arr = np.zeros(num_ports)
vol_arr = np.zeros(num_ports)

sharpe_arr = np.zeros(num_ports)


for ind in range(num_ports):
    weights = np.array(np.random.random(4))

    # rebalance the weights
    weights /= np.sum(weights)
    
    all_weights[ind, :] = weights


    # expected portfolio returns

    ret_arr[ind] = np.sum((log_returns.mean() * weights) * 252)

    vol_arr[ind] = np.sqrt(np.dot(weights.T, np.dot(log_returns.cov()*252, weights)))
    
    sharpe_arr[ind] = ret_arr[ind]/vol_arr[ind]

# sharpe ratio

# sharpe = exp_ret/exp_vola

print(sharpe_arr)

print('max sharpe ratio')
print(sharpe_arr.max())

