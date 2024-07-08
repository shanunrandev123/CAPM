import pandas as pd   
import numpy as np
import matplotlib.pyplot as plt
import quandl


start = pd.to_datetime('2012-01-01')
end = pd.to_datetime('2017-01-01')

#adjusted closing price
aapl = quandl.get('WIKI/AAPL.11',start_date=start,end_date=end)

cisco = quandl.get('WIKI/CSCO.11',start_date=start,end_date=end)

ibm = quandl.get('WIKI/IBM.11',start_date=start,end_date=end)

amzn = quandl.get('WIKI/AMZN.11',start_date=start,end_date=end)


# print(aapl.head())



for stock_df in (aapl,cisco,ibm,amzn):
    stock_df['Normed Return'] = stock_df['Adj. Close']/stock_df.iloc[0]['Adj. Close']
    

#allocating 30% to apple, 20% to cisco, 40% to amazon, 10% to ibm

for stock_df, allo in zip((aapl,cisco,ibm,amzn),[.3,.2,.4,.1]):
    stock_df['Allocation'] = stock_df['Normed Return']*allo
    
# print('aapl with allocation')
    
# print(aapl.tail())


# print('position values')

for stock_df in (aapl,cisco,ibm,amzn):
    stock_df['Position Values'] = stock_df['Allocation']*1000000
    
# print(aapl.head())
# print(cisco.head())




portfolio_val = pd.concat([aapl['Position Values'],cisco['Position Values'],ibm['Position Values'],amzn['Position Values']],axis=1)
portfolio_val.columns = ['AAPL Pos','CISCO Pos','IBM Pos','AMZN Pos']

portfolio_val['Total Pos'] = portfolio_val.sum(axis=1)


portfolio_val['Daily_return'] = portfolio_val['Total Pos'].pct_change(1)


print('cumm_ret')
cumulative_return = 100 * (portfolio_val['Total Pos'][-1]/portfolio_val['Total Pos'][0] - 1)


## calculating sharpe ratio

sharpe_ratio = portfolio_val['Daily_return'].mean() / portfolio_val['Daily_return'].std()

print(sharpe_ratio)

# calculating annual sharpe ratio which has 252 business days
annual_sharpe_ratio = (252 ** 0.5) * sharpe_ratio


#ASR > 1 is good for portfolio holdings



