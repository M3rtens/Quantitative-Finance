import yfinance as yf
import pandas as pd
import numpy as np

print("Results from the simple backtesting method:")

# Obtain the daily ticker ohlvc data
# Returns a pandas.DataFrame
data = yf.download("AAPL", start="2020-01-01", end="2023-12-31")

# Calculate the 50 and 200 MA values  
data['SMA_50'] = data['Close'].rolling(50).mean()
data['SMA_200'] = data['Close'].rolling(200).mean()

# Initialize the trade signals to 0 for each ticker
data['Signal'] = 0

# Set 1 for long signal when 50 MA > 200 MA, and -1 for short signals when 50 MA < 200 MA 
data.loc[data['SMA_50'] > data['SMA_200'], 'Signal'] = 1
data.loc[data['SMA_50'] < data['SMA_200'], 'Signal'] = -1

# Obtain the return relative to the previous ticker
data['Return'] = data['Close'].pct_change()

# Signal is shifted by 1 day forward when multiplying with the return
# This represents it so that it takes a trade after the signal and not before
data['Strategy'] = data['Signal'].shift(1) * data['Return']

# 
cumulative_return = (1 + data['Strategy']).cumprod()[-1]-1

# Takes the average of the daily returns and multiply by 252 to get the yearly return in a linear manner
# Returns do not compound this way 
annualized_return = data['Strategy'].mean() * 252

# Risk-To-Reward ratio of the strategy based on the daily average return and standard deviation
# Multiply by the square root of 252 to turn the daily sharpe ratio into a yearly one
sharpe_ratio = (data['Strategy'].mean() / data['Strategy'].std()) * (252**0.5)

# Round to 2 decimal points
cumulative_return = str(round(cumulative_return * 100, 2))
annualized_return = str(round(annualized_return * 100, 2))
sharpe_ratio = str(round(sharpe_ratio, 2))

print("Cumulative Return: " + cumulative_return + "%")
print("Annualized Return: " + annualized_return + "%")
print("Sharpe Ratio: " + sharpe_ratio)


print("\nResults from using the BackTrader library:")

import backtrader as bt

# Obtain ticker data
data = yf.download("AAPL", start="2018-01-01", end="2023-12-31")
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# print(data)

# BackTrader likes these columns
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

data.reset_index(inplace=True)

# Required column
data['OpenInterest'] = 0

data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Convert to backtrader's data feed formaat
class PandasData(bt.feeds.PandasData):
    params = (
        ('datetime', None),
        ('open', 'Open'),
        ('high', 'High'),
        ('low', 'Low'),
        ('close', 'Close'),
        ('volume', 'Volume'),
        ('openinterest', 'OpenInterest'),
    )

# Define the trading strategy
class SMACrossStrategy(bt.Strategy):
    params = (
        ("fast_period", 50),
        ("slow_period", 200),
    )

    def __init__(self):
        self.sma_fast = bt.ind.SMA(period=self.p.fast_period)
        self.sma_slow = bt.ind.SMA(period=self.p.slow_period)
        self.crossover = bt.ind.CrossOver(self.sma_fast, self.sma_slow)

    def next(self):
        if not self.position and self.crossover > 0:
            self.buy()
        elif self.position and self.crossover < 0:
            self.close()

# Cerebro is BackTrader's engine
cerebro = bt.Cerebro()
cerebro.addstrategy(SMACrossStrategy)

# Add the ticker data
dataFeed = PandasData(dataname=data) 
cerebro.adddata(dataFeed)

# Set the starting cash value
cerebro.broker.set_cash(10000)

print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

cerebro.run()

print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

cerebro.plot()