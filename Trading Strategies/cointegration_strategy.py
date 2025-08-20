import yfinance as yf
import pandas as pd
import backtrader as bt 
import statsmodels.api as sm 
import numpy as np

def getData(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # BackTrader likes these columns
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

    data.reset_index(inplace=True)

    # Required column
    data['OpenInterest'] = 0

    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

    return data

# Convert to backtrader's data feed format
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

class PairTradingStrategy(bt.Strategy):
    params = (
        ('lookback', 60),
        ('entry_z', 1.0),
        ('exit_z', 0.5),
    )

    def __init__(self):
        self.data0_close = self.datas[0].close
        self.data1_close = self.datas[1].close
        self.hedge_ratio = None
        self.spread = []
        self.zscores = []
        self.portfolio_value = []
    
    def next(self):
        if len(self) < self.params.lookback:
            return
        
        y = np.array([self.data0_close[-i] for i in range(self.params.lookback)[::-1]])
        x = np.array([self.data1_close[-i] for i in range(self.params.lookback)[::-1]])

        x_ = sm.add_constant(x)
        model = sm.OLS(y, x_).fit()
        self.hedge_ratio = model.params[1]
        print(self.hedge_ratio)

        spread = y - self.hedge_ratio * x
        mean = spread.mean()
        std = spread.std()
        zscore = (spread[-1] - mean) / std

        self.zscores.append(zscore)

        # total_value = self.broker.getvalue()
        # alloc = 0.025

        # cash_per_trade = total_value * alloc
        # price0 = self.datas[0].close[0]
        # price1 = self.datas[1].close[0]

        # size0 = int(cash_per_trade / price0)
        # size1 = int(cash_per_trade/ price1)

        if zscore > self.params.entry_z and not self.position:
            self.buy(data=self.datas[1], size=self.hedge_ratio)
            self.sell(data=self.datas[0])

        elif zscore < -self.params.entry_z and not self.position:
            self.buy(data=self.datas[0])
            self.sell(data=self.datas[1], size=self.hedge_ratio)

        elif abs(zscore) < self.params.exit_z:
            self.close(self.datas[0])
            self.close(self.datas[1])

        self.portfolio_value.append(self.broker.getvalue())

cerebro = bt.Cerebro()
cerebro.addstrategy(PairTradingStrategy)

start = "2018-01-01"
end = "2023-12-31"

data = getData('KO', start, end)
data2 = getData('PEP', start, end)

dataFeed = PandasData(dataname=data) 
dataFeed2 = PandasData(dataname=data2) 

cerebro.adddata(dataFeed)
cerebro.adddata(dataFeed2)

cerebro.broker.setcash(10000)
results = cerebro.run()

# --------------------------- Print the results of the strategy ---------------------------

# Obtain results of the strategy
strat = results[0]

portfolio_series = pd.Series(strat.portfolio_value)
portfolio_returns = portfolio_series.pct_change().dropna()

# Assumptions
risk_free_rate = 0.0
trading_days = 252

# Sharpe Ratio
excess_returns = portfolio_returns - risk_free_rate / trading_days
sharpe_ratio = np.sqrt(trading_days) * excess_returns.mean() / excess_returns.std() 

# Cumulative Return
cumulative_return = (portfolio_series.iloc[-1] / portfolio_series.iloc[0]) - 1

# Max Drawdown
rolling_max = portfolio_series.cummax()
drawdown = portfolio_series / rolling_max - 1
max_drawdown = drawdown.min()

print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Cumulative Return: {cumulative_return:.2%}")
print(f"Max Drawdown: {max_drawdown:.2%}")

cerebro.plot()
