# import requests
# from bs4 import BeautifulSoup
import time
import statistics
import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm
import matplotlib.pyplot as plt
from fredapi import Fred

FRED_API_KEY = "85304a96fc1ffacd6c24c76004fd49be"
fred = Fred(api_key=FRED_API_KEY)

bullMarkets = [[dt.datetime(1966, 10, 7), dt.datetime(1968, 11, 29)],[dt.datetime(1970, 5, 26), dt.datetime(1973, 1, 11)],[dt.datetime(1974, 10, 3), dt.datetime(1980, 11, 28)],
               [dt.datetime(1982, 8, 12), dt.datetime(1987, 8, 25)], [dt.datetime(1987, 12, 4), dt.datetime(2000, 3, 24)],[dt.datetime(2002, 10, 9), dt.datetime(2007, 10, 9)],
               [dt.datetime(2009, 3, 9), dt.datetime(2020, 2, 19)],[dt.datetime(2020, 3, 23), dt.datetime(2022, 1, 3)], [dt.datetime(2022, 10, 13), dt.datetime(2025, 3, 1)]]

# Excluded [dt.datetime(2020, 2, 19), dt.datetime(2020, 3, 23)] because the timeframe for it is too short to calculate beta on a monthly returns basis
bearMarkets = [[dt.datetime(1968, 11, 29), dt.datetime(1970, 5, 26)],[dt.datetime(1973, 1, 11), dt.datetime(1974, 10, 3)],[dt.datetime(1980, 11, 28), dt.datetime(1982, 8, 12)],
               [dt.datetime(1987, 8, 25), dt.datetime(1987, 12, 4)],[dt.datetime(2000, 3, 24), dt.datetime(2002, 10, 9)],[dt.datetime(2007, 10, 9), dt.datetime(2009, 3, 9)],
               [dt.datetime(2022, 1, 3), dt.datetime(2022, 10, 12)]]

# Get stock tickers from American stocks
# def getUSAStockTickers():
#     url = "https://stockanalysis.com/stocks/"
#     response = requests.get(url)

#     if response.status_code == 200:

#         soup = BeautifulSoup(response.content, 'html.parser')
#         tickers = soup.find_all('a', class_= 'symbol')                                        # have to change this

#         for ticker in tickers:
#             print(ticker.text.strip())
#     else:
#         print(f"Failed to retrieve the page. Status code: {response.status_code}")

# Get S&P500 tickers
def getSPXtickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)                                                                  # Stores all the tables on the site
    sp500_table = tables[0]                                                                     # Stores the first table
    tickers = sp500_table["Symbol"].tolist()                                                    # Gets all the symbols/tickers from the first table
    return tickers

# Obtains the data(close) of a stock(s)
def getData(tickers, start, end):
    stockData = yf.download(tickers, start=start, end=end)['Close']
    return stockData

def getDailyBeta(stocks, start, end):
    # Get the data for the monthly returns of the stock
    stockData = getData(stocks, start, end)
    stockReturnsData = stockData.pct_change().dropna()

    # '%GSPC' = S&P 500 Ticker
    # Get the data for the monthly returns of the market (S&P 500)
    marketData = getData("^GSPC", start, end)
    marketReturnsData = marketData.pct_change().dropna()

    if stockReturnsData.empty or marketReturnsData.empty:
        print(f"Skipping {stocks} due to missing data")
        return None
    
    if marketReturnsData.var().sum() == 0:
        print(f"Skipping {stocks} due to no variance in market returns")
        return None

    common_dates = stockReturnsData.index.intersection(marketReturnsData.index)

    stockReturnsData = stockReturnsData.loc[common_dates]
    marketReturnsData = marketReturnsData.loc[common_dates]

    X = sm.add_constant(marketReturnsData)
    model = OLS(stockReturnsData, X).fit()

    beta = model.params.iloc[1]
    # alpha = model.params.iloc[0]
    # r_squared = model.rsquared
    # p_value = model.pvalues.iloc[1]
    # std_err = model.bse.iloc[1]
    
    return beta

def getMonthlyBeta(stocks, start, end):
    # Get the data for the monthly returns of the stock
    stockData = getData(stocks, start, end)
    monthlyStockData = stockData.resample('ME').last()                                          # ME = Month
    stockReturnsData = monthlyStockData.pct_change().dropna()

    # '%GSPC' = S&P 500 Ticker
    # Get the data for the monthly returns of the market (S&P 500)
    marketData = getData("^GSPC", start, end)
    monthlyMarketData = marketData.resample('ME').last()                                        # ME = Month
    marketReturnsData = monthlyMarketData.pct_change().dropna()

    X = sm.add_constant(marketReturnsData)

    model = OLS(stockReturnsData, X).fit()

    beta = model.params.iloc[1]
    # alpha = model.params.iloc[0]
    # r_squared = model.rsquared
    # p_value = model.pvalues.iloc[1]
    # std_err = model.bse.iloc[1]
    
    return beta

# Get the bull and bear market periods of which there is data available for a specific stock
def getOverlappingDailyPeriods(ticker):
    stock = yf.Ticker(ticker)
    data = stock.history(period="max")["Close"]
    
    if data.empty:
        print(f"Warning: No data available for {ticker}")
        return [], []

    stock_start_date = data.index.min().replace(tzinfo=None)                            # Remove timezone
    stock_end_date = data.index.max().replace(tzinfo=None)                              # Remove timezone

    # period[0] = start, period[1] = end
    overlappingBullMarkets = []
    for period in bullMarkets:
        if stock_start_date <= period[0] and stock_end_date >= period[1]:
            overlappingBullMarkets.append(period)

    overlappingBearMarkets = []
    for period in bearMarkets:
        if stock_start_date <= period[0] and stock_end_date >= period[1]:
            overlappingBearMarkets.append(period)
  
    return overlappingBullMarkets, overlappingBearMarkets

# Get the bull and bear market periods of which there is data available for a specific stock
def getOverlappingMonthlyPeriods(ticker):
    stock = yf.Ticker(ticker)
    data = stock.history(period="max")["Close"]
    
    if data.empty:
        print(f"Warning: No data available for {ticker}")
        return [], []

    monthly_data = data.resample('ME').last().dropna()
    stock_start_date = monthly_data.index.min().replace(tzinfo=None)                            # Remove timezone
    stock_end_date = monthly_data.index.max().replace(tzinfo=None)                              # Remove timezone

    # period[0] = start, period[1] = end
    overlappingBullMarkets = []
    for period in bullMarkets:
        if stock_start_date <= period[0] and stock_end_date >= period[1]:
            overlappingBullMarkets.append(period)

    overlappingBearMarkets = []
    for period in bearMarkets:
        if stock_start_date <= period[0] and stock_end_date >= period[1]:
            overlappingBearMarkets.append(period)
  
    return overlappingBullMarkets, overlappingBearMarkets

# Gets conditional betas of the stocks in S&P 500
def getConditionalBeta(ticker):
    bullMarketPeriods, bearMarketPeriods = getOverlappingDailyPeriods(ticker)

    # Get beta for each bull and bear market
    bullMarketBetaList = []
    for start, end in bullMarketPeriods:
        beta = getDailyBeta(ticker, start, end)
        if beta is not None:  # Only add valid beta values
            bullMarketBetaList.append(beta)

    bearMarketBetaList = []
    for start, end in bearMarketPeriods:
        beta = getDailyBeta(ticker, start, end)
        if beta is not None:  # Only add valid beta values
            bearMarketBetaList.append(beta)

    # Check if we have enough data to proceed
    if len(bullMarketBetaList) == 0 or len(bearMarketBetaList) == 0:
        return []

    # Calculate statistics for bull market betas
    bullMarketBetaAverage = statistics.mean(bullMarketBetaList) if bullMarketBetaList else None
    bullMarketBetaMedian = statistics.median(bullMarketBetaList) if bullMarketBetaList else None
    
    # Only calculate variance and std if we have more than 1 data point
    if len(bullMarketBetaList) > 1:
        x = np.array(bullMarketBetaList)
        bullMarketBetaVariance = np.var(x)
        bullMarketBetaStdDev = np.std(x, ddof=1)
    else:
        bullMarketBetaVariance = 0  # or None
        bullMarketBetaStdDev = 0    # or None
    
    # Calculate statistics for bear market betas
    bearMarketBetaAverage = statistics.mean(bearMarketBetaList) if bearMarketBetaList else None
    bearMarketBetaMedian = statistics.median(bearMarketBetaList) if bearMarketBetaList else None
    
    # Only calculate variance and std if we have more than 1 data point
    if len(bearMarketBetaList) > 1:
        x = np.array(bearMarketBetaList)
        bearMarketBetaVariance = np.var(x)
        bearMarketBetaStdDev = np.std(x, ddof=1)
    else:
        bearMarketBetaVariance = 0  # or None
        bearMarketBetaStdDev = 0    # or None

    return  bullMarketBetaList, bullMarketBetaAverage, bullMarketBetaMedian, bullMarketBetaVariance, bullMarketBetaStdDev, bearMarketBetaList, bearMarketBetaAverage, bearMarketBetaMedian, bearMarketBetaVariance, bearMarketBetaStdDev

# 
def getConditionalBetasSPX():
    stockList = []
    
    for ticker in getSPXtickers():
        changed_ticker = ""

        # Change the . to - so to fit yfinance's format
        changed_ticker = ticker.replace(".", "-")

        print(changed_ticker)
        beta_data = getConditionalBeta(changed_ticker)
        if not beta_data or len(beta_data) == 0:
            print(f"Skipping {changed_ticker} due to insufficient data")
            continue

        # Make the conditions customizable by input from the user
        # For example, bull beta average > 1 and bull data > 3 or bullMarketBetaAverage > bearMarketBetaAverage

        bullMarketBetaAverage = beta_data[1]
        bearMarketBetaAverage = beta_data[6]

        if bullMarketBetaAverage is None or bearMarketBetaAverage is None:
            print(f"Skipping {changed_ticker} due to missing average values")
            continue

        print(bullMarketBetaAverage, bearMarketBetaAverage)

        if bullMarketBetaAverage > 1 and bearMarketBetaAverage < 1:
            stockList.append([changed_ticker, getConditionalBeta(changed_ticker)])
        
        time.sleep(1)

    for stock in stockList:
        print(stock[0])
        print(f"Betas during bull markets: {stock[1][0]}")
        print(f"Beta average: {stock[1][1]}")
        print(f"Beta median: {stock[1][2]}")
        print(f"Beta variance: {stock[1][3]}")
        print(f"Beta standard deviation: {stock[1][4]}")

        print(f"Betas during bear markets: {stock[1][5]}")
        print(f"Beta average: {stock[1][6]}")
        print(f"Beta median: {stock[1][7]}")    
        print(f"Beta variance: {stock[1][8]}")
        print(f"Beta standard deviation: {stock[1][9]}")

# Calculates the expected return of a stock
# Uses the Arbitrage Pricing Theory to calculate the expected return of each stock
def getStockExpectedReturn(ticker):
    # macroeconomic factors to calculate 
    factors = {
        "Market Returns": "SP500",  # S&P 500 (Market return)            
        "Inflation": "CPIAUCSL",  # CPI (Inflation)
        "Interest Rate": "DGS10",  # 10-year Treasury yield
        "Industrial Production": "INDPRO", # Industrial production
        "Oil Price": "DCOILWTICO" # WTI Crude Oil Price
    }

    endDate = dt.datetime.now()
    startDate = endDate - dt.timedelta(days=365*5)

    stockData = getData(ticker, startDate, endDate)
    stock_monthly = stockData.resample('ME').last()
    stock_returns = stock_monthly.pct_change().dropna()

    market_returns = fred.get_series(factors["Market Returns"]).resample("ME").last().pct_change()
    inflation = fred.get_series(factors["Inflation"]).resample("ME").last().pct_change()
    interest_rate = fred.get_series(factors["Interest Rate"]).resample("ME").last().diff()
    industrial_production = fred.get_series(factors["Industrial Production"]).resample("ME").last().pct_change()
    oil_price = fred.get_series(factors["Oil Price"]).resample("ME").last().pct_change()
    risk_free_rate = fred.get_series("DGS3MO").iloc[-1]

    data = pd.concat([stock_returns, market_returns, inflation, interest_rate, industrial_production, oil_price], axis = 1)
    data.columns = ["Stock Returns", "Market Returns", "Inflation", "Interest Rate", "Industrial Production", "Oil Price"]
    data = data.dropna()

    x = data[["Market Returns", "Inflation", "Interest Rate", "Industrial Production", "Oil Price"]]
    x = sm.add_constant(x)
    y = data["Stock Returns"]

    model = sm.OLS(y, x).fit()
    betas = model.params[1:]

    # Obtain risk premiums
    # SP500_betas = []
    # SP500_avg_returns = []
    # for ticker in getSPXtickers():
    #     changed_ticker = ""

    #     # Change the . to - so to fit yfinance's format
    #     changed_ticker = ticker.replace(".", "-")
    #     stockData2 = getData(changed_ticker, startDate, endDate)

    #     if len(stockData2) == 1256:
    #         stock_monthly2 = stockData2.resample("ME").last().pct_change().dropna()   
    #         combined = pd.concat([stock_monthly2, market_returns, inflation, interest_rate, industrial_production, oil_price], axis = 1).dropna()
    #         combined.columns = ["Stock Returns", "Market Returns", "Inflation", "Interest Rate", "Industrial Production", "Oil Price"]

    #         x2 = combined[["Market Returns", "Inflation", "Interest Rate", "Industrial Production", "Oil Price"]]
    #         x2 = sm.add_constant(x2)
    #         y = combined["Stock Returns"]

    #         model = sm.OLS(y, x).fit()
    #         betas = model.params[1:]
    #         SP500_betas.append(betas.values)
    #         SP500_avg_returns.append(y.mean())

    # x_cross = sm.add_constant(np.array(SP500_betas))
    # y_cross = np.array(SP500_avg_returns)

    # model_cross = sm.OLS(y_cross, x_cross).fit()
    # risk_premiums = model_cross.params[1:]
    
    # print(list(zip(["Market Returns", "Inflation", "Interest Rate", "Industrial Production", "Oil Price"], risk_premiums)))
    # print(model_cross.summary())
    # print(data[["Market Returns", "Inflation", "Interest Rate", "Industrial Production", "Oil Price"]].corr())

    # risk_premiums as of 16/05/2025
    risk_free_rate = risk_free_rate / 100
    monthly_risk_free_rate = (1 + risk_free_rate) ** (1 / 12) - 1

    risk_premiums = [0.00658536244823475, -0.00029843717716376795, 0.04996264357867247, 0.0006180019733621838, 0.0012506047042368934]

    monthly_return = monthly_risk_free_rate + sum(betas[j] * risk_premiums[j] for j in range(len(betas)))
    annualized_return = (1 + monthly_return) ** 12 - 1

    # print("Risk premiums:\n", risk_premiums)
    # print("Betas:\n", betas)
    # print(f"Expected Return of {ticker}: {annualized_return * 100:.2f}%")
    return monthly_return, annualized_return

# Gets the sharpe ratio of a stock
def getSharpeSortinoRatio(ticker):
    endDate = dt.datetime.now()
    startDate = endDate - dt.timedelta(days=365*5)

    stockData = getData(ticker, startDate, endDate)
    stock_monthly = stockData.resample('ME').last()
    stock_returns = stock_monthly.pct_change().dropna()

    risk_free_rates = fred.get_series("DGS3MO").resample("ME").last().dropna()
    risk_free_rates = risk_free_rates / 100
    monthly_risk_free_rates = (1 + risk_free_rates) ** (1 / 12) - 1

    data = pd.concat([stock_returns, monthly_risk_free_rates], axis=1)
    data.columns = ["Stock Returns", "Risk-Free Rate"]
    data = data.dropna()

    excess_returns = data["Stock Returns"] - data["Risk-Free Rate"]

    avg_excess_return = excess_returns.mean()
    std_dev = np.std(excess_returns, ddof = 1)

    sharpe_ratio = avg_excess_return / std_dev
    annualized_sharpe = sharpe_ratio * np.sqrt(12)
    annualized_sharpe = round(annualized_sharpe, 4) 

    negative_excess_returns = [ (row["Stock Returns"] - row["Risk-Free Rate"]) for _, row in data.iterrows() if row["Stock Returns"] - row["Risk-Free Rate"] < 0 ]
    negative_std_dev = np.std(negative_excess_returns, ddof = 1)

    sortino_ratio = avg_excess_return / negative_std_dev
    annualized_sortino = sortino_ratio * np.sqrt(12)
    annualized_sortino = round(annualized_sortino, 4)

    # print(f"Sharpe Ratio: {annualized_sharpe}")
    # print(f"Sortino Ratio: {annualized_sortino}")

    return annualized_sharpe, annualized_sortino

# Calculates the expected return of the portfolio
# Forecast the growth of a portfolio using: w1*r1 + w2*r2 + wn*rn ...
# Use the Arbitrage Pricing Theory for the return of each stock
def getPortfolioExpectedReturn(tickers, weights):
    total_weight = 0
    for weight in weights:
        total_weight += weight

    if total_weight != 1:
        return "Weights do not equal to 1"
    
    portfolio_expected_return = 0
    for ticker, weight in zip(tickers, weights):
        _, annual_return = getStockExpectedReturn(ticker)
        annual_return = (annual_return * 100) * weight

        portfolio_expected_return += annual_return

    portfolio_expected_return = round(portfolio_expected_return, 2)    
    
    return portfolio_expected_return

def modernPortfolioTheory(tickers):
    # get linear annual returns (linear returns instead of geometric for MPT) for each stock
    annual_returns = pd.Series(dtype='float')
    for ticker in tickers:
        linear_annual_return, _ = getStockExpectedReturn(ticker)
        linear_annual_return *= 12

        annual_returns[ticker] = linear_annual_return

    # obtain the covariance matrix for the stocks
    endDate = dt.datetime.now()
    startDate = endDate - dt.timedelta(days=365*5)

    stocksData = yf.download(tickers, startDate, endDate)['Close']
    returns = stocksData.pct_change().dropna()
    cov_matrix = returns.cov() * 252

    # Set number of iterations and initialize array with 3 rows and num_iteration columns full of 0s  
    num_iterations = 50000
    num_tickers = len(tickers)

    results = np.zeros((3, num_iterations))
    weights_list = []
    for i in range(num_iterations):
        weights = np.random.random(num_tickers)
        weights /= np.sum(weights)
        weights_list.append(weights)

        portfolio_return = np.dot(weights, annual_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = portfolio_return / portfolio_volatility

        # print(portfolio_return)
        results[0, i] = portfolio_return * 100
        results[1, i] = portfolio_volatility
        results[2, i] = sharpe_ratio

    weights_array = np.array(weights_list)
    max_sharpe_idx = results[2].argmax()
    max_sharpe_weights = weights_array[max_sharpe_idx]
    max_sharpe_portfolio = dict(zip(tickers, max_sharpe_weights))

    print(max_sharpe_portfolio)

    results_df = pd.DataFrame(results.T, columns=["Return", "Risk", "Sharpe"])
    plt.figure(figsize=(10, 6))
    plt.scatter(results_df["Risk"], results_df["Return"], c=results_df["Sharpe"], cmap="viridis", alpha=0.7)
    plt.colorbar(label="Sharpe Ratio")
    plt.xlabel("Risk (Standard Deviation)")
    plt.ylabel("Expected Return")
    plt.title("Efficient Frontier")
    plt.scatter(results[1, max_sharpe_idx], results[0, max_sharpe_idx],
                color='red', marker='*', s=100, label='Max Sharpe Ratio')
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig("plot.png")

    return max_sharpe_portfolio
