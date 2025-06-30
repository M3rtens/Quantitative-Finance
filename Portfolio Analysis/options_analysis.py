import yfinance as yf

def getOptionsDates(ticker):
    ticker = yf.Ticker(ticker)

    expirations = ticker.options
    return expirations

# Gets stike, options price, volume, open interest, implied volatility
def getOptionsData(ticker, target_date):
    currentStockPrice = yf.download(ticker)['Close'].iloc[-1][ticker]
    ticker = yf.Ticker(ticker)

    expirations = ticker.options

    # Go through each possible options date
    final_calls = []
    final_puts = []
    for date in expirations:
        if date == target_date:    
            option_chain = ticker.option_chain(date)
            calls = option_chain.calls
            puts = option_chain.puts

            count = 0
            prev_strike = 0
            for _, row in calls.iterrows():
                strike_price = row['strike']

                if (currentStockPrice == strike_price) or (count > 0 and prev_strike < currentStockPrice and strike_price > currentStockPrice):
                    target_index = count

                prev_strike = strike_price
                count += 1

            upper_display_count = 4
            lower_display_count = 5
            if upper_display_count - 1 > target_index:
                upper_display_count = target_index + 1
            # target_index = 7, count = 9
            if target_index + lower_display_count > count - 1:
                lower_display_count = count - target_index - 1 

            count_2 = 0 
            for _, row in calls.iterrows():
                if count_2 >= target_index - lower_display_count and count_2 <= target_index + upper_display_count:
                    final_calls.append([row['strike'], row['lastPrice'], row['volume'], row['openInterest'], row['impliedVolatility']])

                count_2 += 1
            
            count_2 = 0
            for _, row in puts.iterrows():
                if count_2 >= target_index - lower_display_count and count_2 <= target_index + upper_display_count:
                    final_puts.append([row['strike'], row['lastPrice'], row['volume'], row['openInterest'], row['impliedVolatility']])

                count_2 += 1

    return final_calls, final_puts
