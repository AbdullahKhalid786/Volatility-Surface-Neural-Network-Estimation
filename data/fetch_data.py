# data/fetch_data.py

import yfinance as yf
import pandas as pd

def fetch_data(ticker='SPY'):
    sp500 = yf.Ticker(ticker)

    # Get option expiration dates and option chain
    expirations = sp500.options

    # Fetch option chain for the first expiration date
    opt_chain = sp500.option_chain(expirations[0])

    # Separate call and put data
    calls = opt_chain.calls
    puts = opt_chain.puts

    # Add expiration date to both calls and puts DataFrames
    calls['expiration'] = expirations[0]
    puts['expiration'] = expirations[0]

    # Convert to DataFrames and save as CSV
    calls_df = pd.DataFrame(calls)
    puts_df = pd.DataFrame(puts)
    calls_df.to_csv('data/calls_data.csv', index=False)
    puts_df.to_csv('data/puts_data.csv', index=False)

    return calls_df, puts_df
