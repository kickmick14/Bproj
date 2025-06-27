#######################################
# @author Michael Kane
# @date 07/06/2025
# Defining functions used to interact and
# retrieve data from the Binance Testnet
#######################################
import pandas as pd
import numpy as np
import fetch.connect as connect

def retrieve_market_data(
        client,             # Binance client
        pair,               # Trading pair
        kline_period,       # kline interval period, e.g. 30m, 1h
        timeframe,          # Period into past
        future_window,       # Klines into future
        )->pd.DataFrame:

    # Retrieve klines from binnace client
    klines = client.get_historical_klines(pair, kline_period, timeframe)

    # Convert kline data into a pandas dataframe with these labels
    df = pd.DataFrame( klines, columns=[
        "timestamp", "open", "high","low", "close", "volume", 
        "close_time", "quote_asset_volume", "trades", 
        "taker_buy_base", "taker_buy_quote", "ignore"
        ])
    
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    
    df["close"] = df["close"].astype(float)                 # Ensures "close" label is float valued
    df["future_price"] = df["close"].shift(-future_window)  # New column for future price, looking "future_value" into the future
    
    return df