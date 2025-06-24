#######################################
# @author Michael Kane
# @date 07/06/2025
# Defining functions used to interact and
# retrieve data from the Binance Testnet
#######################################
import pandas as pd
import fetch.connect as connect

def retrieve_dataframe(
        client, # Binance client
        pair, # Trading pair
        kline_period, # kline interval period, e.g. 30m, 1h
        timeframe, # Period into past
        future_window, # How far we look into future to compute the future price
        threshold # The threshold percentage change required
        ):

    # Retrieve klines from binnace client
    klines = client.get_historical_klines(pair, kline_period, timeframe)

    # Convert kline data into a pandas dataframe with these labels
    df = pd.DataFrame( klines, columns=[
        "timestamp", "open", "high","low", "close", "volume", 
        "close_time", "quote_asset_volume", "trades", 
        "taker_buy_base", "taker_buy_quote", "ignore"
        ])
    
    # Ensures "close" label is float valued
    df["close"] = df["close"].astype(float)
    # new column for future price, looking "future_value" into the future
    df["future_price"] = df["close"].shift(-future_window)
    # new column for percentage change
    df["pct_change"] = (df["future_price"] - df["close"]) / df["close"]
    # save those who changed more than 1% as 1 and those who didn't as 0
    df["label"] = (df["pct_change"] >= threshold).astype(int)
    # drop any N/A values
    df.dropna(inplace=True)

    # 1h percentage change
    df["return_1h"] = df["close"].pct_change()
    # Rolling means
    df["rolling_mean_6h"] = df["close"].rolling(6).mean()
    df["rolling_std_6h"] = df["close"].rolling(6).std()
    df.dropna(inplace=True) # Drops any na values

    return df