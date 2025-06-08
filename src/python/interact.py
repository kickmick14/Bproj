#######################################
# @author Michael Kane
# @date 07/06/2025
# Defining functions used to interact and
# retrieve data from the Binance Testnet
#######################################
import pandas as pd
import connect

def retrieve_dataframe(client, pair, kline_period, timeframe, future_window, threshold):

    klines = client.get_historical_klines(pair, kline_period, timeframe)
    df = pd.DataFrame(klines, columns=[
        "timestamp", "open", "high","low", "close", "volume", 
        "close_time", "quote_asset_volume", "trades", 
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    df["close"] = df["close"].astype(float)

    future_window = 3 # predict 3 hours ahead
    threshold = 0.01 # desirable percentage change

    df["future_price"] = df["close"].shift(-future_window) # create new column for future price
    df["pct_change"] = (df["future_price"] - df["close"]) / df["close"] # create new column for percentage change
    df["label"] = (df["pct_change"] >= threshold).astype(int) # save those who changed more than 1% as 1 and those who didn't as 0
    df.dropna(inplace=True) # drop any N/A values

    df["return_1h"] = df["close"].pct_change()
    df["rolling_mean_6h"] = df["close"].rolling(6).mean()
    df["rolling_std_6h"] = df["close"].rolling(6).std()
    df.dropna(inplace=True)

    return df