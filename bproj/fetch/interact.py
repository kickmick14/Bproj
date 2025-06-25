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
        future_window       # Klines into future
        )->pd.DataFrame:

    # Retrieve klines from binnace client
    klines = client.get_historical_klines(pair, kline_period, timeframe)

    # Convert kline data into a pandas dataframe with these labels
    df = pd.DataFrame( klines, columns=[
        "timestamp", "open", "high","low", "close", "volume", 
        "close_time", "quote_asset_volume", "trades", 
        "taker_buy_base", "taker_buy_quote", "ignore"
        ])
    
    df["close"] = df["close"].astype(float)                 # Ensures "close" label is float valued
    df["future_price"] = df["close"].shift(-future_window)  # New column for future price, looking "future_value" into the future
    
    return df


def indicator_6hRolling(
        df,             # Intake dataframe
        threshold       # Percentage threshold
        )->pd.DataFrame:
    
    df["pct_change"] = (df["future_price"] - df["close"]) / df["close"]     # New column for percentage change
    df["label"] = (df["pct_change"] >= threshold).astype(int)               # Save those who changed more than 1% as 1 and those who didn't as 0

    df["return_1h"] = df["close"].pct_change()              # 1h percentage change
    df["rolling_mean_6h"] = df["close"].rolling(6).mean()   # Rolling mean
    df["rolling_std_6h"] = df["close"].rolling(6).std()     # Rolling std
    df.dropna(inplace=True)                                 # Drops any na values

    return df


def indicator_Technical(
        df,
        )->pd.DataFrame:
    
    # New indicators:
    # 1) MACD - Moving Average Convergence Divergence
    ema12             = df["close"].ewm(span=12, adjust=False).mean()
    ema26             = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # 2) Bollinger Bands (20-period) - Upper and lower
    rolling20      = df["close"].rolling(window=20)
    df["bb_mid"]   = rolling20.mean()
    df["bb_upper"] = df["bb_mid"] + 2 * rolling20.std()
    df["bb_lower"] = df["bb_mid"] - 2 * rolling20.std()

    # 3) RSI (14-period) - Relative Strength Index
    delta     = df["close"].diff()
    gain      = delta.clip(lower=0)
    loss      = -delta.clip(upper=0)
    avg_gain  = gain.rolling(window=14).mean()
    avg_loss  = loss.rolling(window=14).mean()
    rs        = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # 4) ATR (14-period) - Average True Range
    high_low     = df["high"] - df["low"]
    high_close   = (df["high"] - df["close"].shift()).abs()
    low_close    = (df["low"]  - df["close"].shift()).abs()
    true_range   = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"]    = true_range.rolling(window=14).mean()

    # 5) OBV - Ob-Balance Volume
    df["obv"] = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()

    # Drop any NaNs introduced by the rolling windows
    df.dropna(inplace=True)

    return df