#######################################
# @author Michael Kane
# @date 27/06/2025
# Code for shaping data inputs
#######################################
import numpy as np
import pandas as pd

def indicators(
        df,             # Intake dataframe
        threshold,       # Percentage threshold
        RSI_window,
        ATR_window
        )->pd.DataFrame:
    
    df["pct_change"] = (df["future_price"] - df["close"]) / df["close"]     # New column for percentage change
    df["label"] = (df["pct_change"] >= threshold).astype(int)               # Save those who changed more than 1% as 1 and those who didn't as 0

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
    avg_gain  = gain.rolling(window=RSI_window).mean()
    avg_loss  = loss.rolling(window=RSI_window).mean()
    rs        = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # 4) ATR (14-period) - Average True Range
    high_low     = df["high"] - df["low"]
    high_close   = (df["high"] - df["close"].shift()).abs()
    low_close    = (df["low"]  - df["close"].shift()).abs()
    true_range   = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"]    = true_range.rolling(window=ATR_window).mean()

    # 5) OBV - Ob-Balance Volume
    df["obv"] = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()

    df.dropna(inplace=True)    # Drops any na values

    return df


def reshape_for_lstm(x, y, timesteps):
    x_lstm, y_lstm = [], []
    for i in range(len(x) - timesteps):
        x_lstm.append(x[i:(i + timesteps)])
        y_lstm.append(y[i + timesteps])
    # Verify shapes
    return np.array(x_lstm), np.array(y_lstm)