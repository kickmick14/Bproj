#######################################
# @author Michael Kane
# @date 27/06/2025
# Creation of indicators and functions
# for manipulating them
#######################################
import numpy as np
import pandas as pd

# Get indicator data using market information
def get(
    df: pd.DataFrame,
    threshold: float,
    RSI_window: int,
    ATR_window: int,
    stochastic_window: int,
    CCI_window: int,
    lag_list: list
    ) -> pd.DataFrame:
    """
    Calculates a comprehensive set of technical indicators for trading signal modeling.
    All indicators are shifted by 1 to prevent lookahead bias.

    Indicators added:
        - MACD and MACD signal
        - Bollinger Bands (mid, upper, lower, std)
        - RSI
        - ATR
        - OBV
        - EMA 20/50
        - Stochastic Oscillator (%K)
        - Momentum (10, 20)
        - CCI
        - VWAP (cumulative and rolling 10, 20)
        - Rolling min/max close prixe
        - Z-price
        - Lagged returns
    """

    df["pct_change"] = (df["future_price"] - df["close"]) / df["close"]
    df["label"] = (df["pct_change"] >= threshold).astype(int)

    # MACD (Moving Average Convergence Divergence)
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = (ema12 - ema26).shift(1)
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean().shift(1)

    # Bollinger Bands
    rolling20 = df["close"].rolling(window=20)
    df["bb_mid"] = rolling20.mean().shift(1)
    df["bb_std"] = rolling20.std().shift(1)
    df["bb_upper"] = df["bb_mid"] + 2 * df["bb_std"]
    df["bb_lower"] = df["bb_mid"] - 2 * df["bb_std"]

    # RSI (Relative Strength Index)
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=RSI_window).mean()
    avg_loss = loss.rolling(window=RSI_window).mean()
    rs = avg_gain / avg_loss
    df["rsi"] = (100 - (100 / (1 + rs))).shift(1)

    # ATR (Average True Range)
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = true_range.rolling(window=ATR_window).mean().shift(1)

    # OBV (On-Balance Volume)
    df["obv"] = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()

    # EMA - Exponential Moving Averages
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean().shift(1)
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean().shift(1)

    # Stochastic Oscillator
    lowest_low = df["low"].rolling(window=stochastic_window).min()
    highest_high = df["high"].rolling(window=stochastic_window).max()
    df["stochastic_k"] = 100 * ((df["close"] - lowest_low) / (highest_high - lowest_low)).shift(1)

    # Momentum indicator
    df["momentum_10"] = df["close"].diff(periods=10).shift(1)
    df["momentum_20"] = df["close"].diff(periods=20).shift(1)

    # Commodity Channel Index (CCI)
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    mean_tp = typical_price.rolling(window=CCI_window).mean()
    mad_tp = typical_price.rolling(window=CCI_window).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
    df["cci"] = ((typical_price - mean_tp) / (0.015 * mad_tp)).shift(1)

    # Volume Weighted Average Price (VWAP)
    df["vwap_cumulative"] = (df["volume"] * df["close"]).cumsum() / df["volume"].cumsum()
    df["vwap_10"] = np.where(df["volume"].rolling(10).sum() == 0,
                             np.nan, (df["volume"] * df["close"]).rolling(10).sum() / df["volume"].rolling(10).sum())
    df["vwap_20"] = (df["volume"] * df["close"]).rolling(20).sum() / df["volume"].rolling(20).sum()

    # Rolling min/max
    df["rolling_min_20"] = df["close"].rolling(window=20).min().shift(1)
    df["rolling_max_20"] = df["close"].rolling(window=20).max().shift(1)

    # Z-score
    rolling_mean_20 = df["close"].rolling(window=20).mean()
    rolling_std_20 = df["close"].rolling(window=20).std()
    df["zscore_20"] = ((df["close"] - rolling_mean_20) / rolling_std_20).shift(1)

    # Lagged returns
    for lag in lag_list:
        df[f"return_{lag}"] = df["close"].pct_change(periods=lag).shift(1)

    df.dropna(inplace=True)

    return df


def reshape_for_lstm(
        x,
        y,
        timesteps: int
        )->np.array:

    x_lstm, y_lstm = [], []
    for i in range(len(x) - timesteps):
        x_lstm.append(x[i:(i + timesteps)])
        y_lstm.append(y[i + timesteps])

    x_lstm = np.array(x_lstm)
    y_lstm = np.array(y_lstm)

    # Verify shapes
    print(f"Data shape: {x_lstm.shape}, {y_lstm.shape}")

    return x_lstm, y_lstm