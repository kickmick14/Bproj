#######################################
# @author Michael Kane
# @date 07/06/2025
# Main script of binance testnet project,
# used to centrally integrate several
# separate components
#######################################
import fetch.connect as connect
import fetch.interact as interact
import functions.shapeData as shaper
import models.buildModel as buildModel
import models.plotModel as plot
import models.analyseModel as analyse
import os
import functions.shapeData as shaper
os.environ["MODEL_NAME"] = MODEL_NAME = "LSTM_mk1" # Model name to be saved
from settings import ARTIFACTS_DIR, DATA_DIR, CREDENTIALS_PATH

test_client = connect.client(CREDENTIALS_PATH, "test")  # Test account client
main_client = connect.client(CREDENTIALS_PATH, "main")  # Main account client
test_account = test_client.get_account()                # Get test account information

df_options = {                         # df options for retrieving klines
    "client": main_client,             # Client to use
    "pair": "ETHUSDT",                 # Pair to trade
    "kline_period": "2h",             # Period of klines
    "timeframe": "30 days ago UTC",  # Timeframe of kline data
    "future_window": 5,                # How far into future to consider for pct change
}

klines_df = interact.retrieve_market_data(**df_options)                             # Use the defined options to retrieve dataframe of binance data

df       = shaper.indicators(klines_df, threshold=0.01, RSI_window=14, ATR_window=14)          # Add the indicators and labels to the dataframe
features = df[["macd_signal", "bb_mid", "bb_lower", "bb_upper", "rsi", "atr", "obv"]]   # "Training data"
labels   = df["label"]                                              # Binary classifier

x_train, x_test, y_train, y_test = buildModel.splitData(features, labels)   # Split data for training and testing
# Set timesteps (e.g., past 12 observations per prediction)

print((labels==1).sum())
print((labels==0).sum())
print( ((labels==1).sum()) / ((labels==0).sum()) )

x_train, y_train = shaper.reshape_for_lstm(x_train, y_train, 12)
x_test, y_test   = shaper.reshape_for_lstm(x_test, y_test, 12)

print(x_test.shape)