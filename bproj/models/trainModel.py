#######################################
# @author Michael Kane
# @date 07/06/2025
# Script to train and save new models
# which can then be loaded into main.py
#######################################
import fetch.connect as connect
import fetch.interact as interact
import functions.shapeData as shaper
import models.buildModel as buildModel
import models.plotModel as plot
import models.analyseModel as analyse
import matplotlib.pyplot as plt
import os
os.environ["MODEL_NAME"] = MODEL_NAME = "LSTM_mk1" # Model name to be saved
from settings import ARTIFACTS_DIR, DATA_DIR, CREDENTIALS_PATH

test_client = connect.client(CREDENTIALS_PATH, "test")  # Test account client
main_client = connect.client(CREDENTIALS_PATH, "main")  # Main account client
test_account = test_client.get_account()                # Get test account information

gpus = buildModel.gpuConfig()      # Configure gpus
print(f"gpus: {gpus}")

df_options = {                         # df options for retrieving klines
    "client": main_client,             # Client to use
    "pair": "ETHUSDT",                 # Pair to trade
    "kline_period": "2h",              # Period of klines
    "timeframe": "30 days ago UTC",   # Timeframe of kline data
    "future_window": 5,                # How far into future to consider for pct change
}

klines_df = interact.retrieve_market_data(**df_options)                             # Use the defined options to retrieve dataframe of binance data

threshold = 0.01
RSI_window = 14
ATR_window = 14
df       = shaper.indicators(klines_df, threshold, RSI_window, ATR_window)              # Add the indicators and labels to the dataframe
features = df[["macd_signal", "bb_mid", "bb_lower", "bb_upper", "rsi", "atr", "obv"]]   # "Training data"
labels   = df["label"]                                              # Binary classifier

x_train, x_test, y_train, y_test = buildModel.splitData(features, labels)   # Split data for training and testing

# Set timesteps
timesteps = 12
# Reshape for LSTM
x_train, y_train = shaper.reshape_for_lstm(x_train, y_train, timesteps)
x_test, y_test   = shaper.reshape_for_lstm(x_test, y_test, timesteps)

validation_split = 0.25
epochs = 50
batch_size = 32
model, history = buildModel.LSTM(x_train, x_test, y_train, y_test, timesteps, validation_split, epochs, batch_size)   # Train model on the split data

plot.history(history)  # Plot accuracy
plot.loss(history)     # Plot loss

analyse.evaluation(model, x_test, y_test)

y_pred, y_pred_labels = buildModel.model_predict(model, x_test)
fpr, tpr, auc, thresholds = analyse.AUCandROCcurve(y_test, y_pred)
plot.ROC_curve(fpr, tpr, auc)

analyse.classification(y_test, y_pred_labels)   # Create classification report
analyse.confusion(y_test, y_pred_labels)        # Create confusion matrix