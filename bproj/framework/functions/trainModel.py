#######################################
# @author Michael Kane
# @date 07/06/2025
# Script to train and save new models
#######################################
import API.fetch.connect as connect
import API.fetch.interact as interact
import API.functions.shapeData as shaper
import framework.functions.configureModel as configure
import framework.functions.plotModel as plot
import framework.functions.analyseModel as analyse
import framework.models.LSTM as framework
import os
os.environ["MODEL_NAME"] = MODEL_NAME = "LSTM_mk1" # Model name to be saved
from settings import ARTIFACTS_DIR, DATA_DIR, CREDENTIALS_PATH

test_client = connect.client(CREDENTIALS_PATH, "test")  # Test account client
main_client = connect.client(CREDENTIALS_PATH, "main")  # Main account client
test_account = test_client.get_account()                # Get test account information

gpus = configure.gpuConfig()          # Configure gpus
print(f"gpus: {gpus}")

df_options = {                         # df options for retrieving klines
    "client": main_client,             # Client to use
    "pair": "ETHUSDT",                 # Pair to trade
    "kline_period": "1h",              # Period of klines
    "timeframe": "720 days ago UTC",   # Timeframe of kline data
    "future_window": 5,               # How far into future to consider for pct change
}

klines_df = interact.retrieve_market_data(**df_options)     # Use the defined options to retrieve dataframe of binance data

threshold = 0.01
RSI_window = 14
ATR_window = 14
test_split = 0.3
df       = shaper.indicators(klines_df, threshold, RSI_window, ATR_window)              # Add the indicators and labels to the dataframe
features = df[["macd_signal", "bb_mid", "bb_lower", "bb_upper", "rsi", "atr", "obv"]]   # "Training data"
labels   = df["label"]                                                                  # Binary classifier
x_train, x_test, y_train, y_test = configure.splitData(features, labels, test_split)   # Split data for training and testing

# Set timesteps
timesteps = 48
# Reshape for LSTM
x_train, y_train = shaper.reshape_for_lstm(x_train, y_train, timesteps)
x_test, y_test   = shaper.reshape_for_lstm(x_test, y_test, timesteps)

epochs           = 10
batch_size       = 128
validation_split = 0.3
model, history   = framework.LSTM(x_train, x_test, y_train, y_test, timesteps, validation_split, epochs, batch_size)   # Train model on the split data

plot.history(history)  # Plot accuracy
plot.loss(history)     # Plot loss

analyse.evaluation(model, x_test, y_test)

y_pred, y_pred_labels = configure.model_predict(model, x_test)
fpr, tpr, auc, thresholds = analyse.AUCandROCcurve(y_test, y_pred)
plot.ROC_curve(fpr, tpr, auc)

analyse.classification(y_test, y_pred_labels)   # Create classification report
analyse.confusion(y_test, y_pred_labels)        # Create confusion matrix