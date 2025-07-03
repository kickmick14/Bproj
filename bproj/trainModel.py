#######################################
# @author Michael Kane
# @date 07/06/2025
# Script to train and save new models
#######################################
import API.fetch.connect as connect
import API.fetch.interact as interact
import API.functions.shapeData as shaper
import framework.functions.configureTraining as configure
import framework.functions.plotModel as plot
import framework.functions.analyseModel as analyse
import framework.models.LSTM as framework
import tensorflow as tf
import numpy as np
import os, uuid, gc
from datetime import datetime
from keras import backend
os.environ["MODEL_NAME"] = MODEL_NAME = "LSTM_Scan3" # Model name to be saved
from settings import DATA_DIR, CREDENTIALS_PATH

# Configure gpus
gpus = configure.gpuConfig()
print(f"gpus: {gpus}")

# Connect to Binance
test_client = connect.client(CREDENTIALS_PATH, "test")
main_client = connect.client(CREDENTIALS_PATH, "main")
test_account = test_client.get_account()

df_options = {                         # df options for retrieving klines
    "pair": "ETHUSDT",                 # Pair to trade
    "kline_period": "2h",              # Period of klines
    "timeframe": "1440 days ago UTC",   # Timeframe of kline data
    "future_window": 10,                # How far into future to consider for pct change
}

# Use the defined options to retrieve dataframe of binance data
klines_df = interact.retrieve_market_data(main_client, **df_options)

indicators_dict = {
    "threshold": 0.01,
    "RSI_window": 14,
    "ATR_window": 14,
    "stochastic_window": 14,
    "CCI_window": 20,
    "lag_list": [1, 5, 10]
    }

# Add the indicators and labels to the dataframe
df = shaper.indicators(klines_df, **indicators_dict)
features = df[
    [ "macd", "macd_signal",
    "bb_mid", "bb_upper", "bb_lower", "bb_std",
    "rsi", "atr", "obv",
    "ema_20", "ema_50",
    "stochastic_k",
    "momentum_10", "momentum_20",
    "cci",
    "vwap_cumulative", "vwap_10", "vwap_20",
    "rolling_min_20", "rolling_max_20",
    "zscore_20",
    *[ f"return_{lag}" for lag in indicators_dict["lag_list"] ]
    ] ]

binance_options = {
     "binance_options":  df_options,
     "features":         list(features.columns),
    }

if not os.path.exists(f"{DATA_DIR}/binance_options.jsonl"):
     configure.log_to_json(f"{DATA_DIR}/binance_options.jsonl", binance_options)

# Binary classifier
labels = df["label"]
# Fraction of data taken for post training testing
test_split  = 0.2
# Split data for training and testing
x_train, x_test, y_train, y_test = configure.splitData(features, labels, test_split)


# Set up unique hash for loop
RUN_ID = uuid.uuid4().hex[:8]
os.environ.setdefault("RUN_ID", str(RUN_ID))
os.makedirs(f"{DATA_DIR}/{RUN_ID}/plots", exist_ok=True)

# Get starting time
now = datetime.now().strftime("%H:%M:%S")

# Defune the systems hyperparameters
hyperparameters = {
    "layer1_units": 64,
    "layer2_units": 8,
    "dropout": 0.30, # Move from 0.1 to 0.5
    "recurrent_dropout": "N/A",
    "kernel_regulariser": 0.02,
    "timesteps": 12,
    "validation_split": 0.3,
    "epochs": 50,
    "batch_size": 48,
    "optimiser": "adam",
    "learning_rate": 0.0005,
    "loss": "binary_crossentropy",
    "patience": 5,
    "probability_id": 0.5,
    }

# Define the optimiser
optimiser = tf.keras.optimizers.Adam(learning_rate=hyperparameters["learning_rate"])

# Reshape for LSTM
x_train, y_train = shaper.reshape_for_lstm(x_train, y_train, hyperparameters["timesteps"])
x_test, y_test   = shaper.reshape_for_lstm(x_test, y_test, hyperparameters["timesteps"])
# Train model on the split data
model, history   = framework.LSTM(x_train, y_train, optimiser, hyperparameters, save=False, RUN_ID=RUN_ID)

# Evaluate the model using test data
eval_loss, eval_accuracy, eval_auc, eval_precision, eval_recall = analyse.evaluation(model, x_test, y_test)
# Make predictions using test data
y_pred, y_pred_labels = analyse.model_predict(model, x_test, hyperparameters["probability_id"])
# Retrieve metrics using sklearn
fpr, tpr, sklearn_auc, thresholds = analyse.AUCandROCcurve(y_test, y_pred)
# Calculate the average out the models output probabilities
y_pred_mean = np.mean(y_pred)

print(f"\tRUN HASH\n->->->->-> {RUN_ID} <-<-<-<-<-")

# Plotting output
plot.plotter(history.history["accuracy"], "Train Acc", history.history["val_accuracy"], "Val Acc", "Epoch", "Accuracy", "Traing vs Validation Accuracy", "accuracy", RUN_ID=RUN_ID)
plot.plotter(history.history["loss"], "Train Loss", history.history["val_loss"], "Val Loss", "Epoch", "Loss", "Traing vs Validation Loss", "loss", RUN_ID=RUN_ID)
plot.plotter(history.history["auc"], "Train AUC", history.history["val_auc"], "Val AUC", "Epoch", "AUC", "Traing vs Validation AUC", "AUC", RUN_ID=RUN_ID)
plot.plotter(history.history["precision"], "Train Precision", history.history["val_precision"], "Val Precision", "Epoch", "Precision", "Traing vs Validation Precision", "precision", RUN_ID=RUN_ID)
plot.plotter(history.history["recall"], "Train Recall", history.history["val_recall"], "Val Recall", "Epoch", "Recall", "Traing vs Validation Recall", "recall", RUN_ID=RUN_ID)
plot.validation_combined(history, RUN_ID=RUN_ID)
plot.ROC_curve(fpr, tpr, sklearn_auc, RUN_ID=RUN_ID)
plot.prediction_histo(y_pred, y_pred_mean, RUN_ID=RUN_ID)

# Create classification report
analyse.classification(y_test, y_pred_labels, RUN_ID=RUN_ID)
# Create confusion matrix
analyse.confusion(y_test, y_pred_labels, RUN_ID=RUN_ID)

# Logging 
end = datetime.now().strftime("%H:%M:%S")
log_dict = {
    "run_id":                  RUN_ID,
    "history_train_loss":      round(history.history["loss"][-1], 4),
    "history_val_loss":        round(history.history["val_loss"][-1], 4),
    "eval_test_loss":          round(eval_loss, 4),
    "history_train_accuracy":  round(history.history["accuracy"][-1], 4),
    "history_val_accuracy":    round(history.history["val_accuracy"][-1], 4),
    "eval_test_accuracy":      round(eval_accuracy, 4),
    "history_train_auc":       round(history.history["auc"][-1], 4),
    "history_val_auc":         round(history.history["val_auc"][-1], 4),
    "eval_test_auc":           round(eval_auc, 4),
    "prediction_mean":         float(y_pred_mean),
    "start":                   now,
    "end":                     end,
    "hyperparameters":         hyperparameters,
    "notes":                   "No notes"
}
configure.log_to_json(f"{DATA_DIR}/model_logs.jsonl", log_dict)
with open(f"{DATA_DIR}/model_logs.jsonl", "a") as f:
        f.write("\n")

backend.clear_session()
gc.collect()