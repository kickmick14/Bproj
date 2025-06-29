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
import os, uuid
from datetime import datetime
import tensorflow as tf
os.environ["MODEL_NAME"] = MODEL_NAME = "LSTM_mk1" # Model name to be saved
from settings import ARTIFACTS_DIR, DATA_DIR, CREDENTIALS_PATH

# Configure gpus
gpus = configure.gpuConfig()
print(f"gpus: {gpus}")

# Get starting time
now = datetime.now().strftime('%H:%M:%S')
run_id = str(uuid.uuid4().hex[:8])
os.environ['RUN_ID'] = run_id

# Connect to Binance
test_client = connect.client(CREDENTIALS_PATH, "test")
main_client = connect.client(CREDENTIALS_PATH, "main")
test_account = test_client.get_account()

df_options = {                         # df options for retrieving klines
    "pair": "ETHUSDT",                 # Pair to trade
    "kline_period": "1h",              # Period of klines
    "timeframe": "720 days ago UTC",   # Timeframe of kline data
    "future_window": 5,               # How far into future to consider for pct change
}

klines_df = interact.retrieve_market_data(main_client, **df_options)     # Use the defined options to retrieve dataframe of binance data

threshold   = 0.01  # Target percentage change
RSI_window  = 14    # RSI window for indicator
ATR_window  = 14    # ATR window for indicator
df          = shaper.indicators(klines_df, threshold, RSI_window, ATR_window)              # Add the indicators and labels to the dataframe
features    = df[["macd_signal", "bb_mid", "bb_lower", "bb_upper", "rsi", "atr", "obv"]]   # "Training data"
labels      = df["label"]                                                                  # Binary classifier
test_split  = 0.3   # Percentage of data taken for training
x_train, x_test, y_train, y_test = configure.splitData(features, labels, test_split)       # Split data for training and testing

# Defune the systems hyperparameters
hyperparameters = {
    "layer1_units": 16,
    "layer2_units": 8,
    "dropout": 0.2,
    "recurrent_dropout": 0.0,
    "kernel_regulariser": 0.1,
    "timesteps": 12,
    "validation_split": 0.3,
    "epochs": 2,
    "batch_size": 128,
    "optimiser": "adam",
    "learning_rate": 0.001,
    "loss": "binary_crossentropy",
    "patience": 5
}

optimiser = tf.keras.optimizers.Adam(learning_rate=hyperparameters["learning_rate"])

# Reshape for LSTM
x_train, y_train = shaper.reshape_for_lstm(x_train, y_train, hyperparameters["timesteps"])
x_test, y_test   = shaper.reshape_for_lstm(x_test, y_test, hyperparameters["timesteps"])
model, history   = framework.LSTM(x_train, x_test, y_train, y_test, optimiser, hyperparameters)      # Train model on the split data

# Evaluate the model using test data
eval_loss, eval_accuracy, eval_auc, eval_precision, eval_recall = analyse.evaluation(model, x_test, y_test)
# Make predictions using test data
y_pred, y_pred_labels = configure.model_predict(model, x_test)
# Retrieve metrics using sklearn
fpr, tpr, sklearn_auc, thresholds = analyse.AUCandROCcurve(y_test, y_pred)

# Plotting output
plot.plotter(history.history["accuracy"], "Train Acc", history.history["val_accuracy"], "Val Acc", "Epoch", "Accuracy", "Traing vs Validation Accuracy", "accuracy")
plot.plotter(history.history["loss"], "Train Loss", history.history["val_loss"], "Val Loss", "Epoch", "Loss", "Traing vs Validation Loss", "loss")
plot.plotter(history.history["auc"], "Train AUC", history.history["val_auc"], "Val AUC", "Epoch", "AUC", "Traing vs Validation AUC", "AUC")
plot.plotter(history.history["precision"], "Train Precision", history.history["val_precision"], "Val Precision", "Epoch", "Precision", "Traing vs Validation Precision", "precision")
plot.plotter(history.history["recall"], "Train Recall", history.history["val_recall"], "Val Recall", "Epoch", "Recall", "Traing vs Validation Recall", "recall")
plot.validation_combined(history)
plot.ROC_curve(fpr, tpr, sklearn_auc)
plot.prediction_histo(y_pred)

# Create classification report
analyse.classification(y_test, y_pred_labels)
# Create confusion matrix
analyse.confusion(y_test, y_pred_labels)

# Logging 
end = datetime.now().strftime('%H:%M:%S')
log_dict = {
    "run_id": run_id,
    "start": now,
    "end": end,

    "hyperparameters": hyperparameters,

    "binance_options": df_options,

    "history_train_loss":      round(history.history['loss'][-1], 4),
    "history_val_loss":        round(history.history['val_loss'][-1], 4),
    "eval_test_loss":          round(eval_loss, 4),

    "history_train_accuracy":  round(history.history['accuracy'][-1], 4),
    "history_val_accuracy":    round(history.history['val_accuracy'][-1], 4),
    "eval_test_accuracy":      round(eval_accuracy, 4),

    "history_train_auc":       round(history.history['auc'][-1], 4),
    "history_val_auc":         round(history.history['val_auc'][-1], 4),
    "eval_test_auc":           round(eval_auc, 4),
    "sklearn_auc":             round(sklearn_auc, 4),

    "features": list(features.columns),
    "notes": "No notes"
}
configure.log_to_json(f"{DATA_DIR}/model_logs.csv", log_dict)
with open(f"{DATA_DIR}/model_logs.csv", 'a') as f:
        f.write("\n")