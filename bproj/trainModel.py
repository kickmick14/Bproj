#######################################
# @author Michael Kane
# @date 07/06/2025
# Script to train and save new models
#######################################
import setup.configure as config
import API.fetch.connect as connect
import API.fetch.interact as interact
import API.functions.indicatorData as indicators
import framework.functions.configureTraining as configTrain
import framework.functions.plotModel as plot
import framework.functions.analyseModel as analyse
import framework.models.LSTM as framework
import tensorflow as tf
import numpy as np
import gc
from datetime import datetime
from keras import backend

# Get starting date and time
start = datetime.now()
start_time = start.strftime( "%H:%M:%S" )
start_date = start.strftime( "%Y-%m-%d" )

# Create necessary file structure
OUTPUTS_DIR, BASE_DIR, CREDENTIALS_PATH, INSTANCE_DIR = config.initialise( start_date, start_time )

# Configure gpus (gpu information available)
config.gpuConfig()

# Connect to Binance
test_client = connect.client( CREDENTIALS_PATH, "test" )
main_client = connect.client( CREDENTIALS_PATH, "main" )

# Obtain test account
test_account = test_client.get_account()

# df options for retrieving klines
df_options = {
    "pair": "ETHUSDT",                  # Pair to trade
    "kline_period": "2h",               # Period of klines
    "timeframe": "1440 days ago UTC",   # Timeframe of kline data
    "future_window": 10,                # How far into future to consider for pct change
}

# Use the defined options to retrieve dataframe of binance data
klines_df = interact.retrieve_market_data( main_client, **df_options )

# List of options which determines the indicators
indicators_dict = {
    "threshold": 0.01,
    "RSI_window": 14,
    "ATR_window": 14,
    "stochastic_window": 14,
    "CCI_window": 20,
    "lag_list": [1, 5, 10]
    }

# Add the indicators and labels to the dataframe
df = indicators.get( klines_df, **indicators_dict )
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
    *[ f"return_{lag}" for lag in indicators_dict["lag_list"] ] ] ]

# Logging binance options
binance_options = {
     "binance_options":  df_options,
     "features":         list(features.columns),
    }

# Log Binance options in use
config.log_to_json( f"{INSTANCE_DIR}/logs/binance_options.jsonl", binance_options )

# Binary classifier
labels = df["label"]

# Fraction of data taken for post training testing
test_split  = 0.2

# Split data for training and testing
x_train, x_test, y_train, y_test = configTrain.splitData( features, labels, test_split )

# Define the systems hyperparameters
hyperparameters = {
    "layer1_units":         64,
    "layer2_units":         8,
    "dropout":              0.30,
    "recurrent_dropout":    "N/A",
    "kernel_regulariser":   0.02,
    "timesteps":            12,
    "validation_split":     0.3,
    "epochs":               50,
    "batch_size":           48,
    "optimiser":            "adam",
    "learning_rate":        0.0001,
    "loss":                 "binary_crossentropy",
    "patience":             5,
    "probability_id":       0.5,
    }

# Define the optimiser
optimiser = tf.keras.optimizers.Adam( learning_rate = hyperparameters["learning_rate"] )

# Reshape for LSTM
x_train, y_train = indicators.reshape_for_lstm( x_train, y_train, hyperparameters["timesteps"] )
x_test, y_test   = indicators.reshape_for_lstm( x_test, y_test, hyperparameters["timesteps"] )

# Train model on the split data
model, history   = framework.LSTM( x_train, y_train, optimiser, hyperparameters, save = False )

# Evaluate the model using test data
eval_loss, eval_accuracy, eval_auc, eval_precision, eval_recall = analyse.evaluation( model, x_test, y_test )

# Make predictions using test data
y_prediction, y_prediction_labels = analyse.model_predict( model, x_test, hyperparameters["probability_id"] )

# Retrieve metrics using sklearn
fpr, tpr, sklearn_auc, thresholds = analyse.AUCandROCcurve( y_test, y_prediction )

# Calculate the average out the models output probabilities
y_prediction_mean = np.mean( y_prediction )

print(f"\tRUN START \n->->->->-> {start_time} <-<-<-<-<-")

# Plot output
plot.plotter( history.history["accuracy"],  "Train Acc",        history.history["val_accuracy"],    "Val Acc",       "Epoch", "Accuracy",  "Traing vs Validation Accuracy",  "accuracy"  )
plot.plotter( history.history["loss"],      "Train Loss",       history.history["val_loss"],        "Val Loss",      "Epoch", "Loss",      "Traing vs Validation Loss",      "loss"      )
plot.plotter( history.history["auc"],       "Train AUC",        history.history["val_auc"],         "Val AUC",       "Epoch", "AUC",       "Traing vs Validation AUC",       "AUC"       )
plot.plotter( history.history["precision"], "Train Precision",  history.history["val_precision"],   "Val Precision", "Epoch", "Precision", "Traing vs Validation Precision", "precision" )
plot.plotter( history.history["recall"],    "Train Recall",     history.history["val_recall"],      "Val Recall",    "Epoch", "Recall",    "Traing vs Validation Recall",    "recall"    )
plot.validation_combined( history )
plot.ROC_curve( fpr, tpr, sklearn_auc )
plot.prediction_histo( y_prediction, y_prediction_mean )

# Create classification report
analyse.classification( y_test, y_prediction_labels )
# Create confusion matrix
analyse.confusion( y_test, y_prediction_labels )

# Get end time
end_time = datetime.now().strftime( "%H:%M:%S" )

# Logging 
log_dict = {
    "history_train_loss":      round( history.history["loss"][-1],          4 ),
    "history_val_loss":        round( history.history["val_loss"][-1],      4 ),
    "eval_test_loss":          round( eval_loss,                            4 ),
    "history_train_accuracy":  round( history.history["accuracy"][-1],      4 ),
    "history_val_accuracy":    round( history.history["val_accuracy"][-1],  4 ),
    "eval_test_accuracy":      round( eval_accuracy,                        4 ),
    "history_train_auc":       round( history.history["auc"][-1],           4 ),
    "history_val_auc":         round( history.history["val_auc"][-1],       4 ),
    "eval_test_auc":           round( eval_auc,                             4 ),
    "prediction_mean":         float( y_prediction_mean ),
    "date":                    start_date,
    "start":                   start_time,
    "end":                     end_time,
    "hyperparameters":         hyperparameters,
    "notes":                   "No notes"
}

# Log model
config.log_to_json( f"{INSTANCE_DIR}/logs/model_logs.jsonl", log_dict )
with open( f"{INSTANCE_DIR}/logs/model_logs.jsonl", "a" ) as f:
    f.write( "\n" )

# Use these tools to ensure no memory collapse when looping over network hyperparameters
backend.clear_session()
gc.collect()