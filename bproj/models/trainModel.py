#######################################
# @author Michael Kane
# @date 07/06/2025
# Script to train and save new models
# which can then be loaded into main.py
#######################################
import fetch.connect as connect
import fetch.interact as interact
import models.buildModel as buildModel
import models.plotModel as plot
import models.analyseModel as analyse
import os

# Model name to be saved
ModelName = "mk1"
# Set local and environmental paths
os.environ["BASE_DIR"]         = BASE_DIR         = "/Users/s2417964/VSProjects/bproj"
os.environ["ARTIFACTS_DIR"]    = ARTIFACTS_DIR    = f"{BASE_DIR}/artifacts/models"
os.environ["DATA_DIR"]         = DATA_DIR         = f"{BASE_DIR}/data/{ModelName}"
os.environ["CREDENTIALS_PATH"] = CREDENTIALS_PATH = f"{BASE_DIR}/config/settings.json"

# Make directories if they don't already exist
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

test_client = connect.client(CREDENTIALS_PATH, "test")  # Test account client
main_client = connect.client(CREDENTIALS_PATH, "main")  # Main account client
test_account = test_client.get_account()                # Get test account information

df_options = {                         # df options for retrieving klines
    "client": main_client,             # Client to use
    "pair": "ETHUSDT",                 # Pair to trade
    "kline_period": "1h",              # Period of klines
    "timeframe": "90 days ago UTC",    # Timeframe of kline data
    "future_window": 3,                # How far into future to consider for pct change
    "threshold": 0.01                  # 1% change for trigger
}

df = interact.retrieve_dataframe(**df_options)                      # Use the defined options to retrieve dataframe of binance data
features = df[["return_1h", "rolling_mean_6h", "rolling_std_6h"]]   # "Training data"
labels = df["label"]                                                # Binary classifier

x_train, x_test, y_train, y_test = buildModel.splitData(features, labels)   # Split data for training and testing

model, history = buildModel.train(x_train, x_test, y_train, y_test)  # Train model on the split data
model.save(f"{ARTIFACTS_DIR}/{ModelName}", save_format="tf")         # Save model artifacts
plot.history(history)                                                # Plot model

loss, accuracy, auc = model.evaluate(x_test, y_test)    # Evaluate model based on test data
print(f"\nLoss: {loss}\nAccuracy: {accuracy}\nAUC: {auc}\n")

y_pred = model.predict(x_test)                  # Feeds x_test into model to make binary predictions on it
y_pred_labels = (y_pred > 0.5).astype(int)      # Sigmoid output greater than 0.5 indicates it predicts an increase, save as binary 2

analyse.classification(y_test, y_pred_labels)   # Create classification report
analyse.confusion(y_test, y_pred_labels)        # Create confusion matrix