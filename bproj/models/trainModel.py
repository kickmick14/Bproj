#######################################
# @author Michael Kane
# @date 07/06/2025
# Script to train and save new models
# whcih can then be loaded into main.py
#######################################
import fetch.connect as connect
import fetch.interact as interact
import models.buildModel as buildModel
import models.plotModel as plot
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from path import Path
import numpy as np
import matplotlib.pyplot as plt
import json, os
import pandas as pd

settingsPath = "config/settings.json" # Project settings
test_client = connect.client(settingsPath, "test") # Test account client
main_client = connect.client(settingsPath, "main") # Main account clinet
test_account = test_client.get_account() # Get test account information

# df options for retrieving klines
df_options = {
    "client": main_client,
    "pair": "ETHUSDT",
    "kline_period": "1h", 
    "timeframe": "90 days ago UTC",
    "future_window": 3,
    "threshold": 0.01
}

# Use the defined options to retrieve dataframe of binance data
df = interact.retrieve_dataframe(**df_options)
# Creates dataframes with features to train on and labels to predict
features = df[["return_1h", "rolling_mean_6h", "rolling_std_6h"]] # "Training data"
labels = df["label"] # Binary classifier

# Split the training and test data
splitData = buildModel.splitData(features, labels)
x_train = splitData[0]
x_test = splitData[1]
y_train = splitData[2]
y_test = splitData[3]

# Model name to be saved
ModelName = "mk1"
# Set save paths
PROJECT_DIR = Path("/Users/s2417964/VSProjects/bproj")
ARTIFACTS_DIR = Path(f"{PROJECT_DIR}/artifacts/models")
DATA_DIR = Path(f"{PROJECT_DIR}/data/{ModelName}")
# Make directories if they don't already exist
if not (os.path.isdir(ARTIFACTS_DIR) and os.path.isdir(DATA_DIR)):
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

model, history = buildModel.train(splitData) # Train model on the split data
model.save(ARTIFACTS_DIR / ModelName, save_format="tf") # Save model artifacts
plot.history(history, ARTIFACTS_DIR) # Plot model

# Evaluate model based on test data
loss, accuracy, auc = model.evaluate(x_test, y_test)
print(f"\nLoss: {loss}\nAccuracy: {accuracy}\nAUC: {auc}\n")

# Feeds x_test into model to make binary preductions on it
y_pred = model.predict(x_test)
# Sigmoid output greater than 0.5 indicates it predicts an increase, save as binary 2
y_pred_labels = (y_pred > 0.5).astype(int)

# Compute classification report
cr = classification_report(y_test, y_pred_labels)
cr_dict = classification_report(
    y_test,
    y_pred_labels,
    output_dict=True         # ← this makes it return a dict
)
with open(DATA_DIR / "classification_report.json", "w") as f:
    json.dump(cr_dict, f, indent=4)

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred_labels)
# Plot the results
plot.confusion(cm, DATA_DIR)
df_cm = pd.DataFrame(
    cm,
    index=["actual_0", "actual_1"],   # rows: actual class 0 then 1
    columns=["pred_0",   "pred_1"]    # cols: predicted class 0 then 1
)
# Write it out as CSV
df_cm.to_csv(DATA_DIR / "confusion_matrix.csv")