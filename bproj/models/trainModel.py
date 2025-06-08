#######################################
# @author Michael Kane
# @date 07/06/2025
# Script to train and save new models
# whcih can then be loaded into main.py
#######################################
import fetch.connect as connect
import fetch.interact as interact
import models.buildModel as buildModel
import models.plotModel as plotModel
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

settingsPath = "config/settings.json"

test_client = connect.client(settingsPath, "test") # For interacting with test account
main_client = connect.client(settingsPath, "main") # For collecting more kline data
test_account = test_client.get_account()

df_options = {
    "client": main_client,
    "pair": "ETHUSDT",
    "kline_period": "1h", 
    "timeframe": "90 days ago UTC",
    "future_window": 3,
    "threshold": 0.01
}

df = interact.retrieve_dataframe(**df_options)
features = df[["return_1h", "rolling_mean_6h", "rolling_std_6h"]]
labels = df["label"]

x, y, splitData = buildModel.splitData(features, labels)
x_train = splitData[0]
x_test = splitData[1]
y_train = splitData[2]
y_test = splitData[3]

model_name = "mk1Model" #TODO make it so that if this directory doesn't already exist, then it is created so that I don't have to do it for every new model
SAVE_DIR = f"data/{model_name}"

model, history = buildModel.train(x, y, splitData, df)

model.save(f"artifacts/models/{model_name}", save_format="tf")
plotModel.history(history, model_name)

loss, accuracy, auc = model.evaluate(x_test, y_test)
print(f"\nLoss: {loss}\nAccuracy: {accuracy}\nAUC: {auc}\n")

y_pred = model.predict(x_test)
y_pred_labels = (y_pred > 0.5).astype(int)

cr = classification_report(y_test, y_pred_labels)
cm = confusion_matrix(y_test, y_pred_labels)

np.savetxt(f"{SAVE_DIR}/confusion_matrix.csv", cm, fmt="%d", delimiter=",",
           header=",".join([f"pred_{i}" for i in range(cm.shape[1])]),
           comments="")

class_names = ["Up", "notUP"] #TODO find out what these class names actually are and set accordingly

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig(f"{SAVE_DIR}/confusion_matrix.png", bbox_inches="tight", dpi=150)

with open(f"{SAVE_DIR}/classification_report.json", "w") as f:
    json.dump(cr, f, indent=4)