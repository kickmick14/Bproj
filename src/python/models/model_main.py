#######################################
# @author Michael Kane
# @date 07/06/2025
# Script to train and save new models
# whcih can then be loaded into main.py
#######################################
import connect, interact, src.python.models.get_model as get_model
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

settingsPath = "settings/settings.json"

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

model, history, x_train, x_test, y_train, y_test = get_model.train(df)
model.save("models/intialModel", save_format="tf")

loss, accuracy, auc = model.evaluate(x_test, y_test)
print(f"\nLoss: {loss}\nAccuracy: {accuracy}\nAUC: {auc}\n")

y_pred = model.predict(x_test)
y_pred_labels = (y_pred > 0.5).astype(int)

print(classification_report(y_test, y_pred_labels))
print(confusion_matrix(y_test, y_pred_labels))