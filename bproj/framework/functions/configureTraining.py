#######################################
# @author Michael Kane
# @date 07/06/2025
# Configuration and training of based
# model
#######################################
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import framework.functions.plotModel as plotModel
import csv, json, os, uuid
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight


def gpuConfig(intra=4, inter=2): # Don't change intra and inter unless you know what you're doing
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.set_visible_devices(gpus[0], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    return gpus


def model_predict(model, x_test, probability):

    y_pred = model.predict(x_test)                  # Feeds x_test into model to make binary predictions on it
    y_pred_labels = (y_pred > probability).astype(int)      # Sigmoid output greater than 0.5 indicates it predicts an increase, save as binary 2

    return y_pred, y_pred_labels


def splitData(features, labels, test_split):

    scaler = StandardScaler()           # Scale features to zero mean, unit variance
    x = scaler.fit_transform(features)  # NxM array where N is amount of data points per indicator and M is the number of indicators
    y = labels.values                   # Store of binary classifier -> Model prediction

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_split, shuffle=False) # Splits data, no shuffling and consistant in time Train on: 80% of the data and test on the other 20%

    return x_train, x_test, y_train, y_test


def log_to_json(filename, log_dict):

    with open(filename, 'a') as f:
        f.write(json.dumps(log_dict) + "\n")


class CSVLogger(tf.keras.callbacks.Callback):

    def __init__(self, filename, run_params):
        super().__init__()
        self.filename = filename
        self.run_params = run_params

    def on_epoch_end(self, epoch, logs=None):
        log_entry = {**self.run_params, **logs, "epoch": epoch}
        log_to_json(self.filename, log_entry)