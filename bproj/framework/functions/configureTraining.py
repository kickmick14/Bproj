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


# Split data into test and training
def splitData(features, labels, test_split):

    # Splits data, no shuffling and consistant in time Train on: 80% of the data and test on the other 20%
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels.values,
        test_size=test_split,
        shuffle=False
        )

    # Scale features to zero mean, unit variance
    scaler = StandardScaler()
    # NxM array where N is amount of data points per indicator and M is the number of features
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

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