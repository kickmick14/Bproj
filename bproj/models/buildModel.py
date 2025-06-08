#######################################
# @author Michael Kane
# @date 07/06/2025
# Configuration and training of based model
#######################################
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import models.plotModel as plotModel


def gpuConfig(intra=4, inter=2): # Don't change intra and inter unless you know what you're doing
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.set_visible_devices(gpus[0], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    return gpus


def splitData(features, labels):

    scaler = StandardScaler()
    x = scaler.fit_transform(features)
    y = labels.values

    splitData = train_test_split(x, y, test_size=0.2, shuffle=False)

    return x, y, splitData


def train(x, y, splitData, plot=True):

    gpus = gpuConfig()
    print(f"gpus: {gpus}")

    x_train = splitData[0]
    x_test = splitData[1]
    y_train = splitData[2]
    y_test = splitData[3]

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(x.shape[1],)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC()])
    history = model.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_test, y_test))

    return model, history