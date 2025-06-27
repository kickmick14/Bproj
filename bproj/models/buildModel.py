#######################################
# @author Michael Kane
# @date 07/06/2025
# Configuration and training of based
# model
#######################################
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import models.plotModel as plotModel
import os


def gpuConfig(intra=4, inter=2): # Don't change intra and inter unless you know what you're doing
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.set_visible_devices(gpus[0], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    return gpus


def model_predict(model, x_test):

    y_pred = model.predict(x_test)                  # Feeds x_test into model to make binary predictions on it
    y_pred_labels = (y_pred > 0.5).astype(int)      # Sigmoid output greater than 0.5 indicates it predicts an increase, save as binary 2

    return y_pred, y_pred_labels


def splitData(features, labels):

    scaler = StandardScaler()           # Scale features to zero mean, unit variance
    x = scaler.fit_transform(features)  # NxM array where N is amount of data points per indicator and M is the number of indicators
    y = labels.values                   # Store of binary classifier -> Model prediction

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False) # Splits data, no shuffling and consistant in time Train on: 80% of the data and test on the other 20%

    return x_train, x_test, y_train, y_test


def basic(x_train, x_test, y_train, y_test, ARTIFACTS_DIR=None, MODEL_NAME=None):

    model = tf.keras.Sequential([                                                           # Network architecture - feed forward, linear neural networks layer by layer
        tf.keras.layers.Dense(128, activation='relu', input_shape=(x_train.shape[1],)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.01,
        decay_steps=10000,
        decay_rate=0.9
        )

    optimiser = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    model.compile(                                     # Compile the model
        optimizer=optimiser,                           # "adam" = adaptive learning rate
        loss='binary_crossentropy',                    # Binary cross entropy is good loss function for binary classification
        metrics=['accuracy', tf.keras.metrics.AUC()]   # Tracks both classification accuracy and the AUC (area under the ROC curve) during training
        )
    
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    history = model.fit(                  # Train model
        x_train, y_train,                 # Set training data
        epochs=50,                        # How many times the model will the training data - how many times data is used to refine model
        batch_size=32,                    # Splits sample size into chunks of 32, each batch produces one gradient descent update
        validation_split=0.2,
        callbacks=[early_stop],
        validation_data=(x_test, y_test)  # Set validation data
        )
    
    if ARTIFACTS_DIR and MODEL_NAME is None:
        DATA_DIR = os.environ.get("DATA_DIR", -1)  # or raise a clear error
    model.save(f"{ARTIFACTS_DIR}/{MODEL_NAME}", save_format="tf")   # Save model artifacts

    return model, history


def LSTM(x_train, x_test, y_train, y_test, timesteps, validation_split, epochs, batch_size, ARTIFACTS_DIR=None, MODEL_NAME=None):

    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(timesteps, x_train.shape[2])),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )

    model.summary()

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )

    history = model.fit(
        x_train, y_train,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        validation_data=(x_test, y_test)
    )

    if ARTIFACTS_DIR and MODEL_NAME is None:
        DATA_DIR = os.environ.get("DATA_DIR", -1)  # or raise a clear error
    model.save(f"{ARTIFACTS_DIR}/{MODEL_NAME}", save_format="tf")   # Save model artifacts

    return model, history


def CNN(x_train, x_test, y_train, y_test, timesteps, num_features):

    model = tf.keras.Sequential(
        tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=(timesteps, num_features)),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu'),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    )