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


def gpuConfig(intra=4, inter=2): # Don't change intra and inter unless you know what you're doing
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.set_visible_devices(gpus[0], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    return gpus


def splitData(features, labels):

    # Scale features to zero mean, unit variance
    # NxM array where N is amount of data points per indicator and M is the number of indicators
    scaler = StandardScaler()
    x = scaler.fit_transform(features)

    # Store of binary classifier -> Model prediction
    y = labels.values

    # Splits data, no shuffling and consistant in time
    # Train on 80% of the data and test on the other 20%
    splitData = train_test_split(x, y, test_size=0.2, shuffle=False)

    return splitData


def train(splitData):

    # Configure gpus
    gpus = gpuConfig()
    print(f"gpus: {gpus}")

    # Extract training and test data from the split data set
    x_train = splitData[0]
    x_test = splitData[1]
    y_train = splitData[2]
    y_test = splitData[3]

    # Network architecture - feed forward, linear neural networks layer by layer
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(splitData[0].shape[1],)), # Dense (fully-connected) with 64 units, ReLU activation, x.shape[1] defines the expected input shape 
        tf.keras.layers.Dropout(0.3), # 30% of neurons randomly dropped each batch to help prevent overfitting
        tf.keras.layers.Dense(32, activation='relu'), # Dense (fully-connected) with 32 units, ReLU activation
        tf.keras.layers.Dense(1, activation='sigmoid') # Dense with 1 unit and sigmoid activation, producing a probability for the “up” class
    ])

    model.compile(
        optimizer='adam', # "adam" = adaptive learning rate
        loss='binary_crossentropy', # Binary cross entropy is good loss function for binary classification
        metrics=['accuracy', tf.keras.metrics.AUC()] # Tracks both classification accuracy and the AUC (area under the ROC curve) during training
        )
    
    history = model.fit(
        x_train, y_train,
        epochs=20, # How many times the model will the training data - how many times data is used to refine model
        batch_size=32, # Splits sample size into chunks of 32, each batch produces one gradient descent update
        validation_data=(x_test, y_test)
        )

    return model, history