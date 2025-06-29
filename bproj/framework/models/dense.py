#######################################
# @author Michael Kane
# @date 07/06/2025
# dense model outline
#######################################
import tensorflow as tf
import numpy as np
import framework.functions.plotModel as plotModel
from sklearn.utils.class_weight import compute_class_weight
import os


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
        initial_learning_rate=0.05,
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