#######################################
# @author Michael Kane
# @date 07/06/2025
# LSTM model outline
#######################################
import tensorflow as tf
import numpy as np
import framework.functions.plotModel as plotModel
from sklearn.utils.class_weight import compute_class_weight
import os


def LSTM(x_train, x_test, y_train, y_test, timesteps, validation_split, epochs, batch_size, ARTIFACTS_DIR=None, MODEL_NAME=None):

    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, input_shape=(timesteps, x_train.shape[2])),
        tf.keras.layers.Dropout(0.15),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )

    model.summary()

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    )

    weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
        )
    
    class_weight = dict(enumerate(weights))

    history = model.fit(
        x_train, y_train,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        validation_data=(x_test, y_test),
        class_weight=class_weight
    )

    if ARTIFACTS_DIR and MODEL_NAME is None:
        DATA_DIR = os.environ.get("DATA_DIR", -1)  # or raise a clear error
    model.save(f"{ARTIFACTS_DIR}/{MODEL_NAME}", save_format="tf")   # Save model artifacts

    return model, history