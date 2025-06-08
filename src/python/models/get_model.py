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


def train(df):

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.set_visible_devices(gpus[0], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)

    features = df[["return_1h", "rolling_mean_6h", "rolling_std_6h"]]
    labels = df["label"]

    scaler = StandardScaler()
    x = scaler.fit_transform(features)
    y = labels.values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(x.shape[1],)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC()])
    history = model.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_test, y_test))

    return model, history, x_train, x_test, y_train, y_test


def plot_model(history):

    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training vs Validation Accuracy')
    plt.savefig("TrainingPlots/TrainingVsValidationAccuracy.png")