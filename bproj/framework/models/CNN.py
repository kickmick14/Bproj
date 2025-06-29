#######################################
# @author Michael Kane
# @date 07/06/2025
# CNN model outline
#######################################
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import framework.functions.plotModel as plotModel
import os


def CNN(x_train, x_test, y_train, y_test, timesteps, num_features):

    model = tf.keras.Sequential(
        tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=(timesteps, num_features)),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu'),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    )