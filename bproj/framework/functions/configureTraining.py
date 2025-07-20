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


# Split data into test and training
def splitData(
        features, 
        labels, 
        test_split
        ):

    # Splits data, no shuffling and consistant in time Train on: 80% of the data and test on the other 20%
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels.values,
        test_size=test_split,
        shuffle=True
        )

    # Scale features to zero mean, unit variance
    scaler = StandardScaler()
    # NxM array where N is amount of data points per indicator and M is the number of features
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, x_test, y_train, y_test