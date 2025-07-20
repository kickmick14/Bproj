#######################################
# @author Michael Kane
# @date 08/06/2025
# Functions to analyse models output
#######################################
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
import tensorflow as tf
import pandas as pd
import framework.functions.plotModel as plot
import json, os


# Evaluate model based on test data
def evaluation(
    model,
    x_test,
    y_test
    ):

    loss, accuracy, auc, precision, recall = model.evaluate(x_test, y_test)
    print(f"\nLoss: {loss}\nAccuracy: {accuracy}\nAUC: {auc}\n")

    return loss, accuracy, auc, precision, recall


# Create ROC and retrieve AUC 
def AUCandROCcurve(
        y_test,
        y_pred,
        PRINT=False
        ):

    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)

    if PRINT == True:
        print(f"\nfpr: {fpr}\ntpr: {tpr}\nThresholds: {thresholds}\nauc: {auc}")

    return fpr, tpr, auc, thresholds


# Create and plot the confusion matrix
def confusion(
        y_test,
        y_pred_labels,
        INSTANCE_DIR=None,
        ):

    if INSTANCE_DIR is None:
        INSTANCE_DIR = os.environ.get("INSTANCE_DIR", -1)

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred_labels)
    # Plot the results
    plot.confusion(cm)


# Create classification report
def classification(
        y_test, 
        y_pred_labels, 
        INSTANCE_DIR=None
        ):

    if INSTANCE_DIR is None:
        INSTANCE_DIR = os.environ.get("INSTANCE_DIR", -1)

    # Compute classification report
    cr = classification_report(
        y_test,
        y_pred_labels,
        output_dict=True )

    with open(f"{INSTANCE_DIR}/metrics/classification_report.json", "w") as f:
        json.dump(cr, f, indent=4)


# Evaluate model based on test data
def evaluation(
        model, 
        x_test, 
        y_test
        ):

    loss, accuracy, auc, precision, recall = model.evaluate(x_test, y_test)
    print(f"\nLoss: {loss}\nAccuracy: {accuracy}\nAUC: {auc}\n")

    return loss, accuracy, auc, precision, recall


# Test model on data subset x_test
def model_predict(
        model, 
        x_test, 
        probability
        ):

    # Feeds x_test into model to make binary predictions on it
    y_pred = model.predict(x_test)
    # Sigmoid output greater than 0.5 indicates it predicts an increase, save as binary 1
    y_pred_labels = (y_pred > probability).astype(int)

    return y_pred, y_pred_labels