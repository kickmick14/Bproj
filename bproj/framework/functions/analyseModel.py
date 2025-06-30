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
def evaluation(model, x_test, y_test):

    loss, accuracy, auc, precision, recall = model.evaluate(x_test, y_test)
    print(f"\nLoss: {loss}\nAccuracy: {accuracy}\nAUC: {auc}\n")

    return loss, accuracy, auc, precision, recall


# Create ROC and retrieve AUC 
def AUCandROCcurve(y_test, y_pred, PRINT=False):

    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)

    if PRINT == True:
        print(f"\nfpr: {fpr}\ntpr: {tpr}\nThresholds: {thresholds}\nauc: {auc}")

    return fpr, tpr, auc, thresholds


# Create and plot the confusion matrix
def confusion(y_test, y_pred_labels, DATA_DIR=None, RUN_ID=None):

    if DATA_DIR is None:
        DATA_DIR = os.environ.get("DATA_DIR", -1) # or raise a clear error
    if RUN_ID is None:
        RUN_ID = os.environ.get("RUN_ID", -1) # or raise a clear error

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred_labels)
    # Plot the results
    plot.confusion(cm, RUN_ID=RUN_ID)


# Create classification report
def classification(y_test, y_pred_labels, DATA_DIR=None, RUN_ID=None):

    if DATA_DIR is None:
        DATA_DIR = os.environ.get("DATA_DIR", -1) # or raise a clear error
    if RUN_ID is None:
        RUN_ID = os.environ.get("RUN_ID", -1) # or raise a clear error

    # Compute classification report
    cr = classification_report(
        y_test,
        y_pred_labels,
        output_dict=True 
    )

    with open(f"{DATA_DIR}/{RUN_ID}/classification_report.json", "w") as f:
        json.dump(cr, f, indent=4)