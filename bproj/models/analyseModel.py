#######################################
# @author Michael Kane
# @date 08/06/2025
# Functions to analyse models output
#######################################
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import models.plotModel as plot
import json, os
from sklearn.metrics import roc_curve, roc_auc_score


def evaluation(model, x_test, y_test):

    loss, accuracy, auc = model.evaluate(x_test, y_test)    # Evaluate model based on test data
    print(f"\nLoss: {loss}\nAccuracy: {accuracy}\nAUC: {auc}\n")

    return loss, accuracy, auc


def AUCandROCcurve(y_test, y_pred, PRINT=False):

    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)

    if PRINT == True:
        print(f"\nfpr: {fpr}\ntpr: {tpr}\nThresholds: {thresholds}\nauc: {auc}")

    return fpr, tpr, auc, thresholds



# Create and plot the confusion matrix
def confusion(y_test, y_pred_labels, DATA_DIR=None):

    if DATA_DIR is None:
        DATA_DIR = os.environ.get("DATA_DIR", -1)  # or raise a clear error

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred_labels)
    # Convert it into dataframe
    df_cm = pd.DataFrame(
        cm,
        index=["actual_0", "actual_1"], # rows: actual class 0 then 1
        columns=["pred_0",   "pred_1"]  # cols: predicted class 0 then 1
    )
    # Write it out as CSV
    df_cm.to_csv(f"{DATA_DIR}/confusion_matrix.csv")
    # Plot the results
    plot.confusion(cm)


# Create classification report
def classification(y_test, y_pred_labels, DATA_DIR=None):

    if DATA_DIR is None:
        DATA_DIR = os.environ.get("DATA_DIR", -1)  # or raise a clear error

    # Compute classification report
    cr = classification_report(
        y_test,
        y_pred_labels,
        output_dict=True 
    )

    with open(f"{DATA_DIR}/classification_report.json", "w") as f:
        json.dump(cr, f, indent=4)