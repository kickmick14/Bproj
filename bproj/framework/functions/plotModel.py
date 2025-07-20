#######################################
# @author Michael Kane
# @date 08/06/2025
# Functions to plot model outputs
#######################################
import matplotlib.pyplot as plt
import seaborn as sns
import os


def plotter(
        y1, name1, 
        y2, name2, 
        x_axis, y_axis, 
        title, savename, INSTANCE_DIR=None
        ):

    if INSTANCE_DIR is None:
        INSTANCE_DIR = os.environ.get("INSTANCE_DIR", -1) # or raise error

    plt.plot(y1, label=name1)
    plt.plot(y2, label=name2)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.legend()
    plt.title(title)
    plt.savefig(f"{INSTANCE_DIR}/plots/{savename}.png")
    plt.clf()


def validation_combined(
        history,
        INSTANCE_DIR=None
        ):

    if INSTANCE_DIR is None:
        INSTANCE_DIR = os.environ.get("INSTANCE_DIR", -1) # or raise error

    plt.plot(history.history['val_precision'], label='Val Precision')
    plt.plot(history.history['val_recall'], label='Val Recall')
    plt.plot(history.history['val_auc'], label='Val AUC')
    plt.xlabel('Epoch')
    plt.title('Validation Precision, Recall, and AUC')
    plt.legend()
    plt.savefig(f"{INSTANCE_DIR}/plots/validation_combined.png")
    plt.clf()


# Plot scikit learn confusion matrix information
def confusion(
        cm,
        ClassNames=["0", "1"],
        INSTANCE_DIR=None
        ):

    if INSTANCE_DIR is None:
        INSTANCE_DIR = os.environ.get("INSTANCE_DIR", -1) # or raise error

    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=ClassNames,
        yticklabels=ClassNames
        )
    
    ax.invert_yaxis() # Insures intuitive plotting of confusion matrix axis
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.savefig(f"{INSTANCE_DIR}/plots/confusion_matrix.png", bbox_inches="tight")
    plt.clf()


def ROC_curve(
        fpr, 
        tpr, 
        auc, 
        INSTANCE_DIR=None
        ):

    if INSTANCE_DIR is None:
        INSTANCE_DIR = os.environ.get("INSTANCE_DIR", -1) # or raise error

    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)  # baseline diagonal
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(f"{INSTANCE_DIR}/plots/ROC_curve.png")
    plt.clf()


def prediction_histo(
        y_pred, 
        y_pred_mean, 
        INSTANCE_DIR=None
        ):

    if INSTANCE_DIR is None:
        INSTANCE_DIR = os.environ.get("INSTANCE_DIR", -1)  # or raise error

    plt.hist(y_pred, bins=50, alpha=0.7, label="Predictions")
    plt.axvline(y_pred_mean, color='red', linestyle='dashed', linewidth=2, label=f"Mean: {y_pred_mean:.3f}")
    plt.xlabel('Prediction Probabilities')
    plt.ylabel('Entries')
    plt.title('Prediction  Histogram')
    plt.legend()
    plt.savefig(f"{INSTANCE_DIR}/plots/prediction_histo.png")
    plt.clf()