#######################################
# @author Michael Kane
# @date 08/06/2025
# Functions to plot model outputs
#######################################
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Plot model.train output
def history(history, DATA_DIR=None):

    if DATA_DIR is None:
        DATA_DIR = os.environ.get("DATA_DIR", -1)  # or raise a clear error

    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.savefig(f"{DATA_DIR}/TrainingAndValidationAccuracy.png")
    plt.clf()


def loss(history, DATA_DIR=None):

    if DATA_DIR is None:
        DATA_DIR = os.environ.get("DATA_DIR", -1)  # or raise a clear error

    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig(f"{DATA_DIR}/TrainingAndValidationLoss.png")
    plt.clf()


# Plot scikit learn confusion matrix information
def confusion(cm, ClassNames=["0", "1"], DATA_DIR=None):

    if DATA_DIR is None:
        DATA_DIR = os.environ.get("DATA_DIR", -1)  # or raise a clear error

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
    fig.savefig(f"{DATA_DIR}/confusion_matrix.png", bbox_inches="tight")
    plt.clf()


def ROC_curve(fpr, tpr, auc, DATA_DIR=None):

    if DATA_DIR is None:
        DATA_DIR = os.environ.get("DATA_DIR", -1)  # or raise a clear error

    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)  # baseline diagonal
    #plt.xlim([0.0, 1.0])
    #plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(f"{DATA_DIR}/ROC_curve.png")
    plt.clf()