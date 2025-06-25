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
    plt.title('Training vs Validation Accuracy')
    plt.savefig(f"{DATA_DIR}/TrainingVsValidationAccuracy.png")


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
