#######################################
# @author Michael Kane
# @date 08/06/2025
# Functions to plot model outputs
#######################################
import matplotlib.pyplot as plt

def history(history, model_name):

    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training vs Validation Accuracy')
    plt.savefig(f"data/{model_name}/TrainingVsValidationAccuracy.png")