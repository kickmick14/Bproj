#######################################
# @author Michael Kane
# @date 07/06/2025
# LSTM model architecture and config
#######################################
import tensorflow as tf
import numpy as np
import setup.configure as config
from sklearn.utils.class_weight import compute_class_weight
import os


# Model function to be called elsewhere
def LSTM(
        x_train,
        y_train,
        optimiser,
        options,
        save,
        INSTANCE_DIR = None
        ):
    
    if INSTANCE_DIR == None:
        INSTANCE_DIR = os.environ.get("INSTANCE_DIR", -1)

    # Model architecture
    model = tf.keras.Sequential([
        # Input layer
        tf.keras.layers.Input( 
            shape                   = (options["timesteps"], x_train.shape[2]) ),
        # Layer 1
        tf.keras.layers.LSTM(
            options["layer1_units"],
            return_sequences        = True,
            dropout                 = options["dropout"],
            kernel_regularizer      = tf.keras.regularizers.l2(options["kernel_regulariser"] ) ),
        # Layer 2
        tf.keras.layers.LSTM(
            options["layer2_units"],
            return_sequences        = False,
            dropout                 = options["dropout"],
            kernel_regularizer      = tf.keras.regularizers.l2(options["kernel_regulariser"] ) ),
        # Output layer
        tf.keras.layers.Dense(
            1,
            activation              = 'sigmoid' )
    ])

    # Compile loss function, optimiser and the metrics for the model
    model.compile(
        loss        = options["loss"],
        optimizer   = optimiser,
        metrics     = [
            'accuracy',
            tf.keras.metrics.AUC(),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
            ] )

    # Prints a summary of the model
    model.summary()
    
    # Conditions for stopping training early
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor                 = 'val_loss',           # Creates callback for validation loss after each loop
        patience                = options["patience"],  # If the validation loss does not improve for x consecutive epochs (patience=x), it will stop the training early
        restore_best_weights    = True                  # After stopping, the model’s weights will be set back to those from the epoch with the best (lowest) validation loss, instead of keeping the weights from the final (possibly overfit) epoch
    )

    #Calculates weights so that the model pays more attention to rare classes (e.g., if you have far fewer 1s than 0s)
    weights = compute_class_weight(
        'balanced',                         # Sklearn sets weights so each class contributes equally to the loss, regardless of its frequency
        classes     = np.unique(y_train),   # List all the unique classes in your target (typically [0, 1])
        y           = y_train               # The target labels
    )
    
    # Converts the array of weights into a dictionary mapping class index to weight.
    class_weight    = dict(enumerate(weights))
    epoch_dict      = {}
    csv_logger      = config.CSVLogger(f"{INSTANCE_DIR}/logs/epoch_logs.jsonl", epoch_dict)

    # Train the model
    history = model.fit(
        x_train,
        y_train,
        validation_split    = options["validation_split"],
        epochs              = options["epochs"],
        batch_size          = options["batch_size"],
        callbacks           = [csv_logger, early_stop],
        class_weight        = class_weight
    )

    # Write logs to output
    with open(f"{INSTANCE_DIR}/logs/epoch_logs.jsonl", 'a') as f:
        f.write("\n")

    # Save model artifacts
    if save == True:
        model.save(f"{INSTANCE_DIR}/artifacts", save_format="tf")

    return model, history