#######################################
# @author Michael Kane
# @date 07/06/2025
# LSTM model
#######################################
import tensorflow as tf
import numpy as np
import framework.functions.plotModel as plotModel
import framework.functions.configureTraining as configure
from sklearn.utils.class_weight import compute_class_weight
import os


# Model function to be called elsewhere
def LSTM(
        x_train, x_test, y_train, y_test,
        optimiser,
        options,
        save,
        ARTIFACTS_DIR=None,
        MODEL_NAME=None,
        DATA_DIR=None,
        RUN_ID=None
        ):
    
    if ARTIFACTS_DIR is None:
        ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR")
    if MODEL_NAME is None:
        MODEL_NAME = os.environ.get("MODEL_NAME")
    if DATA_DIR is None:
        DATA_DIR = os.environ.get("DATA_DIR")
    if RUN_ID is None:
        RUN_ID = os.environ.get("RUN_ID")

    # Model architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(options["timesteps"], x_train.shape[2])),
        tf.keras.layers.LSTM(options["layer1_units"],
                             return_sequences=False,
                             dropout=options["dropout"],
                             #recurrent_dropout=options["recurrent_dropout"],
                             kernel_regularizer=tf.keras.regularizers.l2(options["kernel_regulariser"])),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compile loss function, optimizer and the metrics for the model
    model.compile(
        loss=options["loss"],
        optimizer=optimiser,
        metrics=['accuracy',
                 tf.keras.metrics.AUC(),
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')
                 ]
    )

    # Prints a summary of the model
    model.summary()
    
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',              # Creates callback for validation loss after each loop
        patience=options["patience"],    # If the validation loss does not improve for x consecutive epochs (patience=x), it will stop the training early.
        restore_best_weights=True        # After stopping, the model’s weights will be set back to those from the epoch with the best (lowest) validation loss, instead of keeping the weights from the final (possibly overfit) epoch.
    )

    #Calculates weights so that the model pays more attention to rare classes (e.g., if you have far fewer 1s than 0s).
    weights = compute_class_weight(
        'balanced',                 # Tells sklearn to automatically set the weights so each class contributes equally to the loss, regardless of its frequency.
        classes=np.unique(y_train), # List all the unique classes in your target (typically [0, 1]).
        y=y_train                   # The actual target labels.
    )
    
    # Converts the array of weights into a dictionary mapping class index to weight.
    class_weight = dict(enumerate(weights))
    epoch_dict = {
        "run_id": RUN_ID
    }
    csv_logger = configure.CSVLogger(f"{DATA_DIR}/{RUN_ID}/epoch_logs.jsonl", epoch_dict)

    # Train the model
    history = model.fit(
        x_train, y_train,
        validation_split=options["validation_split"],
        epochs=options["epochs"],
        batch_size=options["batch_size"],
        callbacks=[csv_logger, early_stop],
        validation_data=(x_test, y_test),
        class_weight=class_weight
    )

    with open(f"{DATA_DIR}/{RUN_ID}/epoch_logs.jsonl", 'a') as f:
        f.write("\n")

    if save == True:
        model.save(f"{ARTIFACTS_DIR}/{MODEL_NAME}/{RUN_ID}", save_format="tf")   # Save model artifacts

    return model, history