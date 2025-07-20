#######################################
# @author Michael Kane
# @date 19/07/2025
# Functions used to configure project
#######################################
import tensorflow as tf
import json, os
from dotenv import load_dotenv


# Configure GPU setup
def gpuConfig(
        intra=4, 
        inter=2
        ):
    
    gpus = tf.config.list_physical_devices( "GPU" )

    if gpus:
        tf.config.set_visible_devices( gpus[0], "GPU" )

    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)

    print( f"gpus: {gpus}" )

    return gpus


# Initialise filestructure and environmental variables
def initialise(
        start_date, 
        start_time,
        load_env=True,
        BASE_DIR=None,
        OUTPUTS_DIR=None,
        CREDENTIALS_PATH=None
        ):

    # Acquire environment variables if necessary
    if load_env:
        load_dotenv()
    if BASE_DIR == None:
        BASE_DIR = os.getenv( "BASE_DIR", -1 )
    if OUTPUTS_DIR == None:
        OUTPUTS_DIR = os.getenv( "OUTPUTS_DIR", -1 )
    if CREDENTIALS_PATH == None:
        CREDENTIALS_PATH = os.getenv( "CREDENTIALS_PATH", -1 )

    # Specific location in time for each instance
    INSTANCE_DIR = f"{OUTPUTS_DIR}/{start_date}/{start_time}"

    # Create all recquired directories
    for file in ["", "artifacts", "plots", "logs", "metrics"]:
        os.makedirs(f"{INSTANCE_DIR}/{file}", exist_ok=True)

    os.environ["START_TIME"] = start_time
    os.environ["START_DATE"] = start_date
    os.environ["INSTANCE_DIR"] = INSTANCE_DIR

    return OUTPUTS_DIR, BASE_DIR, CREDENTIALS_PATH, INSTANCE_DIR


# Log output to JSON file
def log_to_json(
        filename,
        log_dict
        ):

    with open(filename, 'a') as f:
        f.write(json.dumps(log_dict) + "\n")


# Create class for CSV/JSON logger
class CSVLogger(
    tf.keras.callbacks.Callback 
    ):

    def __init__( self, filename, run_params ):
        super().__init__()
        self.filename = filename
        self.run_params = run_params

    def on_epoch_end( self, epoch, logs=None ):
        log_entry = { **self.run_params, **logs, "epoch": epoch }
        log_to_json( self.filename, log_entry )