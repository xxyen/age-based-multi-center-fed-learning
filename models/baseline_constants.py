"""Configuration file for common models/experiments"""

MAIN_PARAMS = { 
    'mnist': {
        'small': (30, 10, 15),
        'medium': (100, 10, 30),
        'large': (400, 20, 5)
        }
}
"""dict: Specifies execution parameters (tot_num_rounds, eval_every_num_rounds, clients_per_round)"""

MODEL_PARAMS = {
    'mnist.cnn': (0.0003, 10) # lr, num_classes
}
"""dict: Model specific parameter specification"""

ACCURACY_KEY = 'accuracy'
BYTES_WRITTEN_KEY = 'bytes_written'
BYTES_READ_KEY = 'bytes_read'
LOCAL_COMPUTATIONS_KEY = 'local_computations'
NUM_ROUND_KEY = 'round_number'
NUM_SAMPLES_KEY = 'num_samples'
CLIENT_ID_KEY = 'client_id'


""" constant for tensorflow model shape """
KERNAL_WIDTH, KERNEL_HEIGHT, NUM_INPUT_CHANNEL, NUM_OUTPUT_CHANNEL = 0, 1, 2, 3