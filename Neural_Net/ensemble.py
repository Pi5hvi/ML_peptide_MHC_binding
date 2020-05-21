from multiprocessing import Pool
import os
    

class Ensemble:
    """
    Training function accepts 4 parameters:
    Training data
    Validation data
    weights file
    """
    def __init__(self, partitions, training_function, weights_directory):
        assert(not os.path.exists(weights_directory))
        
