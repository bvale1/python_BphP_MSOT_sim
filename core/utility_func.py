import os
import numpy as np

def create_dir(path):
    # create directory if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)
