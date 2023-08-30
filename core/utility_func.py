import os
import numpy as np

def create_dir(path):
    # create directory if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)

# 3D euclidean rotation around the y-axis
def Ry3D(θ):
    return np.array([
        [np.cos(θ), 0, np.sin(θ)],
        [0, 1, 0],
        [-np.sin(θ), 0, np.cos(θ)]
    ])

# 2D euclidean rotation around the y-axis
def Ry2D(θ):
    return np.array([
        [np.cos(θ), np.sin(θ)],
        [-np.sin(θ), np.cos(θ)]
    ])