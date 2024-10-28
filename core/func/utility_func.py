import os
import json
import logging
import time
import h5py
import numpy as np
from typing import Union

def create_lock_dir(
        file : str, # file to lock
        wait_time : int=60) -> bool: # time to wait for lock in seconds
    '''A simple solution to lock a file is to create a directory with the same
       name, works across different platforms and machines with access
       the same files'''    
    # remove file extension
    file = file.split('.')
    if len(file) == 1:
        file = file[0]
    else:
        file = '.'.join(file[:-1])
    
    while wait_time > 0:
        if not os.path.exists(file):
            os.makedirs(file)
            logging.info(f'created lock on {file}')
            return True
        else:
            time.sleep(0.1)
            wait_time -= 0.1
    # if the lock is still active after the wait time, the process with the lock
    # has almost certainly crashed, so we remove the lock
    logging.info(f'still locked after {wait_time} seconds, taking over lock')
    return False 
    
def delete_lock_dir(file : str):
    file = file.split('.')
    if len(file) == 1:
        file = file[0]
    else:
        file = '.'.join(file[:-1])

    try:
        os.rmdir(file)
        logging.info(f'removed lock on {file}')
        return True
    except Exception as e:
        logging.info(f'could not remove lock on {file}, {e}')    
        return False

def save_json(file : str, dictionary : dict):
    create_lock_dir(file)
    with open(file, 'w') as f:
        json.dump(dictionary, f, indent='\t')
    delete_lock_dir(file)

def load_json(file : str) -> dict:
    create_lock_dir(file)
    logging.info(f'loading {file}')
    with open(file, 'r') as f:
        dictionary = json.load(f)
    delete_lock_dir(file)
    return dictionary

def create_dir(path : str):
    # create directory if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)

# 3D euclidean rotation around the y-axis
def Ry3D(θ : float) -> np.ndarray:
    return np.array([
        [np.cos(θ), 0, np.sin(θ)],
        [0, 1, 0],
        [-np.sin(θ), 0, np.cos(θ)]
    ])

# 2D euclidean rotation around the y-axis
def Ry2D(θ : float) -> np.ndarray:
    return np.array([
        [np.cos(θ), np.sin(θ)],
        [-np.sin(θ), np.cos(θ)]
    ])
    
def square_centre_crop(image : np.ndarray, size : int) -> np.ndarray:
    width, height = image.shape[-2:]
    if width < size or height < size:
        print('Image is smaller than crop size, returning original image')
        return image
    else:
        x = (width - size) // 2
        y = (height - size) // 2
        image = image[..., x:x+size, y:y+size]
        return image

def square_centre_pad(image : np.ndarray, size : int) -> np.ndarray:
    width, height = image.shape[-2:]
    if width > size or height > size:
        print('Image is larger than pad size, returning original image')
        return image
    else:
        padded_image = np.zeros(
            tuple(image.shape[:-2],)+(size, size),
            dtype=image.dtype
        )
        x = (size - width) // 2
        y = (size - height) // 2
        padded_image[..., x:x+width, y:y+height] = image
        return padded_image
        
def crop_p0_3D(p0 : np.ndarray, size : Union[list, np.ndarray]) -> np.ndarray:
    # similar to square_centre_crop but for arrays containing 3D volume data
    # rather than 2D slices
    width, depth, height = p0.shape[-3:]
    if width < size[0] or depth < size[1] or height < size[2]:
        print('Array is smaller than crop size, returning original image')
        return p0
    else:
        x = (width - size[0]) // 2
        y = (depth - size[1]) // 2
        z = (height - size[2]) // 2
        p0 = p0[..., x:x+size[0], y:y+size[1], z:z+size[2]]
        return p0

def pad_p0_3D(p0 : np.ndarray, size : int) -> np.ndarray:
    # similar to square_centre_pad but for arrays containing 3D volume data
    # rather than 2D slices, the crop is applied to the xz plane while the
    # y axis is ignored
    width, depth, height = p0.shape[-3:]
    if width > size or height > size:
        print('Image is larger than pad size, returning original image')
        return p0
    else:
        padded_p0 = np.zeros(
            tuple(p0.shape[:-3],)+(size, depth, size),
            dtype=p0.dtype
        )
        x = (size - width) // 2
        y = (size - height) // 2
        padded_p0[..., x:x+width, :, y:y+height] = p0
        return padded_p0
    
def masked_RMSE(y_true : np.ndarray, y_pred : np.ndarray, mask : np.ndarray) -> float:
    # calculate root mean squared error of y_true and y_pred only where mask is 1
    mask = mask.astype(bool)
    return np.sqrt(np.mean((y_true[mask] - y_pred[mask])**2))

def masked_PSNR(y_true : np.ndarray, y_pred : np.ndarray, mask : np.ndarray) -> float:
    # calculate peak signal to noise ratio of y_true and y_pred only where mask is 1
    mask = mask.astype(bool)
    RMSE = masked_RMSE(y_true, y_pred, mask)
    return 20*np.log10(np.max(y_true[mask]) / RMSE)

def masked_SSIM(y_true : np.ndarray,
                y_pred : np.ndarray,
                mask : np.ndarray,
                k1 : float=0.01,
                k2 : float=0.03) -> float:
    # calculate structural similarity index of y_true and y_pred only where mask is 1
    mask = mask.astype(bool)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    covar = np.cov(y_true, y_pred)[0, 1]
    L = np.max(y_true) - np.min(y_true) # dynamic range of the image
    c1 = (k1*L)**2
    c2 = (k2*L)**2
    numerator = (2*mean_true*mean_pred + c1)*(2*covar + c2)
    denominator = (mean_true**2 + mean_pred**2 + c1)*(var_true + var_pred + c2)
    return numerator / denominator
    
# same function as in https://github.com/bvale1/MSOT_Diffusion.git
def load_sim(path : str, args='all', verbose=False) -> list:
    data = {}
    with h5py.File(os.path.join(path, 'data.h5'), 'r') as f:
        images = list(f.keys())
        if verbose:
            print(f'images found {images}')
        if args == 'all':
            args = f[images[0]].keys()
            if verbose:
                print(f'args found in images[0] {args}')
        for image in images:
            data[image] = {}
            for arg in args:
                if arg not in f[image].keys():
                    print(f'arg {arg} not found in {image}')
                    pass
                # include 90 deg anticlockwise rotation
                elif arg != 'sensor_data':
                    data[image][arg] = np.rot90(
                        np.array(f[image][arg][()]), k=1, axes=(-2,-1)
                    ).copy()
                else:
                    data[image][arg] = np.array(f[image][arg][()])
            
    with open(path+'/config.json', 'r') as f:
        cfg = json.load(f)
        
    return [data, cfg]