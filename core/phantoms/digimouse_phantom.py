import numpy as np
from phantoms.phantom import phantom
import func.geometry_func as gf
import func.utility_func as uf
from scipy import interpolate


class digimouse_phantom(phantom):
    
    def __init__(self, digimouse_path : str):
        super().__init__()
        self.digimouse_path = digimouse_path
        
    def create_volume(self, cfg: dict, y_pos : int, rotate):
        
        # load the digimouse phantom
        with open(self.digimouse_path, 'rb') as f:
            digimouse = np.fromfile(f, dtype=np.int8)
        digimouse = digimouse.reshape(380, 992, 208, order='F')
        
        # spatial dimensions of the digimouse phantom
        dx = 0.0001 # [m]
        [x, y, z] = [dx*380, dx*992, dx*208] 
        
        ny = cfg['mcx_grid_size'][1]
        print(f'ny = {ny}')
        digimouse = digimouse[:, y_pos-(ny//2):y_pos+(ny//2), :]
        print(digimouse.shape)
        
        # interpolate the digimouse phantom to the desired grid size
        interpolate.RegularGridInterpolator(
            (np.arange(380)*dx, np.arange(992)*dx, np.arange(208)*dx),
            digimouse,
            bounds_error=False,
            fill_value=0,
            method='nearest'
        )
        
        # create the simulation volume with digimouse in the center
        
        # assign optical properties to each tissue type
        coupling_medium_mu_a = 0.1 # [m^-1]
        coupling_medium_mu_s = 100 # [m^-1]