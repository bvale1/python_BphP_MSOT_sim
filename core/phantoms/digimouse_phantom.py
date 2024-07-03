import numpy as np
from phantoms.phantom import phantom
import func.geometry_func as gf
import func.utility_func as uf
from scipy import interpolate
from scipy.ndimage import zoom


class digimouse_phantom(phantom):
    
    def __init__(self, digimouse_path : str):
        super().__init__()
        self.digimouse_path = digimouse_path
        
    def create_volume(self,
                      cfg: dict, 
                      y_idx : int,
                      rotate : int
                      ) -> tuple[np.ndarray, np.ndarray]:
        assert rotate in [0, 1, 2, 3], 'Rotation must be 0, 1, 2 or 3 corresponding to 0, pi/2, pi and 3*pi/2 respectively'
        assert 100 <= y_idx <= 875, 'y_idx must be between 100 and 875' 
        
        # load the digimouse phantom
        with open(self.digimouse_path, 'rb') as f:
            digimouse = np.fromfile(f, dtype=np.int8)
        digimouse = digimouse.reshape(380, 992, 208, order='F')
        
        # spatial dimensions of the digimouse phantom
        dx = 0.0001 # [m]
        
        zoom_factors = [n / o for n, o in zip([308, 992, 380], [308, 992, 208])]
        digimouse = zoom(digimouse, zoom_factors, order=0)
        
        # interpolate to mcx grid dx
        zoom_factors = [dx / cfg['dx'], dx / cfg['dx'], dx / cfg['dx']]
        digimouse = zoom(digimouse, zoom_factors, order=0)
        digimouse = np.rot90(digimouse, rotate, axes=(0, 2))
        
        [nx, ny, nz] = cfg['mcx_grid_size']
        # background mask
        bg_mask = digimouse[:,y_idx,:] != 0
        # translate digimouse[380//2, y_idx, 380//2] to be at the center of the
        # volume (volume[nx//2, ny//2, nz//2])
        tissue_types = np.zeros(cfg['mcx_grid_size'], dtype=np.int8)
        if y_idx - ny//2 < 0:
            # pad the digimouse phantom with zeros
            digimouse = np.pad(
                digimouse, 
                ((0, 0), (ny//2 - y_idx, 0), (0, 0)),
                mode='constant',
                constant_values=0
            )
        if y_idx + ny//2 > digimouse.shape[1]:
            # pad the digimouse phantom with zeros
            digimouse = np.pad(
                digimouse, 
                ((0, 0), (0, y_idx + ny//2 - digimouse.shape[1]), (0, 0)),
                mode='constant',
                constant_values=0
            )
        
        digimouse = digimouse[:, (y_idx-ny//2):(y_idx+ny//2), :]
        # place the digimouse phantom in the center of the volume
        tissue_types[(nx-digimouse.shape[0])//2:(nx+digimouse.shape[0])//2, :, (nz-digimouse.shape[2])//2:(nz+digimouse.shape[0])//2] = digimouse        
        
        coupling_medium_mu_a = 0.1 # [m^-1]
        coupling_medium_mu_s = 100 # [m^-1]
        # The following optical properties were collected for digimouse by Qianqian Fang
        prop=np.array([
            [0, coupling_medium_mu_a*1e-3, coupling_medium_mu_s*1e-3, 0.9, 1.37], # 0 --> background
            [1, 0.0191, 6.6, 0.9, 1.37], # 1 --> skin
            [2, 0.0136, 8.6, 0.9, 1.37], # 2 --> skeleton
            [3, 0.0026, 0.01, 0.9, 1.37], # 3 --> eye
            [4, 0.0186, 11.1, 0.9, 1.37], # 4 --> medulla --> whole brain 
            [5, 0.0186, 11.1, 0.9, 1.37], # 5 --> cerebellum --> whole brain 
            [6, 0.0186, 11.1, 0.9, 1.37], # 6 --> olfactory bulbs --> whole brain 
            [7, 0.0186, 11.1, 0.9, 1.37], # 7 --> external cerebrum --> whole brain 
            [8, 0.0186, 11.1, 0.9, 1.37], # 8 --> striatum --> whole brain 
            [9, 0.0240, 8.9, 0.9, 1.37], # 9 --> heart
            [10, 0.0026, 0.01, 0.9, 1.37], # 10 --> rest of the brain --> whole brain 
            [11, 0.0240, 8.9, 0.9, 1.37], # 11 --> masseter muscles
            [12, 0.0240, 8.9, 0.9, 1.37], # 12 --> lachrymal glands
            [13, 0.0240, 8.9, 0.9, 1.37], # 13 --> bladder
            [14, 0.0240, 8.9, 0.9, 1.37], # 14 --> testis
            [15, 0.0240, 8.9, 0.9, 1.37], # 15 --> stomach
            [16, 0.072, 5.6, 0.9, 1.37], # 16 --> spleen
            [17, 0.072, 5.6, 0.9, 1.37], # 17 --> pancreas
            [18, 0.072, 5.6, 0.9, 1.37], # 18 --> liver
            [19, 0.050, 5.4, 0.9, 1.37], # 19 --> kidneys
            [20, 0.024, 8.9, 0.9, 1.37], # 20 --> adrenal glands
            [21, 0.076, 10.9, 0.9, 1.37] # 21 --> lungs
        ], dtype=np.float32)
        prop[:, 1] *= 1e3 # [mm^-1] -> [m^-1]
        prop[:, 2] *= 1e3 # [mm^-1] -> [m^-1]
        # assign optical properties to the volume
        volume = np.zeros(([2]+cfg['mcx_grid_size']), dtype=np.float32)
        volume[0] = prop[tissue_types,1]
        volume[1] = prop[tissue_types,2]
        
        
        
        return (volume, bg_mask)
