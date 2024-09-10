import numpy as np
from phantoms.phantom import phantom
import func.geometry_func as gf
import func.utility_func as uf


class fluence_correction_phantom(phantom):
    
    def __init__(self, 
                 bg_mask : np.ndarray, # xz segmentation mask from sample image
                 *args,
                 **kwargs):
        super(fluence_correction_phantom, self).__init__(*args, **kwargs)
        self.bg_mask = bg_mask[:,np.newaxis,:] # add dimension to broadcast along y-axis
        
    def create_volume(self, mu_a : np.ndarray, mu_s : float, cfg: dict):
        assert len(mu_a.shape) == 2, 'mu_a must be a 2D numpy array'
        assert np.all(mu_a >= 0.0), 'mu_a must be non-negative'
        # the 3d phantom is assumed to be an extrusion from the 2d imaging plane
        # mu_s is assumed to be constant throughout the phantom
        
        # volume[0] = absorption coefficient [m^-1]
        # volume[1] = scattering coefficient [m^-1]
        volume = np.zeros((
            2, cfg['mcx_grid_size'][0], cfg['mcx_grid_size'][1], cfg['mcx_grid_size'][2]            
        ), dtype=np.float32)
        
        volume[0] += self.H2O['mu_a'][0] * (~self.bg_mask)
        volume[1] += self.H2O['mu_s'][0] * (~self.bg_mask)
        volume[0] + mu_a * self.bg_mask
        volume[1] + mu_s * self.bg_mask
        
        return volume