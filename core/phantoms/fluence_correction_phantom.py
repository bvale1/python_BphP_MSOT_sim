import numpy as np
from phantoms.phantom import phantom
from typing import Union

class fluence_correction_phantom(phantom):
    
    def __init__(self, 
                 bg_mask : np.ndarray, # xz segmentation mask from sample image
                 *args,
                 **kwargs):
        super(fluence_correction_phantom, self).__init__(*args, **kwargs)
        assert len(bg_mask.shape) == 2, f'bg_mask must be 2D not shape {bg_mask.shape}'
        assert bg_mask.dtype == bool, 'bg_mask must be a boolean numpy array'
        self.bg_mask = bg_mask[:,np.newaxis,:] # add dimension to broadcast along y-axis
        
    def create_volume(self,
                      mu_a : np.ndarray, 
                      mu_s : Union[float, np.ndarray], 
                      cfg: dict) -> np.ndarray:
        assert len(mu_a.shape) == 2, 'mu_a must be a 2D numpy array'
        assert np.all(mu_a >= 0.0), 'mu_a must be non-negative'
        # the 3d phantom is assumed to be an extrusion from the 2d imaging plane
        mu_a = mu_a[:,np.newaxis,:]
        if type(mu_s) == np.ndarray:
            assert len(mu_s.shape) == 2, 'mu_s must be a 2D numpy array'
            assert np.all(mu_s >= 0.0), 'mu_s must be non-negative'
            mu_s = mu_s[:,np.newaxis,:]
        
        # volume[0] = absorption coefficient [m^-1]
        # volume[1] = scattering coefficient [m^-1]
        volume = np.zeros((
            2, cfg['mcx_grid_size'][0], cfg['mcx_grid_size'][1], cfg['mcx_grid_size'][2]            
        ), dtype=np.float32)
        
        volume[0] += self.H2O['mu_a'][0] * (~self.bg_mask)
        volume[1] += self.H2O['mu_s'][0] * (~self.bg_mask)
        volume[0] += mu_a * self.bg_mask
        volume[1] += mu_s * self.bg_mask
        
        return volume