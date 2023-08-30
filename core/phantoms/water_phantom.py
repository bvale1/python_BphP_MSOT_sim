import numpy as np
import core.phantoms.phantom as phantom


# use this class to define the whole volume as water
# this is useful for testing the optical model geometry / light source
class water_phantom(phantom):
    
    def create_volume(self, cfg) -> np.ndarray:
        
        volume = np.zeros(
            (
                len(cfg['wavelengths']),
                2, 
                cfg['grid_size'][0], 
                cfg['grid_size'][1], 
                cfg['grid_size'][2]            
            ), dtype=np.float32
        )
        for i in range(len(cfg['wavelengths'])):
            volume[i,0,:,:,:] = self.water['mu_a'][i]
            volume[i,1,:,:,:] = self.water['mu_s'][i]

        return volume