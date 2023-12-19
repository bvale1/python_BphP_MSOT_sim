import numpy as np
from phantoms.phantom import phantom


# use this class to define the whole volume as water
# this is useful for testing the optical model geometry / light source
class water_phantom(phantom):
    
    def create_volume(self, cfg) -> np.ndarray:
        
        volume = np.zeros(
            (
                len(cfg['wavelengths']),
                2, 
                cfg['mcx_grid_size'][0], 
                cfg['mcx_grid_size'][1], 
                cfg['mcx_grid_size'][2]            
            ), dtype=np.float32
        )
        for i in range(len(cfg['wavelengths'])):
            volume[i,0,:,:,:] = self.H2O['mu_a'][i]
            volume[i,1,:,:,:] = self.H2O['mu_s'][i]

        # no proteins are in this experiment
        ReBphP_PCM_Pr_c = np.zeros((cfg['mcx_grid_size']), dtype=np.float32)
        ReBphP_PCM_Pfr_c = ReBphP_PCM_Pr_c.copy()
        
        # no background mask
        bg_mask = np.ones(
            (cfg['mcx_grid_size'][0],
             cfg['mcx_grid_size'][2]),
            dtype=np.bool
        )

        return (volume, ReBphP_PCM_Pr_c, ReBphP_PCM_Pfr_c, bg_mask)