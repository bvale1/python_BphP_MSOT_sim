import numpy as np
from phantoms.phantom import phantom
import geometry_func as gf
import BphP_func as bf


class BphP_cylindrical_phantom(phantom):
    
    def create_volume(self, cfg : dict, seed=None) -> tuple:
        
        # instantiate random number generator
        if seed is None:
            seed = np.random.randint(0, 2**32 - 1)
            print(f'no seed provided, random seed: {seed}')
        rng = np.random.default_rng(seed)
        
        # initialise proteins, Pr to Pfr ratio is the steady state
        Pr_frac, Pfr_frac = bf.steady_state_BphP(
            self.ReBphP_PCM['Pr'],
            self.ReBphP_PCM['Pfr'],
            wavelength_idx=1
        )
        
        n_hotspots = np.random.randint(0, 5)
        
        
        # define volume scattering and absorption coefficients
        # index as [1, ..., lambda]->[mu_a, mu_s]->[x]->[y]->[z]
        volume = np.zeros(
            (
                len(cfg['wavelengths']),
                2, 
                cfg['mcx_grid_size'][0], 
                cfg['mcx_grid_size'][1], 
                cfg['mcx_grid_size'][2]            
            ), dtype=np.float32
        )
        
        return (seed, volume, ReBphP_PCM_Pr_c, ReBphP_PCM_Pfr_c)