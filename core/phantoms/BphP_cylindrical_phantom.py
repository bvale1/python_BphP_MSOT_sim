import numpy as np
from phantoms.phantom import phantom
import geometry_func as gf
import BphP_func as bf


class BphP_cylindrical_phantom(phantom):
    
    def create_volume(self, cfg : dict, n_wavelengths=2) -> tuple:
        seed = cfg['seed']
        # instantiate random number generator
        if seed is None:
            seed = np.random.randint(0, 2**32 - 1)
            cfg['seed'] = seed
            print(f'no seed provided, random seed: {seed}')
        rng = np.random.default_rng(seed)
        
        # define background phantom scattering and absorption coefficients
        # Corrigendum: Optical properties of biological tissues: a review
        # Steven L Jacques, 2013
        background_mu_a = rng.normal(0.8, 0.3, size=n_wavelengths)
        background_mu_s = rng.normal(1500, 300, size=n_wavelengths)
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
        # define the cylindrical volume of the phantom
        phantom_mask = gf.cylinder_mask(
            cfg['dx'],
            cfg['mcx_grid_size'],
            0.01,
            [(cfg['mcx_domain_size'][0]/2), 0.0, (cfg['mcx_domain_size'][2]/2)]
        )
        for i in range(len(cfg['wavelengths'])):
            volume[i,0,:,:,:] = background_mu_a[i] * phantom_mask
            volume[i,1,:,:,:] = background_mu_s[i] * phantom_mask

        # define photoswitching proteins
        # initialise proteins, Pr to Pfr ratio is the steady state
        Pr_frac, Pfr_frac = bf.steady_state_BphP(
            self.ReBphP_PCM['Pr'],
            self.ReBphP_PCM['Pfr'],
            wavelength_idx=1
        )
        ReBphP_PCM_Pr_c = np.zeros((cfg['mcx_grid_size']), dtype=np.float32)
        ReBphP_PCM_Pfr_c = np.zeros((cfg['mcx_grid_size']), dtype=np.float32)

        # randomly place hotspots containing proteins based on arbitrary criteria
        n_hotspots = rng.integers(0, 4, 1)
        hotspots = [] # index as [x,y,z,radius]
        for i in range(n_hotspots[0]):
            # propose a new region/hotspot containing proteins
            origin = [rng.uniform(-0.008, 0.008), # x
                      0.0,                        # y
                      rng.uniform(-0.008, 0.008)] # z
            radius = rng.uniform(0.0005, 0.001)
            # ensure the new region/hotspot does not overlap with any other
            intersection = True
            while intersection is True:
                intersection = False
                for hotspot in hotspots:
                    if (np.linalg.norm(np.asarray(hotspot[:3]) - np.asarray(origin))
                    < (radius + hotspot[3])):
                        intersection = True
                if intersection is True:
                    # propose a new region/hotspot, discard the old one
                    origin = [rng.uniform(-0.008, 0.008), # x
                              0.0,                        # y
                              rng.uniform(-0.008, 0.008)] # z
                    radius = rng.uniform(0.0005, 0.001)
                else:
                    hotspots.append(origin + [radius])

            c_tot = rng.normal(0.005, 0.002)
            if c_tot > 0.001: # arbitrary minimum concentration
                c_tot = c_tot * gf.cylinder_mask(
                    cfg['dx'],
                    cfg['mcx_grid_size'],
                    radius,
                    origin
                )
            ReBphP_PCM_Pfr_c += c_tot * Pfr_frac
            ReBphP_PCM_Pr_c += c_tot * Pr_frac

        # mask proteins so they are only inside the cylindrical phantom
        ReBphP_mask = gf.cylinder_mask(
            cfg['dx'],
            cfg['mcx_grid_size'],
            0.0095,
            [(cfg['mcx_domain_size'][0]/2), 0.0, (cfg['mcx_domain_size'][2]/2)]
        )
        ReBphP_PCM_Pr_c = ReBphP_PCM_Pr_c * ReBphP_mask
        ReBphP_PCM_Pfr_c = ReBphP_PCM_Pfr_c * ReBphP_mask

        return (cfg, volume, ReBphP_PCM_Pr_c, ReBphP_PCM_Pfr_c)