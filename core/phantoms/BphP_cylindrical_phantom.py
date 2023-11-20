import numpy as np
from phantoms.phantom import phantom
import geometry_func as gf
import BphP_func as bf
import logging

logging.basicConfig(level=logging.DEBUG)

class BphP_cylindrical_phantom(phantom):
    
    def create_volume(self, cfg : dict, n_wavelengths=2, phantom_r=0.013) -> tuple:
        seed = cfg['seed']
        # instantiate random number generator
        if seed is None:
            seed = np.random.randint(0, 2**32 - 1)
            cfg['seed'] = seed
            logging.info(f'no seed provided, random seed: {seed}')
        rng = np.random.default_rng(seed)
        
        # define background phantom scattering and absorption coefficients
        # Corrigendum: Optical properties of biological tissues: a review
        # Steven L Jacques, 2013
        background_mu_a = rng.normal(1.1, 0.15, size=n_wavelengths)
        background_mu_s = rng.normal(1500, 200, size=n_wavelengths)
        logging.debug(f'background_mu_a: {background_mu_a}')
        logging.debug(f'background_mu_s: {background_mu_s}')
        
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
            phantom_r,
            [(cfg['mcx_domain_size'][0]/2), 0.0, (cfg['mcx_domain_size'][2]/2)]
        )
        # mask to segment the background from the phantom
        bg_mask = phantom_mask[:,0,:]
        for i in range(len(cfg['wavelengths'])):
            volume[i,0,:,:,:] = self.H2O['mu_a'][i]
            volume[i,1,:,:,:] = self.H2O['mu_s'][i]
            volume[i,0,:,:,:][phantom_mask] = background_mu_a[i]
            volume[i,1,:,:,:][phantom_mask] = background_mu_s[i]

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
        n_hotspots = rng.integers(0, 5, 1)
        logging.debug(f'n_hotspots: {n_hotspots}')
        hotspots = [] # index as [x,y,z,radius]
        for i in range(n_hotspots[0]):
            # propose a new region/hotspot containing proteins
            origin = [rng.uniform(-0.01, 0.01), # x
                      0.0,                        # y
                      rng.uniform(-0.01, 0.01)] # z
            radius = rng.uniform(0.0005, 0.002)
            logging.debug(f'origin: {origin}, radius: {radius}')
            # ensure the new region/hotspot does not overlap with any other
            intersection = True
            while intersection is True:
                intersection = False
                for hotspot in hotspots:
                    if (np.linalg.norm(np.asarray(hotspot[:3]) - np.asarray(origin))
                    < (radius + hotspot[3])):
                        intersection = True
                if intersection is True:
                    logging.debug('intersection between {hotspot} and {origin}, {radius}')
                    # propose a new region/hotspot, discard the old one
                    origin = [rng.uniform(-0.01, 0.01), # x
                              0.0,                        # y
                              rng.uniform(-0.01, 0.01)] # z
                    radius = rng.uniform(0.0005, 0.002)
                else:
                    logging.debug(f'no intersection for {origin}, {radius}')
                    hotspots.append(origin + [radius])

            # add proteins to the hotspot
            for hotspot in hotspots:
                c_tot = rng.normal(5e-4, 1e-4) # [mols/m^3] = [10^3 M]
                if c_tot < 1e-4: # arbitrary minimum concentration
                    c_tot = 1e-4
                logging.debug(f'hotspot: {hotspot}, c_tot: {c_tot}')
                c_tot = c_tot * gf.cylinder_mask(
                    cfg['dx'],
                    cfg['mcx_grid_size'],
                    radius,
                    origin
                )
                ReBphP_PCM_Pfr_c += c_tot * Pfr_frac
                ReBphP_PCM_Pr_c += c_tot * Pr_frac
        
        # some of these hotspots will include tumours
        inc_tumour = rng.binomial(1, 0.6, size=n_hotspots[0])
        logging.debug(f'inc_tumour: {inc_tumour}')
        for i in range(n_hotspots[0]):
            if inc_tumour[i] == 0:
                pass
            else:
                # maximum extent of the hotspot
                extent = np.linalg.norm(np.asarray(hotspots[i][:3])) + hotspots[i][3]
                if extent >= phantom_r:
                    # the hotspot is too close to the edge of the phantom
                    pass
                else:
                    # define the tumour
                    tumor_profile = gf.quadratic_profile_tumor(
                        cfg['dx'], # [m]
                        cfg['mcx_grid_size'], # [grid points]
                        hotspots[i][3], # [m]
                        hotspots[i][:3] # [m]
                    )
                    tumor_mask = gf.sphere_mask(
                        cfg['dx'], # [m]
                        cfg['mcx_grid_size'], # [grid points]
                        hotspots[i][3], # [m]
                        hotspots[i][:3] # [m]
                    )
                    tumor_mu_a = rng.normal(1.1, 0.15, size=n_wavelengths) # [m^-1]
                    tumor_mu_s = rng.normal(700, 70, size=n_wavelengths) # [m^-1]
                    logging.debug(f'tumor_mu_a: {tumor_mu_a}')
                    logging.debug(f'tumor_mu_s: {tumor_mu_s}')
                    for j in range(len(cfg['wavelengths'])):
                        volume[j,0] += tumor_mu_a[j] * tumor_profile
                        volume[j,1][tumor_mask] = tumor_mu_s[j]
                    

        # mask proteins so they are only inside the cylindrical phantom
        ReBphP_mask = gf.cylinder_mask(
            cfg['dx'],
            cfg['mcx_grid_size'],
            phantom_r - 0.0005,
            [(cfg['mcx_domain_size'][0]/2), 0.0, (cfg['mcx_domain_size'][2]/2)]
        )
        ReBphP_PCM_Pr_c *= ReBphP_mask
        ReBphP_PCM_Pfr_c *= ReBphP_mask

        return (cfg, volume, ReBphP_PCM_Pr_c, ReBphP_PCM_Pfr_c, bg_mask)