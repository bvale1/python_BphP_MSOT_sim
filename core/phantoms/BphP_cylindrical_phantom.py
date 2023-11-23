import numpy as np
from phantoms.phantom import phantom
import geometry_func as gf
import BphP_func as bf
import logging

logging.basicConfig(level=logging.DEBUG)

class BphP_cylindrical_phantom(phantom):
    
    def gen_regions(self, cfg : dict, rng : np.random._generator.Generator,
                     n_regions : int, phantom_r : float, regions=None) -> list:
        # generates regions to place spherical inomgeneities in the phantom
        # also ensures that they do not intersect with each other
        if regions is None:
            regions = []
        assert isinstance(regions, list)
              
        for i in range(n_regions):
            
            # propose a new region containing proteins
            origin = [rng.uniform(-0.01, 0.01) + (cfg['dx']*cfg['mcx_grid_size'][0]/2), # x
                      (cfg['dx']*cfg['mcx_grid_size'][1]/2),                        # y
                      rng.uniform(-0.01, 0.01) + (cfg['dx']*cfg['mcx_grid_size'][2]/2)] # z
            radius = rng.uniform(0.002, 0.005)
            
            # ensure the new region does not overlap with any other
            intersection, count = True, 0
            while intersection is True:
                intersection = False
                count += 1 # arbitrary maximum number of attempts
                if count > 100:
                    logging.debug(f'maximum number of attempts reached for region {i}')
                    break
                                    
                # maximum extent of the region
                extent = np.linalg.norm(
                    np.asarray(origin) - (cfg['dx']*np.asarray(cfg['mcx_grid_size'])/2)
                ) + radius
                if extent >= phantom_r:
                    logging.debug(f'region {origin}, {radius} is too close to the edge of the phantom')
                    intersection = True
                else:
                    for region in regions:
                        if (np.linalg.norm(np.asarray(region[:3]) - np.asarray(origin))
                        < (radius + region[3])):
                            intersection = True
                            logging.debug('intersection between {region} and {origin}, {radius}')
                            
                if intersection is True:
                    # propose a new region, discard the old one
                    origin = [rng.uniform(-0.01, 0.01) + (cfg['dx']*cfg['mcx_grid_size'][0]/2), # x
                              (cfg['dx']*cfg['mcx_grid_size'][1]/2),                        # y
                              rng.uniform(-0.01, 0.01) + (cfg['dx']*cfg['mcx_grid_size'][2]/2)] # z
                    radius = rng.uniform(0.002, 0.005)
                else:
                    logging.debug(f'no intersection for {origin}, {radius}')
                    regions.append(origin + [radius])

        return regions
        
    def gen_tumour(self, cfg : dict, rng : np.random._generator.Generator,
                   region : list, volume : np.ndarray) -> np.ndarray:
        # define the tumour
        tumor_profile = gf.quadratic_profile_tumor(
            cfg['dx'], # [m]
            cfg['mcx_grid_size'], # [grid points]
            region[3], # [m]
            region[:3] # [m]
        )
        tumor_mask = gf.sphere_mask(
            cfg['dx'], # [m]
            cfg['mcx_grid_size'], # [grid points]
            region[3], # [m]
            region[:3] # [m]
        )
        tumor_mu_a = rng.normal(1.1, 0.15, size=volume.shape[0]) # [m^-1]
        tumor_mu_s = rng.normal(1500, 200, size=volume.shape[0]) # [m^-1]
        logging.debug(f'tumor_mu_a: {tumor_mu_a}, tumor_mu_s: {tumor_mu_s}')
        for j in range(len(cfg['wavelengths'])):
            volume[j,0] += tumor_mu_a[j] * tumor_profile
            volume[j,1][tumor_mask] = tumor_mu_s[j]
            
        return volume
        
            
    def create_volume(self, cfg : dict, n_wavelengths=2, phantom_r=0.013) -> tuple:
        
        # instantiate random number generator
        seed = cfg['seed']
        if seed is None:
            seed = np.random.randint(0, 2**32 - 1)
            cfg['seed'] = seed
            logging.info(f'no seed provided, random seed: {seed}')
        rng = np.random.default_rng(seed)
        
        # define background phantom scattering and absorption coefficients
        # [1] Corrigendum: Optical properties of biological tissues: a review.
        # Steven L Jacques, 2013
        # [2] Diffuse optical tomography of breast cancer during neoadjuvant
        # chemotherapy: a case study with comparison to mri.
        # Regine Choe et al. 2005
        background_mu_a = rng.normal(1.1, 0.15, size=n_wavelengths) # [m^-1]
        background_mu_s = rng.normal(700, 70, size=n_wavelengths)   # [m^-1]
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
        # 2d mask to segment the background from the phantom
        bg_mask = phantom_mask[:,0,:]
        for i in range(len(cfg['wavelengths'])):
            # absorption is zero outside the phantom
            volume[i,1,:,:,:] = self.H2O['mu_s'][i]              # [m^-1]
            volume[i,0,:,:,:][phantom_mask] = background_mu_a[i] # [m^-1]
            volume[i,1,:,:,:][phantom_mask] = background_mu_s[i] # [m^-1]

        # define photoswitching proteins
        # initialise proteins, Pr to Pfr ratio is the steady state
        Pr_frac, Pfr_frac = bf.steady_state_BphP(
            self.ReBphP_PCM['Pr'],
            self.ReBphP_PCM['Pfr'],
            wavelength_idx=1
        )
        logging.debug(f'Pr_frac: {Pr_frac}, Pfr_frac: {Pfr_frac}')
        ReBphP_PCM_Pr_c = np.zeros((cfg['mcx_grid_size']), dtype=np.float32)
        ReBphP_PCM_Pfr_c = np.zeros((cfg['mcx_grid_size']), dtype=np.float32)

        # randomly place regions containing proteins/tumours
        n_proteins = rng.integers(0, 2)
        n_tumor_proteins = rng.integers(0, 4)
        n_tumors = rng.integers(0, 2)
        logging.debug(f'n_proteins: {n_proteins}, n_tumor_proteins: {n_tumor_proteins}, n_tumors: {n_tumors}')
        if n_proteins > 0:
            proteins = self.gen_regions(cfg, rng, n_proteins, phantom_r)
        else:
            proteins = []
        if n_tumor_proteins > 0:
            tumor_proteins = self.gen_regions(
                cfg, rng, n_tumor_proteins, phantom_r, regions=proteins
            )[-n_tumor_proteins:]
        else:
            tumor_proteins = []
        if n_tumors > 0:
            tumors = self.gen_regions(
                cfg, rng, n_tumors, phantom_r, regions=proteins+tumor_proteins
            )[-n_tumors:]
        else:
            tumors = []
        logging.debug(f'proteins: {proteins}, tumor_proteins: {tumor_proteins}, tumors: {tumors}')
                
        # place photoswitching proteins 
        for region in proteins+tumor_proteins:
            c_tot = rng.normal(2e-4, 5e-5) # [mols/m^3] = [10^3 M]
            if c_tot < 5e-5: # arbitrary minimum concentration
                c_tot = 5e-5
            logging.debug(f'region: {region}, c_tot: {c_tot}')
            c_tot = c_tot * gf.sphere_mask(
                cfg['dx'],
                cfg['mcx_grid_size'],
                region[3],
                region[:3]
            )
            ReBphP_PCM_Pfr_c += c_tot * Pfr_frac
            ReBphP_PCM_Pr_c += c_tot * Pr_frac
        
        for tumor in tumors+tumor_proteins:
            logging.debug(f'generating tumor: {tumor}')
            volume = self.gen_tumour(cfg, rng, tumor, volume)
        
        return (cfg, volume, ReBphP_PCM_Pr_c, ReBphP_PCM_Pfr_c, bg_mask)