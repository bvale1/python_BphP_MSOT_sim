import numpy as np
from phantoms.phantom import phantom
import func.geometry_func as gf
import func.BphP_func as bf


class plane_cyclinder_tumour(phantom):

    def create_volume(self, cfg : dict, mu_a_background=0.7, r_tumour=0.001):
    
        mu_s_background = 2500 # [m^-1]
        mu_s_tumour = 1700 # [m^-1]
    
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
        
        background_mask = gf.cylinder_mask(
            cfg['dx'], # [m]
            cfg['mcx_grid_size'], # [grid points]
            0.01, # [m]
            [(cfg['mcx_domain_size'][0]/2), 0.0, (cfg['mcx_domain_size'][2]/2)]
        )
        
        volume[0, 0] += mu_a_background * background_mask
        volume[0, 1] += mu_s_background * background_mask
        volume[0, 0] += self.H2O['mu_a'][0] * np.logical_not(background_mask)
        volume[0, 1] += self.H2O['mu_s'][0] * np.logical_not(background_mask)
        
        # at its peak absorption the tomour has double mu_a of the background
        volume[0, 0] += mu_a_background * gf.quadratic_profile_tumor(
            cfg['dx'], # [m]
            cfg['mcx_grid_size'], # [grid points]
            r_tumour, # [m]
            [
                cfg['mcx_domain_size'][0]/2,
                cfg['mcx_domain_size'][1]/2,
                cfg['mcx_domain_size'][2]/2
            ]
        )
        volume[0, 0] += mu_a_background * gf.quadratic_profile_tumor(
            cfg['dx'], # [m]
            cfg['mcx_grid_size'], # [grid points]
            r_tumour, # [m]
            [
                cfg['mcx_domain_size'][0]/2 + 0.004,
                cfg['mcx_domain_size'][1]/2,
                cfg['mcx_domain_size'][2]/2 + 0.004
            ]
        )
        tumour_mask = gf.sphere_mask(
            cfg['dx'], # [m]
            cfg['mcx_grid_size'], # [grid points]
            r_tumour, # [m]
            [
                cfg['mcx_domain_size'][0]/2,
                cfg['mcx_domain_size'][1]/2,
                cfg['mcx_domain_size'][2]/2
            ]
        )
        volume[0, 1][tumour_mask] = mu_s_tumour
        
        tumour_mask = gf.sphere_mask(
            cfg['dx'], # [m]
            cfg['mcx_grid_size'], # [grid points]
            r_tumour, # [m]
            [
                cfg['mcx_domain_size'][0]/2 + 0.004,
                cfg['mcx_domain_size'][1]/2,
                cfg['mcx_domain_size'][2]/2 + 0.004
            ]
        )
        volume[0, 1][tumour_mask] = mu_s_tumour
        
        # no proteins are in this experiment
        ReBphP_PCM_Pr_c = np.zeros((cfg['mcx_grid_size']), dtype=np.float32)
        ReBphP_PCM_Pfr_c = ReBphP_PCM_Pr_c.copy()
        
        return (volume, ReBphP_PCM_Pr_c, ReBphP_PCM_Pfr_c, background_mask)